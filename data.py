import argparse
import os
import util
from multiprocessing import Pool
import csv
from collections import Counter
import pandas as pd
import random

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=[
    'choose-subs',       # Randomly select some number of subreddits 
    'query-subs',        # Query data for the chosen subreddits
    'query-sample',      # Query a uniform monthly sample of reddit comments
    'split-subs',        # Split raw data into subreddit corpora
    ],  help="Data collection step")
parser.add_argument('credentials', type=str,
        help="Path to Google application credentials file (json)"
             "See: https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python"),
parser.add_argument('--num-subs', type=int, default=1,
        help="Number of subsreddits to randomly select.")
parser.add_argument('--sample-size', type=int, default=100000,
        help="Number of commments per month to sample (query-sample).")
parser.add_argument('--min-comments', type=int, default=15000,
        help="Comment minimum (monthly) for subreddit consideration.")
parser.add_argument('--comment-counts-dir', type=str, default='comment_counts/',
        help="Directory to save subreddit comment counts by year")
parser.add_argument('--chosen-subs-file', type=str, default='chosen_subs.txt',
        help="File to save the list of chosen subs to.")
parser.add_argument('--corpus-dir', type=str, default='data/subreddit_comments/',
        help="Directory for the corpus by subreddit")
parser.add_argument('--raw-corpus-dir', type=str, default='data/raw/',
        help="Directory for the raw files downloaded from Google Storage")
parser.add_argument('--google-project', type=str, default="")
parser.add_argument('--google-bucket-name', type=str, default="")
parser.add_argument('--google-directory-name', type=str, default="reddit_sample")
parser.add_argument('--google-dataset-id', type=str, default="reddit_sample")
parser.add_argument('--debug', action='store_true', default=False,
        help="Print debugging info.")
parser.add_argument('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")

def remove_nul(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = data.replace('\00', '')
    with open(filename, 'w') as f:
        f.write(data)

def split_month(year, month, raw_dir, corpus_dir, subs, log):
    i = 0
    counts = Counter()
    out_files = {sub: open(os.path.join(corpus_dir,sub,f'{year}-{month:02d}.csv'), 'w') for sub in subs}
    writers = {sub: csv.DictWriter(f, util.reddit_columns) for sub, f in out_files.items()}
    for writer in writers.values():
        writer.writeheader()
    while True:
        in_file = os.path.join(raw_dir, f'{year}-{month:02d}-{i:012d}.csv')
        i += 1
        if not os.path.exists(in_file):
            break
        remove_nul(in_file)
        log.info(f"Reading {in_file}.")
        for comment in util.iter_comments(in_file):
            sub = comment['subreddit']
            counts[sub] += 1
            if not sub in subs:
                continue
            writers[sub].writerow(comment)
    for f in out_files.values():
        f.close()
    counts = '\n'.join([f'\t{sub:<25.25} {count}' for sub,count in counts.items()])
    log.info(f"Comment counts for {year}-{month:02d}:\n{counts}")

def get_comment_counts_month(year, month, min_posts, bq_client, bqstorage_client):
    query_string = f"""
    SELECT subreddit, post_count
    FROM (
      SELECT subreddit, COUNT(id) as post_count
      FROM `fh-bigquery.reddit_comments.{year}_{month:02d}`
      GROUP BY subreddit
    )
    WHERE post_count > {min_posts}
    ORDER BY post_count DESC
    """

    df = (
	bq_client.query(query_string)
	.result()
	.to_dataframe(bqstorage_client=bqstorage_client)
    )
    df = df.set_index('subreddit')
    return df

if __name__ == '__main__':

    args = parser.parse_args()
    log = util.create_logger(f"{args.command}", 'data.log', args.debug)

    # Make Google API clients.
    # You have to run `gcloud init` before this will work.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials
    credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    bq = bigquery.Client(credentials=credentials, project=project_id)
    bq_storage = bigquery_storage_v1beta1.BigQueryStorageClient(credentials=credentials)

    subs = util.get_subs(args.chosen_subs_file, include_excluded=True)  # get subs chosen so far, if any

    if args.command == 'choose-subs':

        log.info("Creating comment count CSVs.")
        if not os.path.exists(args.comment_counts_dir):
            os.mkdir(args.comment_counts_dir)
        for year, month in util.iter_months():
            csv_file = os.path.join(args.comment_counts_dir, f"{year}_{month:02d}.csv")
            if os.path.exists(csv_file):
                log.info(f"Skipping {csv_file} (already exists).")
                continue
            log.info(f"Quering comment counts for {csv_file}.")
            df = get_comment_counts_month(year, month, args.min_comments, bq, bq_storage)
            df.to_csv(csv_file)

        log.info("Choosing subreddits at random.")
        candidate_subs = []
        for year, month in util.iter_months():
            csv_file = os.path.join(args.comment_counts_dir, f"{year}_{month:02d}.csv")
            df = pd.read_csv(csv_file, index_col='subreddit')
            candidate_subs.append(set(df.index))
        viable_subs = set.intersection(*candidate_subs)  # subs that have >min_comments for every month
        viable_subs = viable_subs.difference(subs)  # don't select subs that have already been added
        log.info(f"Choosing {args.num_subs} of {len(viable_subs)} subs with at least {args.min_comments} "
                     "comments each month.")
        new_subs = random.sample(viable_subs, args.num_subs)
        with open(args.chosen_subs_file, 'a') as f:
            f.write('\n'.join(new_subs))
        log.info(f"Added new subs: {', '.join(new_subs)}")

    if args.command == 'query-subs':

        for year, month in util.iter_months():

            # Query comments for the selecetd subs and save to a BQ table
            table_id = f"{args.google_project}.{args.google_dataset_id}.{year}_{month:02d}"
            job_config = bigquery.QueryJobConfig(
                allow_large_results=True,
                destination=table_id,
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE'
            )
            query_string = f"""
              SELECT *
              FROM `fh-bigquery.reddit_comments.{year}_{month:02d}`
              WHERE subreddit in ({', '.join([f"'{sub}'" for sub in subs])})
            """
            # Start the query, passing in the extra configuration.
            query_job = bq.query(query_string, job_config=job_config)  # Make an API request.
            query_job.result()  # Wait for the job to complete.
            log.info(f"Comment query results saved to {table_id}.")
    
            # Save the data to Google Storage
            table_id = f"{year}_{month:02d}"
            destination_uri = "gs://{}/{}/{}".format(
                args.google_bucket_name, args.google_directory_name, f"{year}-{month:02d}-*.csv"
            )
            dataset_ref = bq.dataset(args.google_dataset_id, project=args.google_project)
            table_ref = dataset_ref.table(table_id)
            extract_job = bq.extract_table(
                table_ref,
                destination_uri,
                location="US",  # Location must match that of the source table.
            )  # API request
            extract_job.result()  # Waits for job to complete.
            log.info(f"Exported {args.google_project}:{args.google_dataset_id}.{table_id} to {destination_uri}")

    if args.command == 'query-sample':

        for year, month in util.iter_months():

            # Query comments for the selecetd subs and save to a BQ table
            table_id = f"{args.google_project}.{args.google_dataset_id}.{year}_{month:02d}"
            job_config = bigquery.QueryJobConfig(
                allow_large_results=True,
                destination=table_id,
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE'
            )
            query_string = f"""
              SELECT *
              FROM `fh-bigquery.reddit_comments.{year}_{month:02d}`
              WHERE RAND() < {args.sample_size}/(SELECT count(*) FROM `fh-bigquery.reddit_comments.{year}_{month:02d}`)
            """
            # Start the query, passing in the extra configuration.
            query_job = bq.query(query_string, job_config=job_config)  # Make an API request.
            query_job.result()  # Wait for the job to complete.
            log.info(f"Comment query results saved to {table_id}.")

            # Save the data to Google Storage
            table_id = f"{year}_{month:02d}"
            destination_uri = "gs://{}/{}/{}".format(
                args.google_bucket_name, args.google_directory_name, f"{year}-{month:02d}-*.csv"
            )
            dataset_ref = bq.dataset(args.google_dataset_id, project=args.google_project)
            table_ref = dataset_ref.table(table_id)
            extract_job = bq.extract_table(
                table_ref,
                destination_uri,
                location="US",  # Location must match that of the source table.
            )  # API request
            extract_job.result()  # Waits for job to complete.
            log.info(f"Exported {args.google_project}:{args.google_dataset_id}.{table_id} to {destination_uri}")


    if args.command == 'split-subs':
        """
        First you have to do something like: gsutil -m cp -r gs://bill-gu-research/subreddit_comments ~/data/raw
        to copy the raw corpus from wherever you had bigquery save it 
        """

        for sub in subs:
            sub_dir = os.path.join(args.corpus_dir, sub)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

        def __split_month(yearmonth):
            year, month = yearmonth
            split_month(year, month, args.raw_corpus_dir, args.corpus_dir, subs, log)

        pool = Pool(processes=args.n_processes)
        pool.map(__split_month, list(util.iter_months()))
