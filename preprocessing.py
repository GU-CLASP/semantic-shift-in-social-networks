import click
import util
from pathlib import Path
from multiprocessing import Pool
import csv
import os
import random
from collections import Counter

import re
import mistletoe
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English

class Preprocessor():
    def __init__(self, log):

        self.log = log
        banned_chars = open('banned_chars.txt', 'r').read().split('\n')

        # initialize preprocessing regexes
        self.remove_pre = re.compile(r"\^")
        self.remove_banned = re.compile(r"|".join(banned_chars))
        self.remove_post = re.compile(r"\s|\||\*")

        # initialize the tokenizer
        nlp = English()
        self.spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
        # handle special Reddit prefixes
        prefixes = nlp.Defaults.prefixes + (r"/u/", r"/r/", r"u/", r"r/")  # special Reddit
        prefixes_regex = spacy.util.compile_prefix_regex(prefixes)
        self.spacy_tokenizer.prefix_search = prefixes_regex.search

        self.total_tokens = 0
        self.total_comments = 0
        self.removed_tables = 0
        self.removed_code_blocks = 0
        self.removed_block_quotes = 0
        self.exceptions = 0

    def markdown2html(self, text):
        return mistletoe.markdown(text)

    def html2text(self, html):
        soup = BeautifulSoup(html, 'lxml')
        self.removed_tables       += len([t.extract() for t in soup('table')]) # remove tables
        self.removed_code_blocks  += len([t.extract() for t in soup('code')]) # remove code blocks
        self.removed_block_quotes += len([t.extract() for t in soup('blockquote')]) # remove block quotes
        return soup.text

    def tokenize(self, text):
        tokens = [token.text for token in self.spacy_tokenizer(text)]
        self.total_tokens += len(tokens)
        return tokens

    def print_debug(self, debug, s, i=None):
        if debug:
            if i:
                print(f"{'-'*28} {i:02d} {'-'*28}")
            print(s)

    def preprocess_comment(self, comment, debug=False):
        try:
            tokens = self.__pipeline(comment, debug)
            return tokens 
        except:
            if self.log is not None:
                self.log.exception(f"Problem parsing {comment['id']} from {comment['subreddit']}.")
            self.print_debug(True, comment['body'])
            self.exceptions += 1
            return None

    def __pipeline(self, comment, debug):
        """ Preprocessing pipeline. """

        self.total_comments += 1

        # Get the text from the comment
        text = comment['body']
        self.print_debug(debug, '='*60)
        self.print_debug(debug, f"{comment['subreddit']} {comment['link_id']} {comment['id']}")
        self.print_debug(debug, text, 1)

        # Remove some characters before parsing
        text = re.sub(self.remove_pre, ' ', text)
        text = re.sub(self.remove_banned, '', text)

        # For some reason the pushshift.io comments have HTML escapes in the markdown
        # we relpace them since the markdown parser doesn't recognie them.
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')

        # Render the markdown as an HTML string with mistletoe
        html = self.markdown2html(text)
        self.print_debug(debug, html, 2)

        # Extract the text from the HTML with BeautifulSoup
        text = self.html2text(html)
        self.print_debug(debug, text, 3)

        # Remove some characters post-parsing
        text = re.sub(self.remove_post, ' ', text)
        self.print_debug(debug, text, 4)

        # normalize 3+ dots to '...'
        text = re.sub(r"\.\.+(?=\.)", '..', text)

        self.print_debug(debug, text, 11)
        # Remove URLs
        text = re.sub(r"https?:\/\/[^\s]+", '', text)
        self.print_debug(debug, text, 12)

        # Remove extra whitespace
        text = re.sub(r"^\s+|\s+$|\s+(?=\s)", '', text)

        # Use the SpaCy tokenizer
        tokens = self.tokenize(text)
        self.print_debug(debug, ' '.join(tokens), 5)
        return tokens

    def log_summary(self):
        self.log.info(f"Comments:     {self.total_comments}")
        self.log.info(f"Avg. tokens:  {self.total_tokens/self.total_comments:0.2f}")
        self.log.info(f"Tables:       {self.removed_tables}")
        self.log.info(f"Code blocks:  {self.removed_code_blocks}")
        self.log.info(f"Block quotes: {self.removed_block_quotes}")
        self.log.info(f"Exceptions:   {self.exceptions}")


def get_min_token_count(log, tokenized_dir, sub):
    token_counts = []
    for year, month in util.iter_months():
        token_count = 0
        corpus_file = Path(tokenized_dir)/f"{sub}-{year}-{month:02d}.tokenized.csv"
        for comment in iter_text_rows(corpus_file):
            token_count += len(comment)
        token_counts.append(token_count)
        # log.info(f"Token count for {date_str} {sub:20.20}: {token_count:>12,}.")
    min_token_count = min(token_counts)
    log.info(f"Min token count for {sub:20.20}: {min_token_count:>12,}.")
    return min_token_count

def iter_text_rows(filename):
    with open(filename, 'r') as f:
        for row in f:
            yield row.lower().split(' ')

""" Commands are meant to be run in this order:
    - dedupe-subs
    - tokenize 
    - prep-train-files
    - make-vocab
"""

@click.group()
@click.argument("corpus_dir", type=str)
@click.option('--chosen-subs-file', type=str, default='chosen_subs.txt',
        help="File to save the list of chosen subs to.")
@click.option('--n-processes', type=int, default=1,
        help="Number of processes for multiprocessing actions.")
@click.option('--debug/--no-debug', default=False,
        help="Print debugging info.")
@click.pass_context
def cli(ctx, corpus_dir, chosen_subs_file, n_processes, debug):
    ctx.ensure_object(dict)
    ctx.obj['CORPUS_DIR'] = Path(corpus_dir) 
    ctx.obj['SUBS'] = util.get_subs(chosen_subs_file)
    ctx.obj['N_PROCESSES'] = n_processes
    ctx.obj['LOG'] = util.create_logger(f"preprocessing", 'preproc.log', debug)

def find_dupes(corpus_dir, sub, log):
    min_length = 50
    seen_hashes = {}
    dupe_ids = []
    sub_dir = Path(corpus_dir)/sub
    dupes_file = Path(sub_dir)/f"dupes.txt"
    for year, month in util.iter_months((2015,2017)):
        comments_file = Path(sub_dir)/f"{year}-{month:02d}.csv"
        log.info(f"Finding dupes in {comments_file}")
        for comment in util.iter_comments(comments_file):
            comment_len = len(comment['body']) 
            if comment_len < min_length:
                continue
            # only consider the last 50 characters for duplicates (catches most bot-filled forms)
            comment_hash = hash(comment['body'][-min_length:])
            if comment_hash in seen_hashes:
                dupe_ids.append(comment['id'])
                log.debug(f"Dupe id: {comment['id']} hash: {comment_hash:<20} author: {comment['author']}")
                log.debug(comment['body'])
                log.debug('-'*50)
                log.debug(seen_hashes[comment_hash])
            else:
                seen_hashes[comment_hash] = comment['body']
    log.info(f"Found {len(dupe_ids)} dupes for {sub}.")
    with open(dupes_file, 'w') as f:
        f.write('\n'.join(dupe_ids))

@cli.command()
@click.pass_context
def dedupe_subs(ctx):
    log = ctx.obj['LOG']
    with Pool(processes=ctx.obj['N_PROCESSES']) as p:
        args = [(ctx.obj['CORPUS_DIR'], sub, log) for sub in ctx.obj['SUBS']]
        p.starmap(find_dupes, args)

def tokenize_sub_month(corpus_dir, sub, year, month, log):
    pp = Preprocessor(log)
    infile = corpus_dir/sub/f"{year}-{month:02d}.csv"
    outfile = corpus_dir/sub/f"{year}-{month:02d}.tokenized.csv"
    dupes = [line.strip() for line in open(corpus_dir/sub/"dupes.txt").readlines()]
    log.info(f"Preprocessing {infile}")
    fo = open(outfile, 'w')
    with open(outfile, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'tokenized'])
        writer.writeheader()
        for comment in util.iter_comments(infile):
            if comment['id'] in dupes:
                continue
            tokens = pp.preprocess_comment(comment)
            if tokens:
                writer.writerow({'id': comment['id'], 'tokenized': ' '.join(tokens)})
    pp.log_summary()

@cli.command()
@click.pass_context
def tokenize(ctx):
    log = ctx.obj['LOG']
    subs = ctx.obj['SUBS']
    corpus_dir = ctx.obj['CORPUS_DIR']
    args = [(coprus_dir, sub, year, month, log) for sub in subs for year, month in util.iter_months()]
    with Pool(processes=ctx.obj['N_PROCESSES']) as p:
        p.starmap(tokenize_sub_month, args)

def make_vocab_sub(threshold, train_dir, vocab_dir, sub, years, log):
    token_counts = {year: Counter() for year in years}
    for year in years:
        train_file = train_dir/f"{sub}_{year}.txt"
        with open(train_file, 'r') as f:
            for line in f.readlines():
                tokens = line.strip().split()
                token_counts[year].update(tokens)
    vocab = [w for w in token_counts[years[0]] if all(token_counts[year][w] >= threshold for year in years)]
    vocab_counts = Counter({w: sum(token_counts[year][w] for year in years) for w in vocab}) # pseudo counts for gensim to do annealing
    vocab_file = vocab_dir/f"{sub}_vocab.txt"
    log.info(f"{sub} vocab_len: {len(vocab)}")
    util.save_counts(vocab_counts, vocab_file)

@cli.command()
@click.argument("vocab_dir", type=click.Path())
@click.option("--vocab-threshold", type=int, 
        prompt='Vocab threshold (min count per corpus)',
        help="Minmum occurance (in each year) to be included in the final vocab.")
@click.option('--years', multiple=True, default=[2015,2017])
@click.pass_context
def make_vocab(ctx, vocab_dir, vocab_threshold, years):
    log = ctx.obj['LOG']
    corpus_dir = ctx.obj['CORPUS_DIR']
    subs = ctx.obj['SUBS'] 
    vocab_dir = Path(vocab_dir)
    with Pool(processes=ctx.obj['N_PROCESSES']) as p:
        args = [(vocab_threshold, corpus_dir, vocab_dir, sub, years, log) for sub in subs]
        p.starmap(make_vocab_sub, args)

def count_tokens_sub_year(corpus_dir, sub, year, log):
    corpus_size = 0
    for month in range(1,13):
        tokenized_file = corpus_dir/sub/f"{year}-{month:02d}.tokenized.csv"
        for comment in util.iter_tokenized_comments(tokenized_file):
            corpus_size += len(comment['tokenized'])
    log.debug(f"{sub} {year} totals {corpus_size} tokens.")
    return corpus_size 

def prep_train_files_sub_year(corpus_dir, train_dir, sub, year, max_corpus_count, log):
    log.info(f"Prepping Gensim training file for {sub} {year}")
    all_comments = []
    for month in range(1,13):
        tokenized_file = corpus_dir/sub/f"{year}-{month:02d}.tokenized.csv"
        all_comments += list(util.iter_tokenized_comments(tokenized_file))
    random.shuffle(all_comments)
    sampled_tokens = 0
    train_file = train_dir/f"{sub}_{year}.txt"
    with open(train_file, 'w') as f:
        for comment in all_comments:
            tokens = comment['tokenized']
            sampled_tokens += len(tokens)
            f.write(' '.join(tokens) + '\n') 
            if sampled_tokens >= max_corpus_count:
                break

@cli.command()
@click.argument("train_dir", type=click.Path())
@click.option('--years', multiple=True, default=[2015,2017])
@click.pass_context
def prep_train_files(ctx, train_dir, years):
    train_dir = Path(train_dir)
    util.mkdir(train_dir)
    subs = ctx.obj['SUBS']
    corpus_dir = ctx.obj['CORPUS_DIR']
    log = ctx.obj['LOG']
    log.info(f"Computing corpus sizes...")
    with Pool(processes=ctx.obj['N_PROCESSES']) as p:
        args = [(corpus_dir, sub, year, log) for sub in subs for year in years]
        corpus_sizes = p.starmap(count_tokens_sub_year, args)
    smallest_corpus_size =  min(corpus_sizes)
    log.info(f"Standard corpus size (equal to smallest sub/year): {smallest_corpus_size}")
    with Pool(processes=ctx.obj['N_PROCESSES']) as p:
        args = [(corpus_dir, train_dir, sub, year, smallest_corpus_size, log) for sub in subs for year in years]
        p.starmap(prep_train_files_sub_year, args)

if __name__ == '__main__':
    cli(obj={})
