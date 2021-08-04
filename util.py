import os
import logging
import csv
import jsonlines
from tqdm import tqdm
from collections import Counter

reddit_columns = [
    'body', 'score_hidden', 'archived', 'name', 'author', 'author_flair_text',
    'downs', 'created_utc', 'subreddit_id', 'link_id', 'parent_id', 'score',
    'retrieved_on', 'controversiality', 'gilded', 'id', 'subreddit', 'ups',
    'distinguished', 'author_flair_css_class']

def iter_comments(file_path, include_body=True): 
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for comment in reader:
            if comment['body'] in ('[deleted]', '[removed]'):
                continue
            if not include_body:
                del comment['body']
            yield comment

def iter_tokenized_comments(file_path):
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for comment in reader:
            comment['tokenized'] = comment['tokenized'].lower().split(' ')
            yield comment

def iter_tokenized_comments_period(corpus_path, sub, period):
    for y,m in period:
        for comment in iter_tokenized_comments(f"{corpus_path}/{sub}-{y}-{m:02d}.tokenized.csv"):
            yield comment

def get_subs(chosen_subs_file, include_excluded=False):
    if not os.path.exists(chosen_subs_file):
        return []
    with open(chosen_subs_file, 'r') as f:
        subs = f.read().strip().split('\n')
    if not include_excluded:
        subs = [sub for sub in subs if not sub.startswith('#')]
    else:
        subs = [sub.lstrip('#') for sub in subs]
    return subs

def iter_months(years=range(2015,2018)):
     for year in years:
        for month in range(1,13): # 12 = 13-1 months in a year!
            yield year, month

def iter_time_periods(time_period):
    months = list(iter_months())
    month_chunks = [months[t:t+time_period] for t in range(0, len(months), time_period)]
    for chunk in month_chunks:
        yield TimePeriod(chunk)

class TimePeriod:
    def __init__(self, months):
        self.months = months
    def __iter__(self):
        return iter(self.months)
    def __str__(self):
        if len(self.months) == 1:
            y, m = self.months[0]
            return f"{y}-{m:02d}"
        else:
            (y0, m0), (yt, mt)  = self.months[0], self.months[-1]
            return f"{y0}-{m0:02d}--{yt}-{mt:02d}"

def create_logger(name, filename, debug):
    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s [%(levelname)-7s] %(message)s',
            datefmt='%m-%d %H:%M', filename=filename, filemode='a')
    console_log_level = logging.DEBUG if debug else logging.INFO
    console = logging.StreamHandler()
    console.setLevel(console_log_level)
    console.setFormatter(logging.Formatter('[%(levelname)-8s] %(message)s'))
    logger = logging.getLogger(name)
    logger.addHandler(console)
    return logger

def count_rows(file_path, format='csv'):
    with open(file_path) as f:
        if format == 'csv':
            reader = csv.reader(f)
        elif format == 'jsonlines':
            reader = jsonlines.Reader(f)
        elif format == 'txt':
            reader = f.readlines()
        for i, _ in enumerate(reader):
            pass
        return i 

def load_pairs(filename):
    with open(filename) as f:
        for line in f:
            w, c = line.strip().split('\t')
            c = float(c)
            yield w, c

def load_vocab(filename):
    itos = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            itos.append(line.strip())
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi

def mkdir(path):
    if os.path.exists(path):
        return False
    else:
        os.mkdir(path)
        return True

def save_counts(counts, filename, limit=None):
    """ Save a Counter object in plain text format."""
    with open(filename, 'w') as f:
        f.writelines((f"{w}\t{c}\n" for w,c in counts.most_common(limit)))

def load_counts(filename):
    """ Load plan text counts into a Counter object."""
    counts = Counter()
    with open(filename) as f:
        for line in f.readlines():
            w,c = line.rstrip('\n').split()
            counts[w] = int(c)
    return counts

def load_metric(filename):
    """ Load plan text counts into a Counter object."""
    metric = {}
    with open(filename) as f:
        for line in f.readlines():
            w,v = line.rstrip('\n').split()
            metric[w] = float(v)
    return metric 

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def chunk_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


