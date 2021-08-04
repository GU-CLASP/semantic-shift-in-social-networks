# Semantic shift in social networks 

This repository contains the research code for the paper [Semantic shift in social networks](https://aclanthology.org/2021.starsem-1.3).
The paper explores the relationship beetween various community characteristics 
and the amount of semantic change that takes place in how the community uses words. 

I hope the code is useful for people trying to replicate our results or try something new!
If there's anything missing or if you have questions, feel free to email me or even make a pull request.

I used a nix environment to run this, which can be found in [/nix](./nix/), but most of the
dependencies should als be in [requirements.txt](./requirements.txt).

## Downloading data

We use comments from different communities on Reddit (subreddits) to investigate semantic shift at the community level. 
First, we select 46 subreddits with enough data between 2015 and 2017.
The subreddits used in the paper can be found in [chosen_subs.txt](./chosen_subs.txt).
You can also choose new subreddits using the same criteria we used by running

```
python3 -m data path/to/google-api-credentials.json choose-subs --google-project project_name --google-bucket bucket_name 
```

We used [pushshift.io](https://pushshift.io)'s Google BigQuery tables of Reddit comments to filter and download data,
so you need a Google Cloud account to run the commands in [data.py](./data.py). 

The other commands in `data.py` take a similar syntax. `query-sample` downloads a sample of comments (balanced by subreddit) 
to a provided temporary Google Storage bucket, and `split-subs` splits the results of `query-sample` into subreddit/year 
corpora. `query-subs` is like `query-sample` but downloads the full corpus of comments for the 2015 and 2017. This is 
used to build the social network graphs.

## Preprocessing

Preprocessing commands are meant to be run in this order:

	1. `dedupe-subs` -- remove duplicate comments from each subreddit's corpus (including "template" comments made by bots)
 	2. `tokenize` -- tokenize using SpaCy, remove markdown formatting, "banned" characters, etc.
  	3. `prep-train-files` -- normalize each subreddit corpus to the same number of tokens
  	4. `make-vocab` -- create a vocab for each subreddit corpus.

## Social network

Computing the social network and clustering coefficient is done in [network.py](./network.py). Three variants are
computed: strong, weak and full. *Strong* only considers edges with >1 interaction, *weak* only considers edges with
1 interaction and *full* considers both. They are all very closely correlated so we only used *full* in the paper.

## Semantic change 

[train_change_models.sh](./train_change_models.sh) trains SGNS models for both the genuine and shuffled conditions (including shuffling the corpora 10 times)
and computing naive and rectified semantic change. Naturally this step is the most time consuming.

## Results 

Everything else is taken care of in [create_results_df.py](create_results_df.py), including computing the rest of the community features.
