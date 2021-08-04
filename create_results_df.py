####### BIG OLD REGRESSION DATAFRAME
import pickle
import pandas as pd
import numpy as np
import util

## COMMUNITY
subs = list(util.get_subs('chosen_subs.txt'))
cdf = pd.DataFrame(data=None, index=subs)
# Pre-computed users and post counts for both years
users15 = {sub: pd.Series(util.load_counts(f'model/community/{sub}_2015_user_comment_counts.txt')) for sub in subs}
users17 = {sub: pd.Series(util.load_counts(f'model/community/{sub}_2017_user_comment_counts.txt')) for sub in subs}

def active(a):
    """ Active users have at least 10 posts"""
    return a[a>=10]

def jaccard(a, b):
    """ Measures the overlap between two lists of users"""
    a_and_b = (a.index.intersection(b.index))
    return len(a_and_b) / (len(a) + len(b) + len(a_and_b))

cdf['size_active_15']       = [len(active(users15[sub])) for sub in cdf.index]
cdf['size_total_15']       = [len(users15[sub]) for sub in cdf.index]
cdf['stability_active']  = [jaccard(active(users15[sub]), active(users17[sub])) for sub in cdf.index]
cdf['mean_posts_active_15'] = [active(users15[sub]).mean() for sub in cdf.index]

clustering_full = {sub: pickle.load(open(f'model/community/{sub}_2015_clustering_full.pickle', 'rb')) for sub in subs}
cdf['clustering_full_15'] =  pd.Series({sub: sum(clustering_full[sub].values()) /  len(clustering_full[sub]) for sub in subs})

clustering_weak = {sub: pickle.load(open(f'model/community/{sub}_2015_clustering_weak.pickle', 'rb')) for sub in subs}
cdf['clustering_weak_15'] = pd.Series({sub: sum(clustering_weak[sub].values()) /  len(clustering_weak[sub]) for sub in subs})

clustering_strong = {sub: pickle.load(open(f'model/community/{sub}_2015_clustering_strong_corrected.pickle', 'rb')) for sub in subs}
cdf['clustering_strong_15'] = pd.Series({sub: sum(clustering_strong[sub].values()) /  len(clustering_strong[sub]) for sub in subs})

## WORD (generic lex)
wdf = pd.Series(util.load_counts('model/vocab/General_vocab.txt')).to_frame('generic_count')
# cosine change (rectified)
wdf['generic_cosine_change'] = pd.Series(util.load_metric(f'analysis/change_scores/General_rectified.txt'))
# rectified cosine change
control = pd.concat([pd.Series(
    util.load_metric(f'analysis/change_scores/General_control{i+1}.txt'))
    for i in range(10)], axis=1)
wdf['generic_control_change_mean'] = control.mean(axis=1)
wdf['generic_control_change_std'] = control.std(axis=1, ddof=1) # ddof=1 = bessell's correction
wdf['generic_rectified_change'] = (wdf['generic_cosine_change'] - wdf['generic_control_change_mean']) / (wdf['generic_control_change_std'] * np.sqrt(1 + 1/10))

## WORD x COMMUNITY
wcdf = pd.DataFrame()
for sub in subs:
    # Token counts (2015 + 2017; min 50 in each year)
    freq_15 = pd.read_csv(f'analysis/token_freq/{sub}_15.csv').set_index('word')
    freq_17 = pd.read_csv(f'analysis/token_freq/{sub}_17.csv').set_index('word')
    sub_df = freq_15.merge(freq_17, how='outer', left_index=True, right_index=True)
    # index to word
    sub_df['community'] = sub
    # cosine change (rectified)
    sub_df['cosine_change'] = pd.Series(util.load_metric(f'analysis/change_scores/{sub}_rectified.txt'))
    sub_df = sub_df.dropna() # remove words we didn't calculate cosine_change for. (counts include empty string as a token for some reason)
    # rectified cosine change
    control = pd.concat([pd.Series(
        util.load_metric(f'analysis/change_scores/{sub}_control{i+1}.txt'))
        for i in range(10)], axis=1)
    sub_df['control_change_mean'] = control.mean(axis=1)
    sub_df['control_change_std'] = control.std(axis=1, ddof=1) # ddof=1 = bessell's correction
    # Index to community+word 
    sub_df = sub_df.set_index(['community', sub_df.index])
    wcdf = wcdf.append(sub_df)

wcdf['rectified_change'] = (wcdf['cosine_change'] - wcdf['control_change_mean']) / (wcdf['control_change_std'] * np.sqrt(1 + 1/10))

# ALL TOGETHER NOW
df = wcdf.merge(wdf, how='left', left_on='word', right_index=True)
df = df.merge(cdf, how='left', left_on='community', right_index=True)

# Export dataframe
df.to_csv('analysis/community_word.csv', index=True)

# Small DF with means
cdf['mean_rectified_change'] =  wcdf.groupby('community')[['rectified_change']].mean()
cdf.to_csv('analysis/community.csv', index=True)


