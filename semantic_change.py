import click
import numpy as np
import pandas as pd
from pathlib import Path
import util
from gensim.models import KeyedVectors
from multiprocessing import Pool
import csv

def cos_sim(v1, v2):
    if not v1.shape == v2.shape:
        raise ValueError
    if v1.ndim == 1:
        axis = 0
    elif v1.ndim == 2:
        axis = 1
    else:
        raise ValueError
    val = (v1 * v2).sum(axis=axis) / (np.linalg.norm(v1, axis=axis) * np.linalg.norm(v2, axis=axis))
    # sometimes identical vectors produce 1.0000001. floating point problem?
    if v1.ndim == 1:
        return min(val,1)
    else:
        return np.minimum(val, np.ones_like(val))

def angular_distance(v1, v2):
    return np.arccos(cos_sim(v1,v2)) / np.pi

def cosine_change(wv1, wv2, vocab=None):
    """
    Takes two instances of gensim KeyedVectors objects and computes semantic
    distance scores for words in both vocabularies. Semantic distance is
    defined as the angular distance which fall in [0, 1]. If `vocab` is
    provided, it must be a subset of the intersection of the two
    KeyedVector vocabs.
    """
    # vocab is provided; restrict vectors to provided vocab
    if vocab:
        common_vocab = vocab
        vecs1 = np.stack([wv1[w] for w in common_vocab])
        vecs2 = np.stack([wv2[w] for w in common_vocab])
    # vocabs are different; restrict vectors to intersection
    elif wv1.index2word != wv2.index2word:  
        common_vocab = [w for w in wv1.index2word if w in wv2.index2word]
        vecs1 = np.stack([wv1[w] for w in common_vocab])
        vecs2 = np.stack([wv2[w] for w in common_vocab])
    # vocabs are the same; use full vector matrices
    else:
        common_vocab = wv1.index2word
        vecs1 = wv1.vectors
        vecs2 = wv2.vectors
    distances = angular_distance(vecs1, vecs2)
    return dict(zip(common_vocab, distances))

cos_diff = lambda x,y: 1 - cos_sim(x,y)

def rectified_change_score(measured_change, control_samples):
    control_samples = np.array(control_samples)
    N = len(control_samples)
    sample_mean =  np.mean(control_samples)
    sample_std = np.std(control_samples, ddof=1) # ddof=1 = Bessel's correction
    return (measured_change - sample_mean) / (sample_std * np.sqrt(1 + (1/N)))

def neighborhood_change(wv_t1, wv_t2, vocab, neighborhood_size=25):
    # Change metric using angular distance between neighborgood vectors.
    sims_t1 = cos_sim(np.stack([wv_t1[w] for w in vocab]))
    sims_t2 = cos_sim(np.stack([wv_t2[w] for w in vocab]))
    # Sort word indices by similarity 
    sim_sort_t1 = np.argsort(sims_t1, axis=1)
    sim_sort_t2 = np.argsort(sims_t2, axis=1)
    change = {}
    for i,w in enumerate(vocab):
        # Create array of closest neighbors from both times
        neighbors_t1 = sim_sort_t1[i][-neighborhood_size:]
        neighbors_t2 = sim_sort_t2[i][-neighborhood_size:]
        neighbors = np.unique(np.concatenate([neighbors_t1, neighbors_t2]))
        # Delete the word itself from the neighbors list
        neighbors = np.delete(neighbors, np.where(neighbors == i))
        # Create the meta vectors out of similarity scores with neighbors
        meta_t1 = sims_t1[i][neighbors]
        meta_t2 = sims_t2[i][neighbors]
        # Measure distance between the meta vectors
        change[w] = angular_distance(meta_t1, meta_t2)
    return change


@click.command()
@click.argument("model1")
@click.argument("model2")
@click.argument("change_file")
def cli(model1, model2, change_file):

    # Compute change on the genuine condition 
    wv1= KeyedVectors.load_word2vec_format(model1)
    wv2= KeyedVectors.load_word2vec_format(model2)
    change = cosine_change(wv1, wv2)
    with open(change_file, 'w') as f:
        f.writelines((f"{w}\t{delta}\n" for w,delta in change.items()))

if __name__ == '__main__':
    cli()


