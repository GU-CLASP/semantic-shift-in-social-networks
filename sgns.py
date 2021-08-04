import click
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from pathlib import Path
import pickle
import util
from semantic_change import angular_distance
import numpy as np
import random

log = util.create_logger(f"", 'train.log', True)

class StopTrainingException(Exception):
    pass

class AngularChange(CallbackAny2Vec):
    def __init__(self, stop_threshold, epochs, model_prefix):
        self.prev_wv = None
        self.prev_loss = None
        self.stop_threshold = stop_threshold
        self.max_epochs = epochs
        self.epoch = 0
        self.model_prefix = model_prefix
        with open(f'{self.model_prefix}.log', 'w') as f:
            f.write("alpha\tangular_change\ttrain_loss\n")

    def on_epoch_begin(self, model):
        self.cur_alpha = model.alpha - ((model.alpha - model.min_alpha) * float(self.epoch) / self.max_epochs)
        log.info(f"{self.model_prefix}: Starting epoch {self.epoch+1} of {self.max_epochs}.")
        log.info(f"{self.model_prefix}: Alpha: {self.cur_alpha:.06f}")
        self.prev_wv = model.wv.vectors.copy()

    def on_epoch_end(self, model):
        self.epoch += 1
        change = angular_distance(self.prev_wv, model.wv.vectors).mean()
        loss = model.get_latest_training_loss()
        loss_delta = loss - self.prev_loss if self.epoch > 1 else 0
        log.info(f"{self.model_prefix} epoch {self.epoch:02d}: Average angular change: {change:.06f} (threshold: {self.stop_threshold:0.6f}) | Epoch loss: {loss} | Delta loss: {loss_delta:+f}")
        with open(f'{self.model_prefix}.log', 'a') as f:
            f.write(f"{self.cur_alpha:0.6f}\t{change:0.6f}\t{loss}\n")
        if change < self.stop_threshold:
            log.info(f"{self.model_prefix}: Change threshold reached.")
            raise StopTrainingException
        elif self.epoch == self.max_epochs:
            log.info(f"{self.model_prefix}: Last epoch finished.")
            raise StopTrainingException
        self.prev_loss = loss
        model.running_training_loss = 0.0 # see: https://github.com/RaRe-Technologies/gensim/issues/2735


def corpus_counts(filename):
    examples, words = 0, 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            examples += 1
            words += len(line.rstrip('\n').split())
    return examples, words

class EpochSaver(CallbackAny2Vec):
    def __init__(self, model_prefix):
        self.model_prefix = model_prefix
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        output_path = f"{self.model_prefix}_E{self.epoch:02d}"
        log.info(f"Saving {output_path}")
        model.save(output_path)

@click.command()
@click.argument('token_counts_file')
@click.argument('corpus_file')
@click.argument('model_prefix')
@click.option('--init-model', default=None,
        help='Path to model to initalize vectors with.')
@click.option('--min-count', default=100,
        help='Minimum frequency for a word to appear in the vocabulary.')
@click.option('--ns-exponent', default=0.75)
@click.option('--alpha', default=0.025)
@click.option('--epochs', default=50)
@click.option("--stop-threshold", type=float, default=1e-4,
        help="Average angular change between epochs below which training will be stopped.")
@click.option('--save-checkpoints/--no-save-checkpoints', default=True)
@click.option('--n-threads', default=4)
def cli(token_counts_file, corpus_file, model_prefix, init_model, min_count,
        ns_exponent, alpha, epochs, stop_threshold, save_checkpoints, n_threads):

    log.info(f"Loading token counts.")
    token_counts = util.load_counts(token_counts_file)
    total_tokens = sum(token_counts.values())
    log.info(f"Unique tokens: {len(token_counts)}, total tokens: {total_tokens}.")

    model = Word2Vec(size=200, window=5, sg=1, negative=5,
            min_count=min_count, sample=1e-5, ns_exponent=ns_exponent,
            alpha=alpha, workers=n_threads)

    log.info(f"Bulinding vocab with min count {min_count}")
    model.build_vocab_from_freq(token_counts)

    corpus_examples, corpus_words =  corpus_counts(corpus_file)
    log.info(f"Corpus sentences: {corpus_examples}, corpus tokens: {corpus_words}.")

    # https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.intersect_word2vec_format.html
    # In-vocab words not in init_model are left alone (randomly initialized).
    # Out of vocab words in init_model are ignored.
    # lockf=1.0 means that imported word vectors are trained (=0.0 means frozen)
    if init_model is not None:
        model.intersect_word2vec_format(init_model, binary=False, lockf=1.0)

    callbacks = [AngularChange(stop_threshold, epochs, model_prefix)]
    if save_checkpoints:
        callbacks.append(EpochSaver(model_prefix))

    try:
        model.train(corpus_file=corpus_file, epochs=epochs, callbacks=callbacks, compute_loss=True,
                total_examples=corpus_examples, total_words=corpus_words)
    except (StopTrainingException, KeyboardInterrupt):
        pass

    log.info(f"Saving {model_prefix}.")
    model.save(f"{model_prefix}")
    model.wv.save_word2vec_format(f"{model_prefix}.w2v")

if __name__ == '__main__':
    cli()
