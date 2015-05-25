#!/usr/bin/env python
from __future__ import division
from __future__ import unicode_literals

import os
from os import path
import shutil
import codecs
import random
import time
import gzip

import plac
import cProfile
import pstats

import spacy.util
from spacy.en import English
from spacy.en.pos import POS_TEMPLATES, POS_TAGS, setup_model_dir

from spacy.syntax.parser import GreedyParser
from spacy.syntax.util import Config
from spacy.scorer import Scorer
from spacy.syntax.conll import GoldParse



def _parse_line(line):
    pieces = line.split()
    id_ = int(pieces[0]) - 1
    word = pieces[1]
    pos = pieces[3]
    head_idx = int(pieces[6]) - 1
    label = pieces[7]
    if label.lower() == 'root':
        label = 'ROOT'

    if head_idx < 0:
        head_idx = id_
    return word, pos, head_idx, label


def read_gold(loc):
    sents = []
    for sent_str in codecs.open(loc, 'r', 'utf8').read().strip().split('\n\n'):
        ids = []
        words = []
        heads = []
        labels = []
        tags = []
        ner = []
        lines = sent_str.strip().split('\n')
        for i, line in enumerate(lines):
            word, pos, head, label = _parse_line(line)
            words.append(word)
            ids.append(i)
            heads.append(head)
            labels.append(label)
            tags.append(pos)
            ner.append('O')
        sents.append((None, None, (ids, words, tags, heads, labels, ner)))
    return sents
        

def train(Language, sentences, model_dir, n_iter=15, feat_set=u'basic', seed=0,
          gold_preproc=False, force_gold=False):
    dep_model_dir = path.join(model_dir, 'deps')
    pos_model_dir = path.join(model_dir, 'pos')
    if path.exists(dep_model_dir):
        shutil.rmtree(dep_model_dir)
    if path.exists(pos_model_dir):
        shutil.rmtree(pos_model_dir)
    os.mkdir(dep_model_dir)
    os.mkdir(pos_model_dir)
    setup_model_dir(sorted(POS_TAGS.keys()), POS_TAGS, POS_TEMPLATES,
                    pos_model_dir)

    Config.write(dep_model_dir, 'config', features=feat_set, seed=seed,
                 labels=Language.ParserTransitionSystem.get_labels(sentences))

    nlp = Language(data_dir=model_dir)
    
    for itn in range(n_iter):
        scorer = Scorer()
        for _1, _2, (_3, words, tags, heads, labels, ner) in sentences:
            tokens = nlp.tokenizer.tokens_from_list(words)
            assert len(words) == len(tokens) == len(heads)
            string_indices = [token.idx for token in tokens]
            heads = [string_indices[head] for head in heads]
            annot_tuples = (string_indices, words, tags, heads, labels, ner)
            nlp.tagger.tag_from_strings(tokens, tags)
            # Eval before train
            nlp.parser(tokens)
            scorer.score(tokens, GoldParse(tokens, annot_tuples), verbose=False)
            # Make fresh tokens, and train
            tokens = nlp.tokenizer.tokens_from_list(words)
            nlp.tagger.tag_from_strings(tokens, tags)
            try:
                nlp.parser.train(tokens, GoldParse(tokens, annot_tuples))
            except AssertionError:
                continue
	print '%d:\t%.3f\t%.3f' % (itn, scorer.uas, scorer.las)
        random.shuffle(sentences)
    nlp.parser.model.end_training()
    nlp.tagger.model.end_training()
    nlp.vocab.strings.dump(path.join(model_dir, 'vocab', 'strings.txt'))



def _map_indices_to_tokens(ids, heads):
    mapped = []
    for head in heads:
        if head not in ids:
            mapped.append(None)
        else:
            mapped.append(ids.index(head))
    return mapped


def main(train_loc, model_dir):
    train_sents = read_gold(train_loc)
    train(English, train_sents, model_dir, gold_preproc=False, force_gold=False)
    

if __name__ == '__main__':
    plac.call(main)
