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
from spacy.syntax.parser import OracleError
from spacy.syntax.util import Config


def read_tokenized_gold(file_):
    """Read a standard CoNLL/MALT-style format"""
    sents = []
    for sent_str in file_.read().strip().split('\n\n'):
        ids = []
        words = []
        heads = []
        labels = []
        tags = []
        for i, line in enumerate(sent_str.split('\n')):
            id_, word, pos_string, head_idx, label = _parse_line(line)
            words.append(word)
            if head_idx == -1:
                head_idx = i
            ids.append(id_)
            heads.append(head_idx)
            labels.append(label)
            tags.append(pos_string)
        text = ' '.join(words)
        sents.append((text, [words], ids, words, tags, heads, labels))
    return sents


def _parse_line(line):
    pieces = line.split()
    id_ = int(pieces[0])
    word = pieces[1]
    pos = pieces[3]
    head_idx = int(pieces[6])
    label = pieces[7]
    return id_, word, pos, head_idx, label

        
def iter_data(paragraphs, tokenizer, gold_preproc=False):
    for raw, tokenized, ids, words, tags, heads, labels in paragraphs:
        assert len(words) == len(heads)
        for words in tokenized:
            sent_ids = ids[:len(words)]
            sent_tags = tags[:len(words)]
            sent_heads = heads[:len(words)]
            sent_labels = labels[:len(words)]
            sent_heads = _map_indices_to_tokens(sent_ids, sent_heads)
            tokens = tokenizer.tokens_from_list(words)
            yield tokens, sent_tags, sent_heads, sent_labels
            ids = ids[len(words):]
            tags = tags[len(words):]
            heads = heads[len(words):]
            labels = labels[len(words):]


def get_labels(sents):
    left_labels = set()
    right_labels = set()
    for raw, tokenized, ids, words, tags, heads, labels in sents:
        for child, (head, label) in enumerate(zip(heads, labels)):
            if head > child:
                left_labels.add(label)
            elif head < child:
                right_labels.add(label)
    print left_labels
    print right_labels
    return list(sorted(left_labels)), list(sorted(right_labels))


def train(Language, paragraphs, model_dir, n_iter=15, feat_set=u'basic', seed=0,
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

    left_labels, right_labels = get_labels(paragraphs)
    Config.write(dep_model_dir, 'config', features=feat_set, seed=seed,
                 left_labels=left_labels, right_labels=right_labels)

    nlp = Language(data_dir=model_dir)
    
    for itn in range(n_iter):
        heads_corr = 0
        pos_corr = 0
        n_tokens = 0
        for tokens, tag_strs, heads, labels in iter_data(paragraphs, nlp.tokenizer,
                                                         gold_preproc=gold_preproc):
            #nlp.tagger(tokens)
            nlp.tagger.tag_from_strings(tokens, tag_strs)
            try:
                heads_corr += nlp.parser.train_sent(tokens, heads, labels, force_gold=force_gold)
            except OracleError:
                continue
            #pos_corr += nlp.tagger.train(tokens, tag_strs)
            n_tokens += len(tokens)
        acc = float(heads_corr) / n_tokens
        pos_acc = float(pos_corr) / n_tokens
        print '%d: ' % itn, '%.3f' % acc, '%.3f' % pos_acc
        random.shuffle(paragraphs)
    nlp.parser.model.end_training()
    nlp.tagger.model.end_training()
    return acc


def _map_indices_to_tokens(ids, heads):
    mapped = []
    for head in heads:
        if head not in ids:
            mapped.append(None)
        else:
            mapped.append(ids.index(head))
    return mapped


def main(train_loc, model_dir):
    with codecs.open(train_loc, 'r', 'utf8') as file_:
        train_sents = read_tokenized_gold(file_)
    train(English, train_sents, model_dir, gold_preproc=False, force_gold=False)
    

if __name__ == '__main__':
    plac.call(main)
