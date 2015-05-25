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
import sys

import plac
import cProfile
import pstats

import spacy.util
from spacy.en import English
from spacy.en.pos import POS_TEMPLATES, POS_TAGS, setup_model_dir

from spacy.syntax.parser import GreedyParser
from spacy.syntax.parser import OracleError
from spacy.syntax.util import Config


def is_punct_label(label):
    return label.lower() == 'p' or label.lower() == 'punct'


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
 

def parse_and_evaluate(Language, model_dir, dev_loc, out_loc):
    global loss
    nlp = Language(data_dir=model_dir)
    n_corr = 0
    ln_corr = 0
    pos_corr = 0
    n_tokens = 0
    total = 0
    skipped = 0
    loss = 0
    sentences = read_gold(dev_loc)
    out_file = codecs.open(out_loc, 'w', 'utf8')
    for _1, _2, (_3, words, tags, heads, labels, ner) in sentences:
        tokens = nlp.tokenizer.tokens_from_list(words)
        assert len(words) == len(tokens) == len(heads)
        nlp.tagger.tag_from_strings(tokens, tags)
        nlp.parser(tokens)
 
        for i, token in enumerate(tokens):
            pos_corr += token.tag_ == tags[i]
            fmt = '{i}\t{orth}\t{lemma}\t{pos}\t{pos}\t_\t{head}\t{label}\t_\n'
            out_file.write(
                fmt.format(i=i+1, orth=token.orth_, lemma=token.lemma_,
                           pos=token.tag_,
                           head=token.head.i+1 if token.head is not token else 0,
                           label=token.dep_.lower())
                )

            n_tokens += 1
            if is_punct_label(labels[i]):
                continue
            
            n_corr += token.head.i == heads[i]
            ln_corr += token.head.i == heads[i] and token.dep_.lower() == labels[i].lower()
            
            total += 1
        out_file.write('\n')
    out_file.close()
    print >> sys.stderr, pos_corr / n_tokens
    return float(n_corr) / total, float(ln_corr) / total


def parse(Language, model_dir, input_loc, out_loc):
    nlp = Language(data_dir=model_dir)
    out_file = codecs.open(out_loc, 'w', 'utf8')
    fmt = '{i}\t{orth}\t{lemma}\t{pos}\t{pos}\t_\t{head}\t{label}\t_\n'
    _ = nlp.parser
    _ = nlp.tagger    
    n = 0
    for sent_str in codecs.open(input_loc, 'r', 'utf8').read().strip().split('\n\n'):
        words = []
        tags = []
        for tok_str in sent_str.split('\n'):
            pieces = tok_str.split()
            words.append(pieces[1])
            tags.append(pieces[3])
        tokens = nlp.tokenizer.tokens_from_list(words)
        nlp.tagger.tag_from_strings(tokens, tags)
        nlp.parser(tokens)
        for i, token in enumerate(tokens):
            out_file.write(
                fmt.format(
                    i=i+1,
                    orth=token.orth_,
                    lemma=token.lemma_,
                    pos=token.tag_,
                    head=token.head.i+1 if token.head is not token else 0,
                    label=token.dep_.lower())
            )
        out_file.write('\n')
        n += len(tokens)
    out_file.close()
    print n


@plac.annotations(
    score_parse=("Do evaluation", "flag", "v", bool)
)
def main(model_dir, dev_loc, out_loc, score_parse=False):
    if score_parse:
        print >> sys.stderr, parse_and_evaluate(English, model_dir, dev_loc, out_loc)
    else:
        parse(English, model_dir, dev_loc, out_loc)
    

if __name__ == '__main__':
    plac.call(main)
