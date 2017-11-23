from __future__ import absolute_import

import os
import re
import string
import numpy as np

from itertools import chain

def load_task(in_file, BIO=False, SBIEO=False):
    data = get_stories(in_file, BIO=BIO, SBIEO=SBIEO)
    return data

def tokenize(sent):
    return sent.split(' ')

def convert2BIO(data):
    for document in data:
        for sentence in document:
            for i, (word, pos, chunk, ner) in enumerate(sentence):
                is_first = i == 0
                is_last = i == len(sentence) - 1
                t = ner[2:]
                if ner[:2] == 'I-':
                    if is_first or sentence[i - 1][3] == 'O':
                        sentence[i] = ((word, pos, chunk, 'B-' + ner[2:]))
                    else:
                        if sentence[i - 1][3][2:] != ner[2:]:
                            sentence[i] = ((word, pos, chunk, 'B-' + ner[2:]))
    return data

def convert2SBIEO(data):
    data = convert2BIO(data)
    for document in data:
        for sentence in document:
            for i, (word, pos, chunk, ner) in enumerate(sentence):
                is_first = i == 0
                is_last = i == len(sentence) - 1
                current_tag = ner[:2]
                next_tag = '' if is_last else sentence[i + 1][3][:2]
                new_tag = None
                if current_tag == 'B-' and (next_tag in ['B-', 'O', '']):
                    new_tag = 'S-' + ner[2:]
                elif current_tag == 'I-' and (next_tag in ['B-', 'O', '']):
                    new_tag = 'E-' + ner[2:]
                else:
                    new_tag = ner
                sentence[i] = ((word, pos, chunk, new_tag))
    return data

def parse_stories(lines):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    document = []
    sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            # empty line
            if len(sentence) > 0:
                document.append(sentence)
                sentence = []
            continue
        elif line == '-DOCSTART- -X- O O' or line == '-DOCSTART- -X- -X- O':
            # new document line
            if len(document) > 0:
                data.append(document)
                document = []
            continue
        attrs = line.split(' ')
        word = attrs[0]
        pos = attrs[1]
        chunk = attrs[2]
        ner = attrs[3]
        # if I2B and ner[:2] == 'I-':
        #     if len(sentence) == 0 or sentence[-1][3] == 'O':
        #         ner = 'B-' + ner[2:]
        sentence.append((word, pos, chunk, ner))
    
    if len(sentence) > 0:
        document.append(sentence)
    if len(document) > 0:
        data.append(document)
    return data

def get_stories(f, BIO=False, SBIEO=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        data = parse_stories(f.readlines())
        if SBIEO:
            return convert2SBIEO(data)
        elif BIO:
            return convert2BIO(data)
        else:
            return data

def vectorize_data(
        data,
        word2idx,
        sentence_size,
        memory_size,
        ner2idx,
    ):
    nb_sentence = map(len, data)
    nb_sentences = sum(nb_sentence)
    ret_sentences = np.zeros((nb_sentences, sentence_size))
    ret_memories = np.zeros((nb_sentences, memory_size))
    ret_answers = np.zeros((nb_sentences, sentence_size))
    ret_mem_idx = np.zeros((nb_sentences, sentence_size))
    
    for i, document in enumerate(data):
        memory = []
        for j, sentence in enumerate(document):
            for k, (word, pos, chunk, ner) in enumerate(sentence):
                idx = sum(nb_sentence[:i]) + j
                ret_sentences[idx, k] = word2idx[word] if word in word2idx else 1 # 1 for unk
                ret_answers[idx, k] = ner2idx[ner]
                ret_mem_idx[idx, k] = sum([len(s) for s in document[:j]]) + k # memory accessible to the current word inclusively
                memory.append(ret_sentences[idx, k])
        memory = memory[:memory_size]
        idx_start = sum(nb_sentence[:i])
        for j, sentence in enumerate(document):
            ret_memories[idx_start + j, :len(memory)] = memory
    
    return ret_sentences, ret_memories, ret_answers, ret_mem_idx

class AbstractFeature(object):
    def generate_feature(self, word):
        raise NotImplementedError("Not implemented")
    
    def feature_size(self):
        raise NotImplementedError("Not implemented")

class CapitalizationFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if word[0].isupper() else 0
    def feature_size(self):
        return 1

class AllCapitalizedFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if word.isupper() else 0
    def feature_size(self):
        return 1

class AllLowerFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if word.islower() else 0
    def feature_size(self):
        return 1

class NonInitialCapFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if any([c.isupper() for c in word[1:]]) else 0
    def feature_size(self):
        return 1

class MixCharDigitFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if any([c.isalpha() for c in word]) and any([c.isdigit() for c in word]) else 0
    def feature_size(self):
        return 1

class HasPunctFeature(AbstractFeature):
    def __init__(self):
        self._punct_set = set(string.punctuation)
    def generate_feature(self, word):
        return 1 if any([c in self._punct_set for c in word]) else 0
    def feature_size(self):
        return 1

class PreSuffixFeature(AbstractFeature):
    def __init__(self, window_size, is_prefix):
        self._vocab = {}
        self._window_size = window_size
        self._is_prefix = is_prefix
    def generate_feature(self, word):
        w = word.lower()
        fix = w[:self._window_size] if self._is_prefix else w[-self._window_size:]
        if fix in self._vocab:
            return self._vocab[fix]
        else:
            self._vocab[fix] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[fix]
    def feature_size(self):
        return len(self._vocab) + 1

class HasApostropheFeature(AbstractFeature):
    def generate_feature(self, word):
        return 1 if word.lower()[-2:] == "'s" else 0
    def feature_size(self):
        return 1

class LetterOnlyFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}
    
    def generate_feature(self, word):
        w = filter(lambda x: x.isalpha(), word)
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]
    
    def feature_size(self):
        return len(self._vocab) + 1

class NonLetterOnlyFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}

    def generate_feature(self, word):
        w = filter(lambda x: not x.isalpha(), word)
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]

    def feature_size(self):
        return len(self._vocab) + 1

class WordPatternFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}
    def generate_feature(self, word):
        w = []
        for c in word:
            if c.isalpha() and c.islower():
                w.append('a')
            elif c.isalpha() and c.isupper():
                w.append('A')
            elif c.isdigit():
                w.append('0')
            else:
                w.append
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]
    def feature_size(self):
        return len(self._vocab) + 1

class WordPatternSummarizationFeature(AbstractFeature):
    def __init__(self):
        self._vocab = {}
    def generate_feature(self, word):
        w = []
        for c in word:
            if c.isalpha() and c.islower():
                if len(w) == 0 or w[-1] != 'a':
                    w.append('a')    
            elif c.isalpha() and c.isupper():
                if len(w) == 0 or w[-1] != 'A':
                    w.append('A')
            elif c.isdigit():
                if len(w) == 0 or w[-1] != '0':
                    w.append('0')
            else:
                w.append
        if w in self._vocab:
            return self._vocab[w]
        else:
            self._vocab[w] = len(self._vocab) + 1 # +1 for unk
            return self._vocab[w]
    def feature_size(self):
        return len(self._vocab) + 1

def vectorize_lexical_features(data, sentence_size, memory_size):
    feature_list = []
    cap_feature = CapitalizationFeature()
    all_cap_feature = AllCapitalizedFeature()
    all_lower_feature = AllLowerFeature()
    non_init_cap_feature = NonInitialCapFeature()
    mx_char_digit_feature = MixCharDigitFeature()
    has_punct_feature = HasPunctFeature()
    feature_list = [
        cap_feature,
        all_cap_feature,
        all_lower_feature,
        non_init_cap_feature,
        mx_char_digit_feature,
        has_punct_feature,
    ]
    lexical_feature_size = sum([f.feature_size() for f in feature_list])
    nb_sentence = map(len, data)
    nb_sentences = sum(nb_sentence)
    sentence_lexical_features = np.zeros((nb_sentences, sentence_size, lexical_feature_size))
    memory_lexical_features = np.zeros((nb_sentences, memory_size, lexical_feature_size))
    for i, document in enumerate(data):
        mlf = []
        for j, sentence in enumerate(document):
            for k, (word, pos, chunk, ner) in enumerate(sentence):
                idx = sum(nb_sentence[:i]) + j
                features = [f.generate_feature(word) for f in feature_list]
                sentence_lexical_features[idx, k] = features
                mlf.append(features)
        mlf = mlf[:memory_size]
        idx_start = sum(nb_sentence[:i])
        for j, sentence in enumerate(document):
            memory_lexical_features[idx_start + j, :len(mlf), :] = mlf
    return sentence_lexical_features, memory_lexical_features

