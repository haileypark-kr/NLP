import re
import numpy as np
import torch
from collections import Counter


class Preprocess:
    def __init__(self, dataroot, wsize):
        self.dataroot = dataroot
        self.window_size = wsize

    def read_data(self):
        f = open(self.dataroot)
        corpus = f.readlines()

        return corpus

    def tokenize_corpus(self, corpus):
        p = re.compile('<\w*?>*|\w*>|\\n|\s+/|[(].*?[)]')
        tokens = [p.sub("",x).lower().replace(",","").replace("!","").replace(".","").split() for x in corpus]
        return tokens


    def id_map(self):
        corpus = self.read_data()

        self.tokenized_corpus = self.tokenize_corpus(corpus)

        self.voca = set()
        self.counter = dict()
        for sentence in self.tokenized_corpus:
            for token in sentence:
                self.voca.add(token)

            wc = dict(Counter(sentence).items())
            for w in wc:
                if w not in self.counter:
                    self.counter[w] = 1
                else:
                    self.counter[w] += 1

        self.word2id = {w: idx for (idx, w) in enumerate(self.voca)}
        self.id2word = {idx: w for (idx, w) in enumerate(self.voca)}

        self.counter = {self.word2id[w] : count for (w, count) in self.counter.items()}

    def make_id_pair(self):
        self.id_map()

        id_pair = []

        for sentence in self.tokenized_corpus:
            # id of words in each sentence
            indices = [self.word2id[word] for word in sentence]
            # for each word, treated as center word
            # center word id
            for center_word_pos in range(len(indices)):
                # for each window position
                for i in range(self.window_size, len(indices) - self.window_size):  # iterate over sentence length
                    context = [indices[i - 1],
                               indices[i + 1]]
                    target = indices[i]
                    id_pair.append((target, context))

        return id_pair

    def __len__(self):
        return len(self.voca)