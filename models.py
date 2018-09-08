import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, negative_sample = False):
        super(SkipGram, self).__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = Parameter(torch.FloatTensor(self.vocab_size, embedding_dim).uniform_(-0.5/embedding_dim, 0.5/embedding_dim))

        #
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size * context_size)

        self.negative_sampling = negative_sample

    def forward(self,inputs):
        embeds = self.embeddings(inputs)

        if self.negative_sampling:
            return embeds

        # print(embeds.shape)
        out = F.relu(self.linear1(embeds))
        # print(out.shape)
        out = self.linear2(out).view((1,self.context_size,self.vocab_size)).squeeze()

        log_prob = F.log_softmax(out,dim=1)
        return log_prob


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)


    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        # print("E",embeds.shape)
        out = F.relu(self.linear1(embeds))
        # print(out.shape)
        out = self.linear2(out)
        # print(out.shape)
        log_prob = F.log_softmax(out, dim=1)
        return log_prob


class NegativeSampling(nn.Module):
    def __init__(self, vocab_size, wordcount, embedding_dim, window_size, n_negs = 20):
        super(NegativeSampling, self).__init__()
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.context_size = window_size * 2

        self.skipgram = SkipGram(vocab_size, embedding_dim, window_size * 2, True)
        self.embeddings = self.skipgram.embeddings

        wf = np.array(list(wordcount.values()))

        wf = wf / wf.sum()

        wf = np.power(wf, 0.75)
        wf = wf / wf.sum()
        self.weights = torch.tensor(wf, dtype=torch.float)

    def forward(self, inword, outword):
        bs = inword.shape[0]

        nwords = torch.multinomial(self.weights, bs * self.context_size * self.n_negs, replacement=True)
        nwords = nwords.cuda()

        invec = self.skipgram(inword) # 1,128
        invec = torch.t(invec)
        outvec = self.skipgram(outword) # 4, 128
        nvec = self.skipgram(nwords).neg() # 20 * 4, 128

        oloss = (outvec@invec).squeeze().sigmoid().log().mean()
        nloss = (nvec@invec).squeeze().sigmoid().log().view(self.context_size, self.n_negs).sum(1).mean()

        return -(oloss + nloss).sum()