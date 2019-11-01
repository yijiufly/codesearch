"""
This file implements the Skip-Thought architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *


class Encoder(nn.Module):
    thought_size = 128
    word_size = 256

    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(self.word_size, self.thought_size)

    def forward(self, embeddings):
        # sentences = (batch_size, maxlen), with padding on the right.

        # sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)

        # word_embeddings = torch.tanh(self.word2embd(sentences))  # (maxlen, batch_size, word_size)
        _, (thoughts, _) = self.lstm(embeddings)

        return thoughts[-1]


class DuoDecoder(nn.Module):

    word_size = Encoder.word_size

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2*Encoder.thought_size, self.word_size)
        self.worder = nn.Linear(self.word_size, VOCAB_SIZE)

    def forward(self, thoughts):
        thoughts = thoughts.repeat(MAXLEN, 1, 1)
        pred_embd, _ = self.lstm(thoughts)
        pred_word = F.log_softmax(self.worder(pred_embd), dim=2)
        pred_word = pred_word.transpose(0, 1).contiguous()
        return pred_word


class UniSkip(nn.Module):

    def __init__(self):
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE, Encoder.word_size)
        self.encoder = Encoder()
        self.decoders = DuoDecoder()

    def create_mask(self, var, lengths):
        mask = var.data.new().resize_as_(var.data).fill_(0)
        for i, l in enumerate(lengths):
            if l == MAXLEN:
                for j in range(l):
                    mask[i, j] = 1
            else:
                for j in range(l+1):
                    mask[i, j] = 1
        
        mask = Variable(mask)
        if USE_CUDA:
            mask = mask.cuda(var.get_device())
            
        return mask

    def forward(self, sentences, lengths):
        # sentences = (B, maxlen)
        # lengths = (B)
        sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)
        word_embeddings = torch.tanh(self.word2embd(sentences)) 
        # Compute Thought Vectors for each sentence. Also get the actual word embeddings for teacher forcing.
        thoughts = torch.cat([self.encoder(word_embeddings[:,:-2,:]), self.encoder(word_embeddings[:,2:,:])], dim=1)  #  word_embeddings = (B, maxlen, word_size)
        
        # Predict the words for previous and next sentences.
        pred_word = self.decoders(thoughts)  # both = (batch-1, maxlen, VOCAB_SIZE)

        # mask the predictions, so that loss for beyond-EOS word predictions is cancelled.
        # pred_mask = self.create_mask(pred_word, lengths[1:-1])
        
        # masked_pred_word = pred_word * pred_mask

        loss = F.cross_entropy(pred_word.view(-1, VOCAB_SIZE), sentences.transpose(0, 1)[1:-1, :].view(-1))

        return loss, sentences.transpose(0, 1), pred_word













