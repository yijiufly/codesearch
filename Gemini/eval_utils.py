from model import UniSkip
from data_loader import DataLoader
from vocab import load_dictionary
from config import *
from torch import nn

from torch.autograd import Variable
import torch
import numpy as np
import word2vec
import word2vec_new
import os
PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
USE_CUDA = False
class UsableEncoder:

    def __init__(self, loc = PROJ_DIR + "/saved_models/skip-best"):
        print("Preparing the DataLoader. Loading the word dictionary")
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary(PROJ_DIR + '/data/instruction.pkl'))
        self.encoder = None

        print("Loading encoder from the saved model at {}".format(loc))
        self.model = UniSkip()
        self.model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))
        self.encoder = self.model.encoder
        self.decoders = self.model.decoders
        self.word2embd = self.model.word2embd
        if USE_CUDA:
            self.encoder.cuda(CUDA_DEVICE)
            self.decoders.cuda(CUDA_DEVICE)

    # def encode(self, text):
    #     def chunks(l, n):
    #         """Yield successive n-sized chunks from l."""
    #         for i in range(0, len(l), n):
    #             yield l[i:i + n]

    #     ret = []

    #     for chunk in chunks(text, 1000):
    #         print("encoding chunk of size {}".format(len(chunk)))
    #         indices = [self.d.convert_sentence_to_indices(sentence) for sentence in chunk]
    #         indices = torch.stack(indices)
    #         indices, _ = self.encoder(indices)
    #         indices = indices.view(-1, self.encoder.thought_size)
    #         indices = indices.data.cpu().numpy()

    #         ret.extend(indices)
    #     ret = np.array(ret)

    #     return ret

    def predict(self, text):
        prng = np.random.RandomState(2019)
        inds = np.arange(1, int(len(text)/100))
        prng.shuffle(inds)
        curr_text = [text[i].strip() for i in inds]
        prev_text = [text[i-1].strip() for i in inds]
        next_text = [text[i+1].strip() for i in inds]
        indices = [self.d.convert_sentence_to_indices(sentence) for sentence in curr_text]
        indices = torch.stack(indices)
        thoughts, word_embeddings = self.encoder(indices)
        prev_pred, next_pred = self.decoders(thoughts, word_embeddings)
        return curr_text, prev_text, next_text, prev_pred, next_pred



    def encode(self, text):
        result = []
        indices = [self.d.convert_sentence_to_indices(sentence) for sentence in text]
        indices = torch.stack(indices)
        indices = indices.transpose(0, 1)
        indices = torch.tanh(self.word2embd(indices))
        if USE_CUDA:
            indices = indices.cuda(CUDA_DEVICE)
        indices = self.encoder(indices)
        indices = indices.view(-1, self.encoder.thought_size)
        indices = indices.data.cpu().numpy()
        result.extend(indices)
        result = np.array(result)
        return result


class UsableWord2Vec:

    # def __init__(self,vocab_size, loc="./saved_models/w2v_best"):
    #     print("Loading encoder from the saved model at {}".format(loc))
    #     self.model = word2vec.SkipGram(vocab_size, 256)
    #     self.model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))

    # def encode(self, text, w2i):
    #     # vocab = set(text)
    #     # vocab_size = len(vocab)
    #     # print('vocab_size:', vocab_size)
    #     # w2i = {w: i for i, w in enumerate(vocab)}
    #     # i2w = {i: w for i, w in enumerate(vocab)}
    #     test_data = word2vec.create_skipgram_dataset(text)
    #     result = []
    #     for in_w in text:
    #         if in_w not in w2i:
    #             # in_w = "call error"
    #             in_w = "retn"
    #         in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
    #         embedding = self.model.embeddings(in_w_var).view((1,-1))
    #         result.extend(embedding.data.numpy())
    #     result = np.array(result)
    #     return result

    def __init__(self,vocab_size, loc="./saved_models/w2v_new-best"):
        print("Loading encoder from the saved model at {}".format(loc))
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary('data/word2vec.pkl'))
        self.model = word2vec_new.CBOW()
        self.word2embd = self.model.word2embd
        self.model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))

    def encode(self, text, w2i):
        result = []
        indices = []
        for in_w in text:
            if self.d.word_dict.get(in_w, VOCAB_SIZE + 1) < VOCAB_SIZE:
                ind = self.d.word_dict.get(in_w)
            else:
                ind = self.d.UNK
            indices.append(ind)
        indices = np.array(indices)
        indices = Variable(torch.from_numpy(indices))
        # if USE_CUDA:
        #     indices = indices.cuda(CUDA_DEVICE)
        indices = torch.tanh(self.word2embd(indices))
        indices = indices.data.cpu().numpy()
        return indices
