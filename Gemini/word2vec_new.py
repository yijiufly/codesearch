import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from datetime import datetime, timedelta
from config import *
import numpy as np
import random
import re
import _pickle as pkl

VOCAB_SIZE_W2V = 1000


class DataLoader:
    EOS = 0  # to mean end of sentence
    UNK = 1  # to mean unknown token

    maxlen = MAXLEN

    def __init__(self, text_file=None, sentences=None, word_dict=None):

        if text_file:
            sentences = []
            for txt_file in text_file:
                print("Loading text file at {}".format(txt_file))
                with open(txt_file, "rt") as f:
                    text = f.readlines()
                    for i, line in enumerate(text):
                        if i % 2:
                            sentences.extend(line.strip().split(';'))
                print("Making dictionary for these words")
            word_dict = self.build_and_save_dictionary(sentences, source="data/word2vec")

        assert sentences and word_dict, "Please provide the file to extract from or give sentences and word_dict"

        self.sentences = sentences
        self.word_dict = word_dict
        print("Making reverse dictionary")
        self.revmap = list(self.word_dict.items())

        self.lengths = [len(sent) for sent in self.sentences]

    def build_and_save_dictionary(self, text, source):
        save_loc = source+".pkl"
        try:
            cached = self.load_dictionary(save_loc)
            print("Using cached dictionary at {}".format(save_loc))
            return cached
        except:
            pass
        # build again and save
        print("unable to load from cached, building fresh")
        worddict, wordcount = self.build_dictionary(text)
        print("Got {} unique words".format(len(worddict)))
        print("Saveing dictionary at {}".format(save_loc))
        self.save_dictionary(worddict, wordcount, save_loc)
        return worddict

    def build_dictionary(self, text):
        """
        Build a dictionary
        text: list of sentences (pre-tokenized)
        """
        wordcount = {}
        for w in text:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
        sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)
        worddict = OrderedDict()
        for idx, word in enumerate(sorted_words):
            worddict[word] = idx + 2 # 0: <eos>, 1: <unk>
        return worddict, wordcount


    def load_dictionary(self, loc='./data/book_dictionary_large.pkl'):
        """
        Load a dictionary
        """
        with open(loc, 'rb') as f:
            worddict = pkl.load(f)
        return worddict


    def save_dictionary(self, worddict, wordcount, loc='./data/book_dictionary_large.pkl'):
        """
        Save a dictionary to the specified location
        """
        with open(loc, 'wb') as f:
            pkl.dump(worddict, f)
            pkl.dump(wordcount, f)


    # def convert_sentence_to_indices(self, sentence):
    #     sentence = re.split(',| ', sentence)
    #     tokn_lst = []
    #     for s in sentence:
    #         tokn_lst.extend(re.split('([0-9A-Za-z@_.]+)', s))
    #     tokn_lst = [t for t in tokn_lst if t]
    #     indices = [
    #                   # assign an integer to each word, if the word is too rare assign unknown token
    #                   self.word_dict.get(w) if self.word_dict.get(w, VOCAB_SIZE_W2V + 1) < VOCAB_SIZE_W2V else self.UNK

    #                   for w in tokn_lst  # split into words on spaces
    #               ][: self.maxlen - 1]  # take only maxlen-1 words per sentence at the most.

    #     # last words are EOS
    #     indices += [self.EOS] * (self.maxlen - len(indices))

    #     indices = np.array(indices)
    #     indices = Variable(torch.from_numpy(indices))
    #     return indices

    def convert_index_to_word(self, idx):

        idx = idx.data.item()
        if idx == 0:
            return "EOS"
        elif idx == 1:
            return "UNK"
        
        search_idx = idx - 2
        if search_idx >= len(self.revmap):
            return "NA"
        
        word, idx_ = self.revmap[search_idx]

        assert idx_ == idx
        return word

    def fetch_batch(self, batch_size):

        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            if self.word_dict.get(self.sentences[i], VOCAB_SIZE_W2V + 1) < VOCAB_SIZE_W2V:
                ind = self.word_dict.get(self.sentences[i])
            else:
                ind = self.UNK
            batch.append(ind)
        batch = np.array(batch)
        batch = Variable(torch.from_numpy(batch))
        if USE_CUDA:
            batch = batch.cuda(CUDA_DEVICE)
        return batch


class CBOW(nn.Module):
    word_size = 128

    def __init__(self):
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE_W2V, self.word_size)
        self.worder = nn.Linear(self.word_size, VOCAB_SIZE_W2V)

    def forward(self, sentences):
        word_embeddings = torch.tanh(self.word2embd(sentences))
        prev_embd = word_embeddings[:-2]
        next_embd = word_embeddings[2:]
        pred_word = F.log_softmax(self.worder(prev_embd + next_embd), dim=1)
        loss = F.cross_entropy(pred_word.view(-1, VOCAB_SIZE_W2V), sentences[1:-1].view(-1))
        return loss, sentences[1:-1], pred_word


def debug(i, loss, prev, pred, mod):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.data.item()
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time
    print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(
              i, time_elapsed, last_best_loss, this_loss))

    for i in range(3,12):
        _, pred_ids = pred[i].max(0)
        print("current = {}\npred = {}".format(
            d.convert_index_to_word(prev[i]),
            d.convert_index_to_word(pred_ids)
        ))
        print("=============================================")
    
    try:
        trail_loss = sum(loss_trail)/len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            print("Loss improved from {} to {}".format(last_best_loss, trail_loss))
            
            save_loc = "./saved_models/w2v_new-best".format(lr, VOCAB_SIZE_W2V)
            print("saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)
            
            last_best_loss = trail_loss

            #save embeddings:
    except Exception as e:
       print("Couldn't save model because {}".format(e))


if __name__ == "__main__":
    data_list = ['data/training/sequences' + str(i) + '.txt' for i in range(20)]
    d = DataLoader(data_list)
    model = CBOW()
    if USE_CUDA:
        model.cuda(CUDA_DEVICE)    
    lr = 1e-4
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_trail = []
    last_best_loss = None
    current_time = datetime.utcnow()


    print("Starting training...")

    for i in range(0, 800000):
        sentences = d.fetch_batch(32) 
        loss, prev, pred = model(sentences)
        if i % 2000 == 0:
            debug(i, loss, prev, pred, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.data.item())
   

