import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from config import *
import random
import pickle

def create_skipgram_dataset(text):
    data = []
    for i in range(2, len(text) - 2):
        data.append((text[i], text[i-2], 1))
        data.append((text[i], text[i-1], 1))
        data.append((text[i], text[i+1], 1))
        data.append((text[i], text[i+2], 1))
        # negative sampling
        for _ in range(4):
            if random.random() < 0.5 or i >= len(text) - 3:
                rand_id = random.randint(0, i-1)
            else:
                rand_id = random.randint(i+3, len(text)-1)
            data.append((text[i], text[rand_id], 0))
    return data



class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
    
    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = F.logsigmoid(score)
    
        return log_probs

    def save_embeddings(i2w):
        embedding = self.embeddings.weight.data.numpy()
        with open('data/embeddings.txt', 'w') as f:
            for idx, word in i2w.items():
                e = embedding[idx]
                e = ' '.join(map(lambda x:str(x),e))
                f.write('{} {}\n'.format(word, e))

    def save_model(self, loc):
        print("saving model at {}".format(loc))
        try:
            torch.save(self.state_dict(), loc)
        except Exeception as e:
            print("Couldn't save model because {}".format(e))



def train_skipgram(skipgram_train, vocab_size, w2i):
    embd_size = 256
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocab_size, embd_size)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(10):
        total_loss = .0
        i = 0
        last_best_loss = None
        current_loss = .0
        print("iteration: {}".format(epoch))
        for in_w, out_w, target in skipgram_train:
            in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
            out_w_var = Variable(torch.LongTensor([w2i[out_w]]))
            target_var = Variable(torch.Tensor([target]))


            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], target_var)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            current_loss += loss.data.item()
            if i % 1000 == 0 and i > 0:
                print("iteration: {}-{}".format(epoch, i))
                print("loss: {}".format(current_loss))
                if last_best_loss is None or last_best_loss > current_loss:
                    model.save_model("./saved_models/w2v_best")
                    last_best_loss = current_loss
                current_loss = .0
            i+=1
        losses.append(total_loss)
    return model, losses



def test_skipgram(test_data, model, w2i):
    print('====Test SkipGram===')
    correct_ct = 0
    for in_w, out_w, target in test_data:
        in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
        out_w_var = Variable(torch.LongTensor([w2i[out_w]]))

        model.zero_grad()
        log_probs = model(in_w_var, out_w_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted = predicted[0]
        if predicted == target:
            correct_ct += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct/len(test_data)*100, correct_ct, len(test_data)))


if __name__ == "__main__":
    text_file = ['data/training/sequences' + str(i) + '.txt' for i in range(10)]
    sentences = []
    for txt_file in text_file:
        print("Loading text file at {}".format(txt_file))
        with open(txt_file, "rt") as f:
            text = f.readlines()
            
            for i, line in enumerate(text):
                if i % 2:
                    sentences.extend(line.strip().split(';'))
        
    vocab = set(sentences)
    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    with open('./saved_models/w2i.pkl', 'wb') as f:
        pickle.dump(w2i, f, pickle.HIGHEST_PROTOCOL)
    skipgram_train = create_skipgram_dataset(sentences)
    print('skipgram sample', skipgram_train[0])
    sg_model, sg_losses = train_skipgram(skipgram_train, vocab_size, w2i)




        