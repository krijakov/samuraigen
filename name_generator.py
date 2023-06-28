#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Description: Trains the model or samples it: "train" command line argument for training, "generate" for sampling

'''
import torch
import sys

import hiragana_tokenizer as hir
import model

# training parameters:
max_iters = 1600
eval_iters = 100
learning_rate = 1e-3
config = model.ModelConfigs

if __name__ == "__main__":
    # import data:
    with open('Data/samurai_names.txt', 'r') as d:
        nams = d.read().lower()
    names = [' '.join(n.lower().split(" ")) + "\n" for n in nams.splitlines()]

    # tokenize:
    toker = hir.HiraganaTokenizer()
    enc = []
    for n in names: 
        enc_n = toker(n)
        enc += enc_n
    # reduce tokens and re-tokenize the data:
    toker.reduceTokens(enc)
    num = 0
    enc = []
    for n in names: 
        enc_n = toker(n)
        enc += enc_n
        if len(enc_n):
            num += 1
    print(f'Successfully encoded names with minimal tokens: {num}')

    config.vocab_size = toker.vocab_size # fit vocabsize after token reduction

    # construct data and batches:

    data = torch.tensor(enc, dtype=torch.long)
    # split up data:
    n1 = int(len(data) * config.train_rate)
    train_data = data[:n1]
    val_data = data[n1:]

    # construct batches:
    def getBatch(data_split):
        dat = train_data if data_split == 'train' else val_data
        # generate the random batch indices:
        ix = torch.randint(0, len(dat) - config.block_size, (config.batch_size, ))
        x = torch.stack([dat[i:i+config.block_size] for i in ix]) # input
        y = torch.stack([dat[i+1:i+config.block_size+1] for i in ix]) # expected output
        return x, y

    @torch.no_grad()
    def evalLoss(data_split, model):
        dat = train_data if data_split == 'train' else val_data
        set_loss = []
        for _  in range(eval_iters):
            xb, yb = getBatch(dat)
            _, loss = model(xb, yb)
            set_loss.append(loss)
        return sum(set_loss)/len(set_loss)




    if str(sys.argv[1]) == "train": 
        # initialize the model:
        modell = model.NameGenerator(config)
        print(f'Total number of parameters: {sum(p.numel() for p in modell.parameters())}')

        # training loop:
        # create optimizer:
        optimizer = torch.optim.AdamW(modell.parameters(), lr=learning_rate)
        # track losses:
        lossi = []  
        for i in range(max_iters):
            # construct a batch:
            xb, yb = getBatch('train')

            # evaluate loss:
            _, loss = modell(xb, yb)

            # backward and update:
            optimizer.zero_grad(set_to_none=True)
            loss.backward() # backpropagate
            optimizer.step() # update

            # track and print stats:
            if i % 100 == 0:
                loss_tr = evalLoss("train", modell)
                loss_val = evalLoss("val" , modell)
                lossi.append(loss_val)
                print(f'{i:5d}/{max_iters:5d}: train loss: {loss_tr:4f}, validation loss: {loss_val:4f}')

        # export the parameters:
        torch.save(modell.state_dict(), "Params/params.pt")

    # when generating:
    if str(sys.argv[1]) == "generate":

        modell = model.NameGenerator(config)
        modell.load_state_dict(torch.load("Params/params.pt"))

        # generate:
        context = torch.tensor([[2]], dtype=torch.long)
        print(toker.decode(modell.generate(context, 500)[0].tolist(), joined=True))

