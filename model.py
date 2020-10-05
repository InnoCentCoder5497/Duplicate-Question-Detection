# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 07:37:21 2020

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:35:27 2020

@author: Asus
"""

import torch
import torch.nn as nn
from nltk import word_tokenize
from tqdm import tqdm

class SiameseNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super(SiameseNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.cell = nn.LSTM(emb_dim, hidden_size, batch_first = True)
        
    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.cell(x)
        
        out = out.mean(dim = 1)
        
        return out / out.norm(dim = 1, keepdim = True)


def loss_fn(q1, q2, alpha = 0.25, device = 'cpu'):
    sim = torch.mm(q1, q2.transpose(0, 1))
    
    b = sim.shape[0]
    
    sim_ap = torch.diag(sim)
    sim_an = sim - torch.diag(sim_ap)
    
    mean_neg = torch.sum(sim_an, dim = 1, keepdim = True) / (b - 1)
    
    m1 = torch.eye(b, device = device) == 1

    m2 = sim_an > sim_ap.view(b, 1)
    m = m1 | m2
    sim_an[m] = -2
    closest_neg, _ = torch.topk(sim_an, 1, dim = 1)
    
    l1 = torch.max(mean_neg - sim_ap.view(b, 1) + alpha, torch.zeros_like(mean_neg))
    l2 = torch.max(closest_neg - sim_ap.view(b, 1) + alpha, torch.zeros_like(closest_neg))
    
    return torch.sum(l1 + l2)

def accuracy_score(data_loader, net, sim_formula, threshold = 0.7, device = 'cpu'):
    num_correct = 0
    total = 0
    
    for batch in tqdm(data_loader):
        o1 = net(batch['q1'].to(device))
        o2 = net(batch['q2'].to(device))
        
        sims = sim_formula(o1, o2)
        sims[sims > threshold] = 1
        sims[sims <= threshold] = 0
        
        num_correct += torch.sum(sims == batch['labels'].to(device)).item()
        total += sims.shape[0]
        
    return (num_correct * 100) / total

def run_on_example(q1, q2, vocab, net, sim, threshold = 0.7, device = 'cpu'):
    q1 = word_tokenize(q1.lower())
    q2 = word_tokenize(q2.lower())
    in1 = [vocab[word] for word in q1]
    in2 = [vocab[word] for word in q2]

    in1_tens = torch.LongTensor([in1])
    in2_tens = torch.LongTensor([in2])

    with torch.no_grad():
        oo1 = net(in1_tens.to(device))
        oo2 = net(in2_tens.to(device))
        
    sim_val = sim(oo1, oo2).item()

    print(f"Q1 : {q1}\nQ2 : {q2}")
    print(f"Similarity Score : {sim_val}")
    print(f"Is duplicate : {sim_val > threshold}")
