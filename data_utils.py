# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 07:28:40 2020

@author: Asus
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from nltk import word_tokenize

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

# Read the dataset into pandas dataframe
def get_dataset(path):
    print("Loading dataset")
    dataset = pd.read_csv(path)
    dataset = dataset.dropna()
    dataset.question1 = dataset.question1.str.lower()
    dataset.question2 = dataset.question2.str.lower()
    return dataset

# Run over all the sentences and create vocabulary
def build_vocab(data):
    vocab_dict = defaultdict(int)
    vocab_dict['<pad>'] = 1

    for sentence1, sentence2, _ in tqdm(data):
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)

        for word in tokens1 + tokens2:
            if vocab_dict[word] == 0:
                vocab_dict[word] = len(vocab_dict)
                
    return vocab_dict

# Pytorch Dataset class for creating Dataloader
class DQDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q1 = self.data[idx, 0]
        q2 = self.data[idx, 1]
        label = self.data[idx, 2]
        input1 = [self.vocab[word] for word in word_tokenize(q1)]
        input2 = [self.vocab[word] for word in word_tokenize(q2)]
        
        return {'q1' : input1, 'q2' : input2, 'label' : label}

# Collate function to form batches of equal sized sentences
def collate_fn(batch):
    q1 = []
    q2 = []
    q1_len = []
    q2_len = []
    labels = []
    
    bs = len(batch)
    
    for questions in batch:
        q1.append(questions['q1'])
        q2.append(questions['q2'])
        
        q1_len.append(len(questions['q1']))
        q2_len.append(len(questions['q2']))
        labels.append(questions['label'])
    
    max_len = max(max(q1_len), max(q2_len))
    
    q1_batch = np.ones([bs, max_len], dtype=np.long)
    q2_batch = np.ones([bs, max_len], dtype=np.long)
    
    for i, (in1, in2) in enumerate(zip(q1, q2)):
        q1_batch[i, :q1_len[i]] = in1
        q2_batch[i, :q2_len[i]] = in2
        
    return {
        'q1' : torch.LongTensor(q1_batch), 
        'q2' : torch.LongTensor(q2_batch),
        'labels' : torch.LongTensor(labels)
    }
