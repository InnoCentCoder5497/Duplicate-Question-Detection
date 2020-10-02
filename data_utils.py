# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 07:28:40 2020

@author: Asus
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from nltk import word_tokenize

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
