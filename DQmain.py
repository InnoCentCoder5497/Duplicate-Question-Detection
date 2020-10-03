
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from torch.utils.data import random_split
from tqdm import tqdm

from data_utils import get_dataset, build_vocab, DQDataset, collate_fn

DATA_PATH = r'D:/Jupyter work/Duplicate Question Detection/questions.csv';

TRAIN_BATCH_SIZE = 32
VALIDATE_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

dataset = get_dataset(DATA_PATH)

s = dataset[['question1', 'question2', 'is_duplicate']].values

print("Spliting dataset")
train, test_and_val = train_test_split(s, test_size = 0.3)

same_idx = np.where(train[:, 2] == 1)[0]

train_set = train[same_idx]

print("Building vocab")
vocab = build_vocab(train_set)


print("Creating Dataloader")
dlt = DQDataset(train_set, vocab)
dl = DataLoader(dlt, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn)

test_val_ds = DQDataset(test_and_val, vocab)

VAL_SPLIT_SIZE = len(test_val_ds) - len(test_val_ds) // 3
TEST_SPLIT_SIZE = len(test_val_ds) // 3

val_set, test_set = random_split(test_val_ds, [VAL_SPLIT_SIZE, TEST_SPLIT_SIZE])

vdl = DataLoader(val_set, batch_size=VALIDATE_BATCH_SIZE, collate_fn=collate_fn)
tdl = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, collate_fn=collate_fn)