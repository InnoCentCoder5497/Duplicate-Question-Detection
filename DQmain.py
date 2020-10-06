import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.data import random_split
from tqdm import tqdm

from data_utils import get_dataset, build_vocab, DQDataset, collate_fn
from model import SiameseNet, loss_fn, accuracy_score, run_on_example

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

VOCAB_SIZE = len(vocab) + 1
EMB_DIM = 100
HIDDEN_SIZE = 128

LR = 0.001
N_EPOCH = 10

print("Creating network")
net = SiameseNet(VOCAB_SIZE, EMB_DIM, HIDDEN_SIZE)

opt = optim.Adam(net.parameters(), lr = LR)

net = net.cuda()

sim = nn.CosineSimilarity()
print(net)
print("Running training loop")
cost_book = []
val_acc_book = []
for j in range(N_EPOCH):
    cost = 0
    pbar = tqdm(dl)
    for i, b in enumerate(pbar):
        opt.zero_grad()
        
        o1 = net(b['q1'].cuda())
        o2 = net(b['q2'].cuda())
        
        loss = loss_fn(o1, o2, device = 'cuda')
        l = loss.item()
        cost += l
        loss.backward()
        opt.step()
        pbar.set_postfix({'Epoch' : j + 1, 'Train_loss': l})
    
    pbar.close()
      
    print(f"\nEpoch Loss : {cost / (i + 1):.3f}\n")
    cost_book.append(cost / (i + 1))
    print("\nRunning on validation set\n")
    with torch.no_grad():
        acc = accuracy_score(vdl, net, sim, device='cuda')
        val_acc_book.append(acc)
        print(f"\nAccuracy of val set {acc:.3f}%\n")
    

print("\nRunning on Test set\n")
with torch.no_grad():
    acc = accuracy_score(tdl, net, sim, device='cuda')
    print(f"\nAccuracy of test set {acc:.3f}%\n")


import matplotlib.pyplot as plt

plt.title("Training Loss curve")
plt.plot(list(range(len(cost_book))), cost_book)

plt.title("Validation Accuracy curve")
plt.plot(list(range(len(val_acc_book))), val_acc_book)

question1 = "When will I see you?"
question2 = "When can I see you again?"
run_on_example(question1, question2, vocab, net, sim, threshold = 0.7, device = 'cuda')