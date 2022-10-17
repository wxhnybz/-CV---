import os
import cv2
import string
from tqdm import tqdm_notebook
import click
import numpy as np
from CRNN_1 import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import editdistance

def test(net, data, abc, cuda, batch_size=50):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

    count = 0
    tp = 0
    avg_ed = 0
    iterator = tqdm_notebook(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        out = net(imgs, decode=True)
        gt = (sample["seq"].numpy() - 1).tolist()
        lens = sample["seq_len"].numpy().tolist()
        pos = 0
        key = ''
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
            pos += lens[i]
            if gts == out[i]:
                tp += 1
            else:
                avg_ed += editdistance.eval(out[i], gts)
            count += 1
        iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))

    acc = tp / count
    avg_ed = avg_ed / count
    return acc, avg_ed