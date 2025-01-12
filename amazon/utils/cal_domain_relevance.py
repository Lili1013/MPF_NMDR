import os

import os
import sys
curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import pandas as pd
import pickle
import random
import torch
import numpy as np
import torch.nn.functional as F
from loguru import logger
def calculate_relevance(text_feat_path_1,text_feat_path_2):
    with open(text_feat_path_1 ,'rb') as f_text_1:
        text_emb_1 = torch.from_numpy(np.load(f_text_1))
    with open(text_feat_path_2 ,'rb') as f_text_2:
        text_emb_2 = torch.from_numpy(np.load(f_text_2))
    avg_text_emb_1 = torch.mean(text_emb_1,dim=0)
    avg_text_emb_2 = torch.mean(text_emb_2, dim=0)
    sim = F.cosine_similarity(avg_text_emb_1,avg_text_emb_2,dim=0)
    print(sim)

if __name__ == '__main__':
    path = f'../../v0/datasets/amazon_review'
    # domains = f'health_cloth_beauty'
    domains = f'phone_cloth_sport'
    dataset_1 = 'sport'
    dataset_2 = 'cloth'
    text_feat_path_1 = f'{path}/{domains}_single/{dataset_1}/{dataset_1}_text_features.npy'
    text_feat_path_2 = f'{path}/{domains}_single/{dataset_2}/{dataset_2}_text_features.npy'
    calculate_relevance(text_feat_path_1,text_feat_path_2)

