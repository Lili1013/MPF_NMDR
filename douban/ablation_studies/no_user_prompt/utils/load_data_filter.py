# import os
# import sys
# curPath = os.path.abspath(os.path.dirname((__file__)))
# rootPath = os.path.split(curPath)[0]
# PathProject = os.path.split(rootPath)[0]
# sys.path.append(rootPath)
# sys.path.append(PathProject)
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
from loguru import logger

# from para_parser import parse
# args = parse()


class Load_Data(object):

    def __init__(self, **params):
        self.rating_file_path = params['rating_path']
        self.args = params['args']
        self.text_feat_path = params['text_feat_path']
        # self.visual_feat_path = params['visual_feat_path']
        self.train_path = params['train_path']
        # self.test_path = params['test_path']
        self.test_neg_path_zero_shot = params['test_neg_path_zero_shot']
        self.test_neg_path_few_shot = params['test_neg_path_few_shot']
        self.test_neg_path_warm_shot = params['test_neg_path_warm_shot']
        # self.sim_item_path = params['sim_item_path']
        # self.u_neg_path = params['u_neg_path']
        logger.info('load datasets')
        self.load_orig_dataset()
        # self.split_datasets()
        # logger.info('calculate user neighbors')
        # self.calculate_user_neighbors()
        logger.info('create data loader')
        self.create_data_loader()
        #modify version, rduce test time
        logger.info('read test samples')
        self.test_pos_samples_zero_shot, self.test_neg_samples_zero_shot = self.read_test_samples(self.test_neg_path_zero_shot)
        self.test_pos_samples_few_shot, self.test_neg_samples_few_shot = self.read_test_samples(
            self.test_neg_path_few_shot)
        self.test_pos_samples_warm_shot, self.test_neg_samples_warm_shot = self.read_test_samples(
            self.test_neg_path_warm_shot)

    def generate_user_inter_lists(self,df):
        ui_interaction = {}
        for x in df.groupby(by='userID'):
            ui_interaction[x[0]] = list(x[1]['itemID'])
        return ui_interaction


    def load_orig_dataset(self):
        self.df_rating = pd.read_csv(self.rating_file_path)[['userID','itemID','rating','domain_id','user_cat']]
        self.num_users = len(self.df_rating['userID'].unique())
        self.num_items = len(self.df_rating['itemID'].unique())
        self.train_df = pd.read_csv(self.train_path)
        # self.test_df = pd.read_csv(self.test_path)
        # self.df_rating['centroid'] = [np.array([0]) for i in range(len(self.df_rating))]
        with open(self.text_feat_path,'rb') as f_text:
            self.text_emb = torch.from_numpy(np.load(f_text))
        # with open(self.visual_feat_path,'rb') as f_visual:
        #     self.visual_emb = torch.from_numpy(np.load(f_visual))


    def create_data_loader(self):
        train_u, train_v, train_d,train_r = self.train_df['userID'].values.tolist(), self.train_df['itemID'].values.tolist(), \
                                    self.train_df['domain_id'].values.tolist(),self.train_df['rating'].values.tolist(),
        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),torch.FloatTensor(train_d),
                                                  torch.FloatTensor(train_r))
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
    def generate_train_instances(self,batch_u,batch_v,batch_d):
        num_items = len(self.df_rating['itemID'].unique())
        uids,iids,domains,ratings = [],[],[],[]
        for i in range(len(batch_u)):
            u_id = batch_u[i]
            pos_i_id = batch_v[i]
            uids.append(u_id)
            iids.append(pos_i_id)
            domains.append(int(batch_d[i]))
            ratings.append(1)
            for t in range(self.args.train_neg_num):
                j = np.random.randint(num_items)
                while len(self.df_rating[(self.df_rating['userID'] == u_id) & (self.df_rating['itemID'] == j)]) > 0:
                    j = np.random.randint(num_items)
                uids.append(u_id)
                iids.append(j)
                domains.append(int(batch_d[i]))
                ratings.append(0)
        return uids,iids,domains,ratings



    def read_test_samples(self,test_neg_path):
        test_pos_samples = []
        test_neg_samples = []
        with open(test_neg_path) as f:
            for line in f:
                # each_neg_samples = []
                line = line.replace('\n','')
                each_content = line.split(' ')[:102]
                each_content = list(map(int,each_content))
                test_pos_samples.append([each_content[0],each_content[1],each_content[2]])
                test_neg_samples.append(each_content[3:102])
        return test_pos_samples,test_neg_samples




if __name__ == '__main__':

    # data_params = {
    #     'rating_path': '../datasets/debug_datasets/douban_book/book_review_data_new.csv',
    #     'user_rev_emb_path':'../datasets/debug_datasets/douban_book/doc_embs/book_user_doc_emb_32.npy',
    #     'item_rev_emb_path':'../datasets/debug_datasets/douban_book/doc_embs/book_item_doc_emb_32.npy',
    #     'args':args,
    #     'overlap_user_num':6
    # }
    # df = pd.read_csv('/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth/cloth_data.csv')
    # print(len(df))
    data_params = {
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_cloth/cloth_data.csv',
        'args':args,
        'overlap_user_num': 1284
    }
    load_data = Load_Data(**data_params)
    print('gg')