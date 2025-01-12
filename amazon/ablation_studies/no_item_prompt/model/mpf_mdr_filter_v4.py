import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering



class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """

    def __init__(self, input_size, output_size, device, dropout=0.0):
        super(PWLayer, self).__init__()
        self.device = device

        self.dropout = nn.Dropout(p=dropout).to(self.device)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True).to(self.device)
        self.lin = nn.Linear(input_size, output_size, bias=False).to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MPF_MDR_Model(nn.Module):
    def __init__(self, **params):
        super(MPF_MDR_Model, self).__init__()
        self.device = params['device']
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.prompt_dim = params['prompt_dim']
        self.multi_modal_dim = params['multi_modal_dim']
        self.domain_num = params['domain_num']
        self.pre_text_features = params['text_feat']
        self.pre_visual_features = params['visual_feat'].to(torch.float32)
        self.user_inter_dict = params['user_inter_dict']
        self.df = params['df']
        self.ssl_temp = params['ssl_temp']
        self.k = params['k']
        self.aug_way = params['aug_way']
        self.delta = params['delta']
        self.n = params['n']
        # self.pre_visual_features = params['visual_feat']
        # PW
        pw_layer_text = PWLayer(input_size=384, output_size=self.multi_modal_dim, device=self.device)
        self.text_features = pw_layer_text.forward(self.pre_text_features.to(self.device))
        pw_layer_visual = PWLayer(input_size=768, output_size=self.multi_modal_dim, device=self.device)
        # pw_layer_visual = PWLayer(input_size=4096, output_size=self.multi_modal_dim, device=self.device)
        self.visual_features = pw_layer_visual.forward(self.pre_visual_features.to(self.device))
        self.mulit_modal_features = torch.cat([self.text_features, self.visual_features], dim=0)
        # domain prompt
        self.domain_prompt = nn.Embedding(self.domain_num, self.prompt_dim).to(self.device)
        nn.init.xavier_uniform_(self.domain_prompt.weight)
        # user prompt
        self.user_prompt = nn.Embedding(self.user_num, self.prompt_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_prompt.weight)
        # item prompt
        self.item_prompt = nn.Embedding(self.item_num, self.prompt_dim).to(self.device)
        nn.init.xavier_uniform_(self.item_prompt.weight)

        #weight for user prompt and user general interests
        self.beta = nn.Parameter(torch.rand(1,self.prompt_dim)).to(self.device)
        # mlp layers
        self.mlp_layer_1 = nn.Linear(self.multi_modal_dim * 2, self.multi_modal_dim).to(self.device)
        self.mlp_norm_1 = nn.BatchNorm1d(self.multi_modal_dim).to(self.device)
        self.mlp_layer_2 = nn.Linear(self.multi_modal_dim, int(self.multi_modal_dim/2)).to(self.device)
        self.mlp_norm_2 = nn.BatchNorm1d(int(self.multi_modal_dim/2)).to(self.device)
        self.pred_layer = nn.Linear(int(self.multi_modal_dim/2), 1).to(self.device)
        self.criterion = nn.BCELoss()

    # def forward(self, nodes_u, nodes_v, nodes_d, cluster_assignments, cluster_center_dict):
    #     # Prompt embeddings
    #     p_u = self.user_prompt(nodes_u)  # user prompt
    #     p_v = self.item_prompt(nodes_v)  # item prompt
    #     p_d = self.domain_prompt(nodes_d)  # domain prompt
    # 
    #     # Batch processing
    #     batch_users = nodes_u.tolist()
    #     batch_items = nodes_v.tolist()
    # 
    #     # Cache prototypes as a tensor
    #     prototypes = torch.stack([torch.from_numpy(value) for value in cluster_center_dict.values()]).to(self.device)
    # 
    #     # Create inter_items_tensor for batch users excluding their current items
    #     inter_items = [torch.tensor(self.user_inter_dict[user], device=self.device) for user in batch_users]
    #     inter_items_tensor = torch.nn.utils.rnn.pad_sequence(inter_items, batch_first=True, padding_value=-1)
    # 
    #     # Mask to exclude the current item
    #     inter_items_tensor = inter_items_tensor.masked_fill(
    #         inter_items_tensor == torch.tensor(batch_items, device=self.device).unsqueeze(1), -1)
    #     inter_items_tensor = inter_items_tensor[inter_items_tensor != -1]
    # 
    #     # Multi-modal feature extraction for all inter items
    #     inter_item_text_features = self.text_features[inter_items_tensor]
    #     inter_item_visual_features = self.visual_features[inter_items_tensor]
    # 
    #     # Batch prototype calculations (no loop)
    #     top_n_text_prototypes = self.calculate_top_n_similar_prototypes(inter_item_text_features, prototypes)
    #     top_n_visual_prototypes = self.calculate_top_n_similar_prototypes(inter_item_visual_features, prototypes)
    #     top_n_merge_index = torch.unique(torch.cat((top_n_text_prototypes, top_n_visual_prototypes), dim=1), dim=1)
    # 
    #     # Compute user-general interests by batch-wise aggregation
    #     general_user_interest_list = torch.mean(prototypes[top_n_merge_index], dim=1)
    # 
    #     # Combine features for all batches
    #     h_u = torch.cat(general_user_interest_list, dim=0)
    #     e_t = self.text_features[nodes_v]
    #     e_v = self.visual_features[nodes_v]
    #     h_u_hat = p_u + h_u + p_d
    #     e_i_hat = p_v + e_t + e_v
    #     inter_feat = torch.cat([h_u_hat, e_i_hat], dim=1)
    # 
    #     # Remaining layers remain unchanged.
    #     inter_feat_1 = self.mlp_norm_1(F.relu(self.mlp_layer_1(inter_feat)))
    #     inter_feat_2 = self.mlp_norm_2(F.relu(self.mlp_layer_2(inter_feat_1)))
    #     pred = F.sigmoid(self.pred_layer(inter_feat_2))
    # 
    #     return pred.squeeze(), p_u, h_u, p_d

    def calculate_top_n_similar_prototypes(self, multi_modal_features, prototypes):
        """
        :param multi_modal_features: shape: (item_num, feature_dim)
        :param prototypes: shape: (prototype_num, feature_dim)
        :return: top_n_indices
        """
        similarity_matrix = torch.matmul(multi_modal_features, prototypes.T)
        _, top_n_indices = torch.topk(similarity_matrix, self.n, dim=1)
        return top_n_indices

    def forward(self, nodes_u, nodes_v, nodes_d, top_n_index, prototypes):
        p_u = self.user_prompt(nodes_u)  # user prompt
        p_v = self.item_prompt(nodes_v)  # item prompt
        p_d = self.domain_prompt(nodes_d)  # domain prompt

        # Retrieve user-item interaction history in a batched manner
        inter_items_cache = {user: torch.tensor(self.user_inter_dict[user], device=self.device) for user in
                             nodes_u.tolist()}
        inter_items_batch = [
            torch.tensor([x for x in inter_items_cache[user].tolist() if x != item], device=self.device)
            for user, item in zip(nodes_u.tolist(), nodes_v.tolist())]

        # Get inter-item indices for text and visual features
        inter_top_n_index_text = [top_n_index[inter_items] for inter_items in inter_items_batch]
        inter_top_n_index_visual = [top_n_index[inter_items + self.item_num] for inter_items in inter_items_batch]

        # Flatten and process indices in a batched way
        inter_top_n_merge_index = [torch.unique(torch.cat((text_idx, visual_idx), dim=0).flatten())
                                   for text_idx, visual_idx in zip(inter_top_n_index_text, inter_top_n_index_visual)]

        # Retrieve prototypes for batch and compute general user interests
        rep_prototypes_batch = [torch.mean(prototypes[merge_idx], dim=0).unsqueeze(0) for merge_idx in
                                inter_top_n_merge_index]
        h_u = torch.cat(rep_prototypes_batch, dim=0)
        h_u_final = h_u

        e_t = self.text_features[nodes_v]
        e_v = self.visual_features[nodes_v]

        # Combine embeddings
        h_u_hat = h_u_final + p_d
        e_i_hat = p_v + e_t + e_v
        inter_feat = torch.cat([h_u_hat, e_i_hat], dim=1)

        # Pass through MLP layers
        inter_feat_1 = self.mlp_norm_1(F.relu(self.mlp_layer_1(inter_feat)))
        inter_feat_2 = self.mlp_norm_2(F.relu(self.mlp_layer_2(inter_feat_1)))
        pred = torch.sigmoid(self.pred_layer(inter_feat_2))

        return pred.squeeze(), p_u, h_u, p_d

    def forward_zero_shot(self, nodes_u, nodes_v, nodes_d, cluster_assignments, cluster_center_dict):
        p_u = torch.zeros(len(nodes_u), self.prompt_dim, dtype=torch.float32, device=self.device)  # user prompt
        p_v = self.item_prompt(nodes_v)  # item prompt
        p_d = self.domain_prompt(nodes_d)  # domain prompt
        h_u = torch.zeros(len(nodes_u), self.multi_modal_dim, dtype=torch.float32, device=self.device)
        e_t = self.text_features[nodes_v]
        e_v = self.visual_features[nodes_v]
        h_u_hat = p_d
        # e_i_hat = torch.cat([p_v, e_t, e_v], dim=1)
        e_i_hat = p_v+e_t+e_v
        inter_feat = torch.cat([h_u_hat, e_i_hat], dim=1)
        inter_feat_1 = self.mlp_norm_1(F.relu(self.mlp_layer_1(inter_feat)))
        inter_feat_2 = self.mlp_norm_2(F.relu(self.mlp_layer_2(inter_feat_1)))
        pred = F.sigmoid(self.pred_layer(inter_feat_2))
        return pred.squeeze(), p_u, h_u, p_d

    

    def learn_general_multi_interest(self):
        # Example input
        text_emb = self.text_features.detach().cpu().numpy()  # (item_num, text_feat_dim)
        visual_emb = self.visual_features.detach().cpu().numpy()  # (item_num, visual_feat_dim)

        # Combine text and visual embeddings
        combined_emb = np.vstack((text_emb, visual_emb))  # Shape: (item_num, text_feat_dim + visual_feat_dim)

        # Perform K-Means clustering
        n_clusters = self.k  # Adjust based on your needs
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(combined_emb)
        cluster_centers = kmeans.cluster_centers_

        # Identify clusters that contain features from at least two domains
        domain_per_cluster = {i: set() for i in range(n_clusters)}

        # Fill domain_per_cluster with the domains present in each cluster
        for idx, cluster_id in enumerate(cluster_assignments):
            if idx >= self.item_num:
                idx = idx - self.item_num
            domain_per_cluster[cluster_id].add(
                self.df[self.df['itemID'] == idx]['domain_id'].iloc[0])

        # Filter clusters to only include those with at least two domains
        valid_clusters = {cid: cluster_centers[cid] for cid, domains in domain_per_cluster.items() if len(domains)/self.domain_num > self.delta}

        # Create a dictionary for quick lookup of cluster centers
        cluster_center_dict = {cid: center for cid, center in valid_clusters.items()}
        return cluster_assignments, cluster_center_dict

    def feature_augment(self, emb):
        if self.aug_way == 'noise':
            # gaussian noise augment
            noise_level = 0.01
            noise = torch.randn_like(emb) * noise_level
            aug_emb = emb + noise
        elif self.aug_way == 'dropout':
            #dropout augment
            dropout_layer = nn.Dropout(p=0.2)
            aug_emb = dropout_layer(emb)
        else:
            #masking augment
            mask_fraction = 0.1
            mask = torch.rand(emb.shape) > mask_fraction
            aug_emb = emb*mask
        return aug_emb

    # def calculate_total_loss(self,pred,label):
    #     main_loss = self.calculate_main_loss(pred,label)
    #     cl_loss = self.calculate_cl_loss()

    def calculate_main_loss(self, pred, label):
        source_pred_loss = self.criterion(pred, label.to(torch.float32))
        return source_pred_loss

    def calculate_cl_loss(self, p_d, p_u, h_u, batch_domain_id):
        # aug_p_d = self.feature_augment(p_d)
        # aug_p_u = self.feature_augment(p_u)
        aug_h_u = self.feature_augment(h_u)

        # neg_p_u = p_u[torch.randperm(len(p_u))]
        neg_h_u = h_u[torch.randperm(len(h_u))]

        # batch_domain_id_tensor = batch_domain_id.view(-1, 1)
        # domain_mask = batch_domain_id_tensor != batch_domain_id_tensor.T
        # valid_domains = torch.where(domain_mask, batch_domain_id_tensor.T, torch.tensor(float('inf')))
        # sorted_valid_domains, _ = valid_domains.sort(dim=1)
        # neg_domain_ids_1 = sorted_valid_domains[:, 0]
        # neg_domain_ids_2 = sorted_valid_domains[:, 1]
        # neg_p_d_1 = self.domain_prompt(neg_domain_ids_1.to(torch.int32))
        # neg_p_d_2 = self.domain_prompt(neg_domain_ids_2.to(torch.int32))
        #
        # norm_p_u = F.normalize(p_u, p=2, dim=1)
        # norm_aug_p_u = F.normalize(aug_p_u, p=2, dim=1)
        # norm_neg_p_u = F.normalize(neg_p_u, p=2, dim=1)
        # pos_score_p_u = torch.sum(torch.mul(norm_p_u, norm_aug_p_u), dim=1)
        # neg_score_p_u = torch.sum(torch.mul(norm_p_u, norm_neg_p_u), dim=1)
        # pos_score_p_u = torch.exp(pos_score_p_u / self.ssl_temp)
        # neg_score_p_u = torch.exp(neg_score_p_u / self.ssl_temp)
        # L_cl_p_u = -torch.sum(torch.log(pos_score_p_u / (pos_score_p_u + neg_score_p_u)))

        norm_h_u = F.normalize(h_u, p=2, dim=1)
        norm_aug_h_u = F.normalize(aug_h_u, p=2, dim=1)
        norm_neg_h_u = F.normalize(neg_h_u, p=2, dim=1)
        pos_score_h_u = torch.sum(torch.mul(norm_h_u, norm_aug_h_u), dim=1)
        neg_score_h_u = torch.sum(torch.mul(norm_h_u, norm_neg_h_u), dim=1)
        pos_score_h_u = torch.exp(pos_score_h_u / self.ssl_temp)
        neg_score_h_u = torch.exp(neg_score_h_u / self.ssl_temp)
        L_cl_h_u = -torch.sum(torch.log(pos_score_h_u / (pos_score_h_u + neg_score_h_u)))

        # norm_p_d = F.normalize(p_d, p=2, dim=1)
        # norm_aug_p_d = F.normalize(aug_p_d, p=2, dim=1)
        # norm_neg_p_d_1 = F.normalize(neg_p_d_1, p=2, dim=1)
        # norm_neg_p_d_2 = F.normalize(neg_p_d_2, p=2, dim=1)
        # pos_score_p_d = torch.sum(torch.mul(norm_p_d, norm_aug_p_d), dim=1)
        # neg_score_p_d_1 = torch.sum(torch.mul(norm_p_d, norm_neg_p_d_1), dim=1)
        # neg_score_p_d_2 = torch.sum(torch.mul(norm_p_d, norm_neg_p_d_2), dim=1)
        # pos_score_p_d = torch.exp(pos_score_p_d / self.ssl_temp)
        # neg_score_p_d_1 = torch.exp(neg_score_p_d_1 / self.ssl_temp)
        # neg_score_p_d_2 = torch.exp(neg_score_p_d_2 / self.ssl_temp)
        # L_cl_p_d = -torch.sum(torch.log(pos_score_p_d / (pos_score_p_d + neg_score_p_d_1))) - torch.sum(
        #     torch.log(pos_score_p_d / (pos_score_p_d + neg_score_p_d_2)))

        return L_cl_h_u
    def calculate_cl_loss_single_train(self, p_d, p_u, h_u, batch_domain_id):
        # aug_p_d = self.feature_augment(p_d)
        aug_p_u = self.feature_augment(p_u)
        aug_h_u = self.feature_augment(h_u)

        neg_p_u = p_u[torch.randperm(len(p_u))]
        neg_h_u = h_u[torch.randperm(len(h_u))]

        # batch_domain_id_tensor = batch_domain_id.view(-1, 1)
        # domain_mask = batch_domain_id_tensor != batch_domain_id_tensor.T
        # valid_domains = torch.where(domain_mask, batch_domain_id_tensor.T, torch.tensor(float('inf')))
        # sorted_valid_domains, _ = valid_domains.sort(dim=1)
        # neg_domain_ids_1 = sorted_valid_domains[:, 0]
        # neg_domain_ids_2 = sorted_valid_domains[:, 1]
        # neg_p_d_1 = self.domain_prompt(neg_domain_ids_1.to(torch.int32))
        # neg_p_d_2 = self.domain_prompt(neg_domain_ids_2.to(torch.int32))

        norm_p_u = F.normalize(p_u, p=2, dim=1)
        norm_aug_p_u = F.normalize(aug_p_u, p=2, dim=1)
        norm_neg_p_u = F.normalize(neg_p_u, p=2, dim=1)
        pos_score_p_u = torch.sum(torch.mul(norm_p_u, norm_aug_p_u), dim=1)
        neg_score_p_u = torch.sum(torch.mul(norm_p_u, norm_neg_p_u), dim=1)
        pos_score_p_u = torch.exp(pos_score_p_u / self.ssl_temp)
        neg_score_p_u = torch.exp(neg_score_p_u / self.ssl_temp)
        L_cl_p_u = -torch.sum(torch.log(pos_score_p_u / (pos_score_p_u + neg_score_p_u)))

        norm_h_u = F.normalize(h_u, p=2, dim=1)
        norm_aug_h_u = F.normalize(aug_h_u, p=2, dim=1)
        norm_neg_h_u = F.normalize(neg_h_u, p=2, dim=1)
        pos_score_h_u = torch.sum(torch.mul(norm_h_u, norm_aug_h_u), dim=1)
        neg_score_h_u = torch.sum(torch.mul(norm_h_u, norm_neg_h_u), dim=1)
        pos_score_h_u = torch.exp(pos_score_h_u / self.ssl_temp)
        neg_score_h_u = torch.exp(neg_score_h_u / self.ssl_temp)
        L_cl_h_u = -torch.sum(torch.log(pos_score_h_u / (pos_score_h_u + neg_score_h_u)))

        # norm_p_d = F.normalize(p_d, p=2, dim=1)
        # norm_aug_p_d = F.normalize(aug_p_d, p=2, dim=1)
        # norm_neg_p_d_1 = F.normalize(neg_p_d_1, p=2, dim=1)
        # norm_neg_p_d_2 = F.normalize(neg_p_d_2, p=2, dim=1)
        # pos_score_p_d = torch.sum(torch.mul(norm_p_d, norm_aug_p_d), dim=1)
        # neg_score_p_d_1 = torch.sum(torch.mul(norm_p_d, norm_neg_p_d_1), dim=1)
        # neg_score_p_d_2 = torch.sum(torch.mul(norm_p_d, norm_neg_p_d_2), dim=1)
        # pos_score_p_d = torch.exp(pos_score_p_d / self.ssl_temp)
        # neg_score_p_d_1 = torch.exp(neg_score_p_d_1 / self.ssl_temp)
        # neg_score_p_d_2 = torch.exp(neg_score_p_d_2 / self.ssl_temp)
        # L_cl_p_d = -torch.sum(torch.log(pos_score_p_d / (pos_score_p_d + neg_score_p_d_1))) - torch.sum(
        #     torch.log(pos_score_p_d / (pos_score_p_d + neg_score_p_d_2)))

        return L_cl_p_u + L_cl_h_u










