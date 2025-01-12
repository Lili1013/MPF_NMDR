import pandas as pd
from loguru import logger
from utils.para_parser import parse
import torch
import os
import numpy as np
# from model.mpf_mdr_filter import MPF_MDR_Model
from model.mpf_mdr_filter_v4 import MPF_MDR_Model
from utils.load_data_filter import Load_Data
from utils.evaluation_filter import calculate_hr_ndcg
from datetime import datetime
import pickle


def train(args, model, load_data,optimizer,device):
    # logger.info('start train')
    model.train()
    total_loss,total_main_loss, total_reg_loss= [],[],[]
    # cluster_assignments, cluster_center_dict = model.learn_general_multi_interest()
    # prototypes = torch.stack([torch.from_numpy(value) for value in cluster_center_dict.values()]).to(device)
    # top_n_index = model.calculate_top_n_similar_prototypes(model.mulit_modal_features, prototypes)
    for i, data in enumerate(load_data.train_loader, 0):
        # if i >= 1:
        #     break
        # logger.info(i)
        batch_nodes_u,batch_nodes_v,batch_nodes_d,batch_nodes_r = data
        batch_nodes_u, batch_nodes_v, domain_list, labels_list = load_data.generate_train_instances(batch_nodes_u.tolist(),
                                                                                        batch_nodes_v.tolist(),batch_nodes_d.tolist())
        batch_nodes_u, batch_nodes_v, batch_nodes_d,labels_list = torch.tensor(batch_nodes_u), torch.tensor(batch_nodes_v), \
                                                                    torch.tensor(domain_list,dtype=int),torch.tensor(labels_list)
        optimizer.zero_grad()
        prob,p_u,h_u,p_d = model.forward(batch_nodes_u.to(device),batch_nodes_v.to(device),
                                         batch_nodes_d.to(device))
        main_loss = model.calculate_main_loss(prob,labels_list.to(device))
        # reg_loss = torch.norm(torch.matmul(p_u.T,h_u))**2
        # cl_loss = model.calculate_cl_loss(p_d,p_u,p_v,batch_nodes_d.to(device))
        loss = main_loss
        # logger.info(f'batch main loss:{main_loss.item()}, cl loss:{cl_loss.item()}')
        total_loss.append(loss.item())
        total_main_loss.append(main_loss.item())
        # total_reg_loss.append(reg_loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
    total_loss.append(sum(total_loss)/len(total_loss))
    total_main_loss.append(sum(total_main_loss)/len(total_main_loss))
    # total_reg_loss.append(sum(total_reg_loss)/len(total_reg_loss))
    return total_loss,total_main_loss

def test(args,load_data,model,device):
    model.eval()
    with torch.no_grad():
        # pos_samples,neg_samples = local_data.generate_test_instances(list(local_data.test_df['userID'].values),list(local_data.test_df['itemID'].values))
        pos_samples_zero_shot, neg_samples_zero_shot = load_data.test_pos_samples_zero_shot,load_data.test_neg_samples_zero_shot
        pos_samples_few_shot, neg_samples_few_shot = load_data.test_pos_samples_few_shot, load_data.test_neg_samples_few_shot
        pos_samples_warm_shot, neg_samples_warm_shot = load_data.test_pos_samples_warm_shot, load_data.test_neg_samples_warm_shot
        (hr_zero_shot, ndcg_zero_shot,mrr_zero_shot,hr_5_zero_shot,ndcg_5_zero_shot,mrr_5_zero_shot,hr_1_zero_shot,ndcg_1_zero_shot,mrr_1_zero_shot) = calculate_hr_ndcg(model=model,test_ratings=pos_samples_zero_shot,test_negatives=neg_samples_zero_shot,
                                           k=args.top_k,device=device,user_cat='0')
        (hr_few_shot, ndcg_few_shot, mrr_few_shot, hr_5_few_shot, ndcg_5_few_shot, mrr_5_few_shot, hr_1_few_shot,
         ndcg_1_few_shot, mrr_1_few_shot) = calculate_hr_ndcg(model=model, test_ratings=pos_samples_few_shot,
                                                              test_negatives=neg_samples_few_shot,
                                                              k=args.top_k, device=device,
                                                              user_cat='1')
        (hr_warm_shot, ndcg_warm_shot, mrr_warm_shot, hr_5_warm_shot, ndcg_5_warm_shot, mrr_5_warm_shot, hr_1_warm_shot,
         ndcg_1_warm_shot, mrr_1_warm_shot) = calculate_hr_ndcg(model=model, test_ratings=pos_samples_warm_shot,
                                                                test_negatives=neg_samples_warm_shot,
                                                                k=args.top_k, device=device,
                                                                user_cat='2')
        # hr = sum(hits) / len(hits)
        # ndcg = sum(ndcgs) / len(ndcgs)
    return (hr_zero_shot, ndcg_zero_shot,mrr_zero_shot,hr_5_zero_shot,ndcg_5_zero_shot,mrr_5_zero_shot,hr_1_zero_shot,ndcg_1_zero_shot,mrr_1_zero_shot,
            hr_few_shot, ndcg_few_shot, mrr_few_shot, hr_5_few_shot, ndcg_5_few_shot, mrr_5_few_shot, hr_1_few_shot,ndcg_1_few_shot, mrr_1_few_shot,
            hr_warm_shot, ndcg_warm_shot, mrr_warm_shot, hr_5_warm_shot, ndcg_5_warm_shot, mrr_5_warm_shot, hr_1_warm_shot,ndcg_1_warm_shot, mrr_1_warm_shot)


if __name__ == '__main__':
    args = parse()
    logger.info(f'parameter settings: batch_size:{args.batch_size},lr:{args.lr},prompt_dim:{args.prompt_dim}, multi_modal_dim: {args.multi_modal_dim},'
                f'alpha:{args.alpha},delta:{args.delta},k:{args.k},n:{args.n},domain_num:{args.domain_num},domain_names:{args.domain_names}'
                f'top_k:{args.top_k},epochs:{args.epochs}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(0)
    domains = f'book_movie_music'
    path = f'../../../v0/datasets/douban_review'
    load_data_params = {
        'rating_path': f'{path}/pretrain/all_domain_inter.csv',
        'text_feat_path': f'{path}/pretrain/all_domain_text_features.npy',
        # 'visual_feat_path':f'{path}/pretrain/all_domain_image_feature_filter.npy',
        'train_path': f'{path}/pretrain/train.csv',
        # 'test_path_zero_shot':'datasets/amazon_review/health_cloth_beauty_pretrain/test_zero_shot_filter.csv',
        # 'test_path_few_shot': 'datasets/amazon_review/health_cloth_beauty_pretrain/test_few_shot_filter.csv',
        'args': args,
        'test_neg_path_zero_shot': f'{path}/pretrain/test_zero_shot.txt',
        'test_neg_path_few_shot': f'{path}/pretrain/test_few_shot.txt',
        'test_neg_path_warm_shot': f'{path}/pretrain/test_warm_shot.txt',
    }
    load_data = Load_Data(**load_data_params)
    user_inter_dict = load_data.generate_user_inter_lists(load_data.train_df)

    params = {
        'user_num':load_data.num_users,
        'item_num':load_data.num_items,
        'text_feat':load_data.text_emb,
        # 'visual_feat':load_data.visual_emb,
        'alpha': args.alpha,
        'ssl_temp': args.tau,
        'prompt_dim':args.prompt_dim,
        'multi_modal_dim':args.multi_modal_dim,
        'device':device,
        'domain_num':args.domain_num,
        'user_inter_dict':user_inter_dict,
        'df':load_data.df_rating,
        'ssl_temp':args.tau,
        'k':args.k,
        'aug_way':args.aug_way,
        'delta':args.delta,
        'n':args.n
    }
    model = MPF_MDR_Model(**params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info('start training and testing')
    best_hr = 0.0
    best_ndcg = 0.0
    endure_num = 0
    for epoch in range(1, args.epochs + 1):
        if endure_num >= 5:
            logger.info('End')
            break
        total_loss,total_main_loss = train(args, model, load_data,optimizer,device)
        epoch_loss = sum(total_loss)/len(total_loss)
        epoch_main_loss = sum(total_main_loss)/len(total_main_loss)
        # epoch_cl_loss = sum(total_cl_loss)/len(total_cl_loss)
        logger.info('epoch {}, training total loss is {}, main loss is {}'
                    .format(epoch,epoch_loss,epoch_main_loss))
        (hr_zero_shot, ndcg_zero_shot, mrr_zero_shot, hr_5_zero_shot, ndcg_5_zero_shot, mrr_5_zero_shot, hr_1_zero_shot, ndcg_1_zero_shot, mrr_1_zero_shot,
         hr_few_shot, ndcg_few_shot, mrr_few_shot, hr_5_few_shot, ndcg_5_few_shot, mrr_5_few_shot, hr_1_few_shot, ndcg_1_few_shot, mrr_1_few_shot,
         hr_warm_shot, ndcg_warm_shot, mrr_warm_shot, hr_5_warm_shot, ndcg_5_warm_shot, mrr_5_warm_shot, hr_1_warm_shot, ndcg_1_warm_shot, mrr_1_warm_shot) = test(args,load_data,model,device)
        hr = (hr_zero_shot+hr_few_shot+hr_warm_shot)/3
        ndcg= (ndcg_warm_shot+ndcg_few_shot+ndcg_zero_shot)/3
        mrr = (mrr_zero_shot+mrr_few_shot+mrr_warm_shot)/3
        hr_5 = (hr_5_zero_shot + hr_5_few_shot + hr_5_warm_shot) / 3
        ndcg_5 = (ndcg_5_warm_shot + ndcg_5_few_shot + ndcg_5_zero_shot)/3
        mrr_5 = (mrr_5_zero_shot + mrr_5_few_shot + mrr_5_warm_shot) / 3
        hr_1 = (hr_1_zero_shot + hr_1_few_shot + hr_1_warm_shot) / 3
        ndcg_1 = (ndcg_1_warm_shot + ndcg_1_few_shot + ndcg_1_zero_shot)/3
        mrr_1 = (mrr_1_zero_shot + mrr_1_few_shot + mrr_1_warm_shot) / 3
        logger.info(f"Hr_zero_shot:{hr_zero_shot}, NDCG:{ndcg_zero_shot},Mrr_zero_shot:{mrr_zero_shot}, Hr_5_zero_shot:{hr_5_zero_shot},Ndcg_5_zero_shot:{ndcg_5_zero_shot},Mrr_5_zero_shot:{mrr_5_zero_shot}, Hr_1_zero_shot:{hr_1_zero_shot}, Ndcg_1_zero_shot:{ndcg_1_zero_shot}, Mrr_zero_shot:{mrr_1_zero_shot}")
        logger.info(
            f"Hr_warm_shot:{hr_warm_shot}, NDCG:{ndcg_warm_shot},Mrr_warm_shot:{mrr_warm_shot}, Hr_5_warm_shot:{hr_5_warm_shot},Ndcg_5_warm_shot:{ndcg_5_warm_shot},Mrr_5_warm_shot:{mrr_5_warm_shot}, Hr_1_warm_shot:{hr_1_warm_shot}, Ndcg_1_warm_shot:{ndcg_1_warm_shot}, Mrr_warm_shot:{mrr_1_warm_shot}")
        logger.info(
            f"Hr_few_shot:{hr_few_shot}, NDCG:{ndcg_few_shot},Mrr_few_shot:{mrr_few_shot}, Hr_5_few_shot:{hr_5_few_shot},Ndcg_5_few_shot:{ndcg_5_few_shot},Mrr_5_few_shot:{mrr_5_few_shot}, Hr_1_few_shot:{hr_1_few_shot}, Ndcg_1_few_shot:{ndcg_1_few_shot}, Mrr_few_shot:{mrr_1_few_shot}")
        logger.info(
            f"Hr:{hr}, NDCG:{ndcg},Mrr:{mrr}, Hr_5:{hr_5},Ndcg_5:{ndcg_5},Mrr_5:{mrr_5}, Hr_1:{hr_1}, Ndcg_1:{ndcg_1}, Mrr:{mrr_1}")
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            endure_num = 0
            #save parameters
            # torch.save(model.state_dict(), f'best_model_filter/pretrain/b_2048_d_256_k_5_aug_way_noise_alpha_range/health_cloth_beauty_pretrain_alpha_{args.alpha}_10_02.pth')
            # np.save(f'best_model_filter/pretrain/b_2048_d_256_k_5_aug_way_noise_alpha_range/health_cloth_beauty_cluster_assignments_alpha_{args.alpha}_10_02.npy',cluster_assignments)
            # with open(f'best_model_filter/pretrain/b_2048_d_256_k_5_aug_way_noise_alpha_range/health_cloth_beauty_cluster_center_dict_alpha_{args.alpha}_10_02.pkl','wb') as f:
            #     pickle.dump(cluster_center_dict,f)
            torch.save(model.state_dict(),
                       f'best_model/{domains}_b_{args.batch_size}_d_{args.prompt_dim}_'
                       f'delta_{args.delta}_k_{args.k}_n_{args.n}_12_18.pth')
            # np.save(
            #     f'best_model_v1/{domains}_b_{args.batch_size}_d_{args.prompt_dim}_'
            #            f'alpha_{args.alpha}_k_{args.k}_aug_{args.aug_way}_cluster_assignments_11_08.npy',
            #     cluster_assignments)
            # with open(
            #         f'best_model_v4/{domains}_b_{args.batch_size}_d_{args.prompt_dim}_'
            #            f'delta_{args.delta}_k_{args.k}_n_{args.n}_cluster_center_dict_11_08.pkl',
            #         'wb') as f:
            #     pickle.dump(cluster_center_dict, f)
        else:
            endure_num += 1


