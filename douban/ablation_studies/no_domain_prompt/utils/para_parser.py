import argparse

def parse():
    parser = argparse.ArgumentParser(description='MPF_MDR')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--prompt_dim', type=int, default=256, metavar='N', help='gnn embedding size')
    # parser.add_argument('--disen_embed_dim', type=int, default=256, metavar='N', help='disen feature embedding size')
    parser.add_argument('--multi_modal_dim', type=int, default=256, metavar='N', help='review embedding size')
    # parser.add_argument('--n_layers', type=int, default=3, metavar='N', help='the number of GNN layers')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2_regularization', type=float, default=0.0001, metavar='weight decay', help='the weight decay of optimizer')
    # parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    # parser.add_argument('--rounds', type=int, default=100, metavar='N', help='number of rounds for the communication between the server and clients')
    parser.add_argument('--alpha', type=float, default=0.2 , metavar='N',
                        help='cl loss')
    parser.add_argument('--delta', type=float, default=0.5, metavar='N',
                        help='the generalization threshold')
    parser.add_argument('--k', type=int, default=10, metavar='N', help='the cluster number')
    parser.add_argument('--n', type=int, default=1, metavar='N', help='the top-n for the representative prototypes')
    parser.add_argument('--aug_way', type=str, default='noise', metavar='N', help='the feature augmentation method.')
    # parser.add_argument('--beta', type=float, default=0.4, metavar='N',
    #                     help='beta distribution')
    # parser.add_argument('--gamma', type=float, default=0.2, metavar='N',
    #                     help='intral cl loss')
    # parser.add_argument('--lap_noise', type=float, default=0.01, metavar='N',
    #                     help='the laplace noise')
    # parser.add_argument('--lap_noise', type=float, default=0.3, metavar='N',
    #                     help='the laplace noise, eta')
    # parser.add_argument('--C', type=float, default=0.1, metavar='N',
    #                     help='the clipping threshold of sensitivity')
    parser.add_argument('--train_neg_num', type=int, default=1, metavar='N', help='the number of training negative sample')
    parser.add_argument('--test_neg_num', type=int, default=99, metavar='N', help='the number of testing negative sample')
    # parser.add_argument('--pos_neg_num', type=int, default=4, metavar='N',
    #                     help='the number of potential positive sample')
    parser.add_argument('--top_k', type=int, default=10, metavar='N', help='the length of recommendation lists')
    parser.add_argument('--domain_num', type=int, default=3, metavar='N', help='the number of domains')
    # parser.add_argument('--client_names',type=list,default = ['amazon_phone','amazon_sport'],metavar='N', help='the clients')
    # parser.add_argument('--client_names', type=list, default=['amazon_elec_cloth_phone_elec', 'amazon_elec_cloth_phone_phone'],
    #                     metavar='N', help='the clients')
    parser.add_argument('--domain_names', type=list,
                        default=['amazon_health', 'amazon_cloth','amazon_beauty'],
                        metavar='N', help='the domains')

    parser.add_argument('--tau', type=float, default=0.2,
                        metavar='N', help='the temperature of contrastive loss')
    parser.add_argument('--mode', type=str, default='prompt-tune',
                        metavar='N', help='the train way : pretrain or prompt tuning')
    parser.add_argument('--tune_domain', type=str, default='health',
                        metavar='N', help='the target tuning domain for the prompt tuning')
    # parser.add_argument('--interpo_way', type=str, default='gaus',
    #                     metavar='N', help='the interpolation way for potential interest mixing module')

    # parser.add_argument('--save_model_path', type=list,
    #             default=['best_models_02_23/douban/book_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #                      'best_models_02_23/douban/movie_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #                      'best_models_02_23/douban/music_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model'])
    # parser.add_argument('--train_model_weights_path', type=list,
    # default=['best_models_02_23/douban/train_model_weights_epochs_douban_book_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #         'best_models_02_23/douban/train_model_weights_epochs_douban_movie_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #         'best_models_02_23/douban/train_model_weights_epochs_douban_music_emb_32_disen_32_c_30_alpha_0.1_beta_0.1_lr_0.001_b_128/'], metavar='N',
    #                     help='the best source path')
    # parser.add_argument('--save_model_path', type=list, default=[
    #     'best_models/amazon_review/amazon_elec_cloth_phone/elec_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model',
    #     'best_models/amazon_review/amazon_elec_cloth_phone/cloth_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model',
    # 'best_models/amazon_review/amazon_elec_cloth_phone/phone_emb_32_disen_32_c_20_alpha_0.1_beta_0.1_b_128_02_16.model'],
    # metavar='N', help='the bets model path')
    # parser.add_argument('--save_model_path', type=list, default=[
    #     'best_models/amazon_review/amazon_phone_sport/phone_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model',
    #     'best_models/amazon_review/amazon_phone_sport/sport_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128_02_25.model'],
    #                     metavar='N', help='the bets model path')
    # parser.add_argument('--train_model_weights_path', type=list,
    # default=['best_models_02_23/amazon_review/amazon_phone_sport/train_model_weights_epochs_amazon_phone_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    # 'best_models_02_23/amazon_review/amazon_phone_sport/train_model_weights_epochs_amazon_sport_emb_32_disen_32_c_10_alpha_0.1_beta_0.1_lr_0.001_b_128/',
    #                                                  ], metavar='N',
    #                                         help='the best source path')


    args = parser.parse_args()
    return args