import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

def split_user_group(path):
    df = pd.read_csv(path)
    user_inter_cnt = df.groupby('user_id')['item_id'].count().reset_index()
    user_inter_cnt.columns = ['user_id','inter_cnt']
    print(len(user_inter_cnt[user_inter_cnt['inter_cnt']<=5]))
    print(len(user_inter_cnt[(user_inter_cnt['inter_cnt']>5)&(user_inter_cnt['inter_cnt']<10)]))
    print(len(user_inter_cnt[user_inter_cnt['inter_cnt']>=10]))
    df_new = pd.merge(df,user_inter_cnt,on='user_id',how='left')
    df_zero_shot = df_new[df_new['inter_cnt']<=5]
    df_few_shot=df_new[(df_new['inter_cnt']>5)&(df_new['inter_cnt']<10)]
    df_few_shot['user_cat'] = '1'
    df_warm_shot = df_new[df_new['inter_cnt']>=10]
    df_warm_shot['user_cat'] = '2'
    # df_zero_shot = df_zero_shot.groupby('user_id').first().reset_index()
    df_zero_shot['user_cat'] = '0'
    df_all = pd.concat([df_zero_shot,df_few_shot,df_warm_shot])
    print(len(df_all))
    return df_all

if __name__ == '__main__':
    # df_health = split_user_group(path='../datasets/amazon_review/health_inter_non_overlap.csv')
    # df_health.to_csv('../datasets/amazon_review/health_inter_non_overlap_filter.csv',index=False)
    # df_cloth = split_user_group(path='../datasets/amazon_review/cloth_inter_non_overlap.csv')
    # df_cloth.to_csv('../datasets/amazon_review/cloth_inter_non_overlap_filter.csv',index=False)
    # df_beauty = split_user_group(path='../datasets/amazon_review/beauty_inter_non_overlap.csv')
    # df_beauty.to_csv('../datasets/amazon_review/beauty_inter_non_overlap_filter.csv',index=False)
    path = f'../datasets/douban_review'
    # df_1 = split_user_group(path=f'{path}/book/book_review_data_map.csv')
    # df_1.to_csv(f'{path}/book/book_inter_non_overlap.csv', index=False)
    # df_2 = split_user_group(path=f'{path}/movie/movie_review_data_map.csv')
    # df_2.to_csv(f'{path}/movie/movie_inter_non_overlap.csv', index=False)
    df_3 = split_user_group(path=f'{path}/music/music_review_data_map.csv')
    df_3.to_csv(f'{path}/music/music_inter_non_overlap.csv', index=False)


