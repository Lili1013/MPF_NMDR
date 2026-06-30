import pandas as pd

import pandas as pd
import random
from loguru import logger
import numpy as np

def filter_g_k_one(data,k_user=5,k_item=5,u_name='user_id',i_name='item_id',y_name='rating'):
    '''
    delete the records that user and item interactions lower than k
    '''
    item_group = data.groupby(i_name).agg({y_name:'count'}) #every item has the number of ratings
    item_g10 = item_group[item_group[y_name]>=k_item].index
    data_new = data[data[i_name].isin(item_g10)]

    user_group = data_new.groupby(u_name).agg({y_name: 'count'})  # every item has the number of ratings
    user_g10 = user_group[user_group[y_name] >= k_user].index
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new

def review_data_ststistic(source_path,columns):
    df = pd.read_csv(source_path,delimiter='\t')
    df.rename(columns=columns,inplace=True)
    sample_num = len(df)
    user_num = len(df['user_id'].unique())
    item_num = len(df['item_id'].unique())
    df_new = df[['user_id','item_id','rating','labels','reviews']]
    return sample_num, user_num, item_num,df_new

def sample_movie_data(df):
    items = list(df['item_id'].unique())
    sample_items = random.sample(items,5000)
    df_new = df[df['item_id'].isin(sample_items)]
    print(len(df_new))
    return df_new

def replace_newline(value):
    return value.replace('\n',' ').replace('\r',' ')

if __name__ == '__main__':
    book_columns = {
            "user_id":"user_id",
            "book_id":"item_id",
            "rating":"rating",
            "labels":"labels",
            "comment":"reviews",
            "time":"time",
            "ID":"ID"
        }
    book_r_sample_num,book_r_user_num,book_r_item_num,df_book_new = review_data_ststistic(source_path='/data/lwang9/datasets/Douban_review/book/bookreviews_cleaned.txt',columns=book_columns)
    df_book_new = filter_g_k_one(df_book_new,k_user=5,k_item=5)
    df_book_new['reviews'] = df_book_new['reviews'].astype(str)
    df_book_new['reviews'] = df_book_new['reviews'].apply(replace_newline)
    # df_book_new.dropna(subset=['labels'],inplace=True)
    # df_book_new = df_book_new[df_book_new['labels'] != ' ']
    df_book_new['labels'].replace({np.nan:'书籍', ' ':'书籍'},inplace=True)
    logger.info(f'book num:{len(df_book_new)}')
    df_book_new.to_csv('../datasets/douban_review/book/book_review_data.csv',index=False)
    # df_book_new[['user_id','item_id','rating']].to_csv('../datasets/douban_review/book_inter.csv',index=False)
    # df_book_multi_modal = process_multi_modal_data(df_book_new,'书籍')
    # df_book_multi_modal.to_csv('../datasets/douban_review/book_multi_modal_data.csv',index=False)

    movie_columns = {"user_id":"user_id",
                         "movie_id":"item_id",
                         "rating":"rating",
                         "comment":"reviews",
                         "time":"time",
                         "labels":"labels",
                         "useful_num":"useful_num",
                         "CategoryID":"CategoryID",
                         "ID":"ID"}
    movie_r_sample_num, movie_r_user_num, movie_r_item_num, df_movie_new = review_data_ststistic(
            source_path='/data/lwang9/datasets/Douban_review/movie//moviereviews_cleaned.txt',columns=movie_columns)
    df_movie_new.dropna(subset=['labels'],inplace=True)
    df_movie_new['reviews'] = df_movie_new['reviews'].astype(str)
    df_movie_new['reviews'] = df_movie_new['reviews'].apply(replace_newline)
    df_movie_new = sample_movie_data(df_movie_new)
    df_movie_new = filter_g_k_one(df_movie_new,k_user=5,k_item=5)

    # df_movie_new.to_csv('../datasets/douban_review/movie_review_data.csv',index=False)

    print('hh')
    # df_movie_new = df_movie_new[df_movie_new['labels'] != ' ']
    # df_movie_new = df_movie_new[['user_id', 'item_id', 'rating','labels']]
    df_movie_new['labels'].replace({np.nan: '电影', ' ': '电影'}, inplace=True)
    logger.info(f'movie num:{len(df_movie_new)}')
    df_movie_new.to_csv('../datasets/douban_review/movie/movie_review_data.csv',index=False)

    # df_movie_new[['user_id','item_id','rating']].to_csv('../datasets/douban_review/movie_inter.csv', index=False)
    # df_movie_multi_modal = process_multi_modal_data(df_movie_new,'电影')
    # df_movie_multi_modal.to_csv('../datasets/douban_review/movie_multi_modal_data.csv',index=False)


    music_columns = {"user_id":"user_id",
                         "music_id":"item_id",
                         "rating":"rating",
                         "labels":"labels",
                         "comment":"reviews",
                         "useful_num":"userful_num",
                         "time":"time",
                         "ID":"ID"}
    music_r_sample_num, music_r_user_num, music_r_item_num, df_music_new = review_data_ststistic(
            source_path='/data/lwang9/datasets/Douban_review/music/musicreviews_cleaned.txt',columns=music_columns)
    df_music_new = filter_g_k_one(df_music_new,k_user=5,k_item=5)
    # df_music_new.dropna(subset=['labels'],inplace=True)
    # df_music_new = df_music_new[df_music_new['labels'] != ' ']
    df_music_new['labels'].replace({np.nan: '音乐', ' ': '音乐'}, inplace=True)
    logger.info(f'music num:{len(df_music_new)}')
    df_music_new.to_csv('../datasets/douban_review/music/music_review_data.csv',index=False)

    # df_music_new[['user_id', 'item_id', 'rating']].to_csv('../datasets/douban_review/music_inter.csv', index=False)
    # df_music_multi_modal = process_multi_modal_data(df_music_new,'音乐')
    # df_music_multi_modal.to_csv('../datasets/douban_review/music_multi_modal_data.csv',index=False)

    # # common_users = list(
    # #     set(df_book_new['user_id']).intersection((set(df_movie_new['user_id']))).intersection((set(df_music_new['user_id']))))
    # book_users = list(df_book_new['user_id'].unique())
    # movie_users = list(df_movie_new['user_id'].unique())
    # music_users = list(df_music_new['user_id'].unique())
    #
    # # common_users = list((set(book_users) & set(movie_users)) | (set(book_users) & set(music_users)) | (
    # #             set(movie_users) & set(music_users)))
    # common_users = list((set(music_users) & set(book_users)))
    # print(len(common_users))
    # # delete common users for all datasets
    # df_book_new = df_book_new[~df_book_new['user_id'].isin(common_users)]
    # df_movie_new = df_movie_new[~df_movie_new['user_id'].isin(common_users)]
    # df_music_new = df_music_new[~df_music_new['user_id'].isin(common_users)]
    # print(len(df_book_new))
    # print(len(df_movie_new))
    # print(len(df_music_new))



