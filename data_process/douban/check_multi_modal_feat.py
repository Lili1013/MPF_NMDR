import pandas as pd

def filter_g_k_one(data,k_user=10,k_item=10,u_name='user_id',i_name='item_id',y_name='rating'):
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
df_book_new.dropna(subset=['labels'],inplace=True)
df_book_new = df_book_new[df_book_new['labels']!= ' ']
null_num = df_book_new['labels'].isnull().sum()
print(null_num)
print(len(df_book_new))
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
df_movie_new = df_movie_new[df_movie_new['labels']!= ' ']
null_num = df_movie_new['labels'].isnull().sum()
print(null_num)
print(len(df_movie_new))
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
df_music_new.dropna(subset=['labels'],inplace=True)
df_music_new = df_music_new[df_music_new['labels']!= ' ']
null_num = df_music_new['labels'].isnull().sum()
print(null_num)
print(len(df_music_new))

# df_movie_new = filter_g_k_one(data=df_movie_new,k_user=5,k_item=5,u_name='user_id',i_name='item_id',y_name='rating')
# print(len(df_movie_new))
# df_book_new = filter_g_k_one(data=df_book_new,k_user=5,k_item=5,u_name='user_id',i_name='item_id',y_name='rating')
# print(len(df_book_new))
# df_music_new = filter_g_k_one(data=df_music_new,k_user=5,k_item=5,u_name='user_id',i_name='item_id',y_name='rating')
# print(len(df_music_new))