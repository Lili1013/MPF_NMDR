import pandas as pd
from loguru import logger

def process_multi_modal_data(df):
    df['labels'] = df['labels'].str.replace('|',',')
    df_new = df.groupby('itemID')['labels'].apply(lambda x:','.join(x)).reset_index()
    # df_new['labels'] = df_new['labels'].apply(lambda x: x+','+cat)
    return df_new
    # for index,row in df.groupby(by='item_id'):
    #     print('hh')

def id_map(df,user_to_path,item_to_path,to_path,u_start_index,i_start_index):
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df_items = df.sort_values(by=['item_id'])
    uni_items = df_items['item_id'].unique().tolist()
    i_id_map = {k: i+i_start_index for i, k in enumerate(uni_items)}
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['item_id', 'itemID'])
    df['itemID'] = df['item_id'].map(i_id_map)
    df['itemID'] = df['itemID'].astype(int)

    df_users = df.sort_values(by=['user_id'])
    uni_users = df_users['user_id'].unique().tolist()
    u_id_map = {k: i+u_start_index for i, k in enumerate(uni_users)}
    u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
    df['userID'] = df['user_id'].map(u_id_map)
    df['userID'] = df['userID'].astype(int)

    u_df.to_csv(user_to_path,index=False)
    i_df.to_csv(item_to_path,index=False)
    print(len(df))
    df.to_csv(to_path,index=False)
    return df

if __name__ == '__main__':
    # #step 1
    # path = f'../datasets/douban_review'
    # dataset = 'book'
    # df = pd.read_csv(f'{path}/{dataset}/{dataset}_review_data.csv')
    # logger.info(f'total sample:{len(df)}, user num:{len(df["user_id"].unique())}, item num:{len(df["item_id"].unique())}')
    # df_new = id_map(df=df,user_to_path=f'{path}/{dataset}/user_id_map.csv',
    #                 item_to_path=f'{path}/{dataset}/item_id_map.csv',to_path=f'{path}/{dataset}/{dataset}_review_data_map.csv',
    #                 u_start_index=0,i_start_index=0)
    #step 2
    dataset ='music'
    df = pd.read_csv(f'../datasets/douban_review/{dataset}/{dataset}_review_data_map.csv')
    df_multi_modal = process_multi_modal_data(df)
    df_multi_modal.to_csv(f'../datasets/douban_review/{dataset}/{dataset}_multi_modal_data.csv',index=False)

