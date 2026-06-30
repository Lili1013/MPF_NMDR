import pandas as pd
from loguru import logger
import gzip
import array

def preprocess(json_file_path,to_path):
    ''''
    extract all users and items from the original files
    '''
    # json_gz_file_path = 'C:\work\Amazon_datasets\\Electronics_5.json'

    fin = gzip.open(json_file_path, 'r')
    review_list = []
    i = 0# 存储筛选出来的字段，如果数据量过大可以尝试用dict而不是list
    for line in fin:
        # 顺序读取json文件的每一行
        try:
            if i % 10000 == 0:
                logger.info(i)
            d = eval(line, {"true":True,"false":False,"null":None})
            review_list.append([d['reviewerID'],d['asin'],d['overall']])
        except:
            continue
        i += 1
    df = pd.DataFrame(review_list, columns =['user_id', 'item_id','rating']) # 转换为dataframe
    df.to_csv(to_path,index=False)

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()
# def process_visual_feat(source_image_path,to_path):
#   # df = pd.read_csv(source_path)
#   # df.sort_values(by=['item_id'], inplace=True)
#   # item2id = df['item_id'].unique().tolist()
#   img_data = readImageFeatures(source_image_path)
#   # item2id = dict(zip(df['asin'], df['itemID']))
#   feats = {}
#   avg = []
#   for d in img_data:
#     print('gg')
#     if d[0] in item2id:
#       feats[d[0]] = d[1]
#       avg.append(d[1])
#   avg = np.array(avg).mean(0).tolist()
#
#   ret = []
#   non_no = []
#   for i in item2id:
#     if i in feats:
#       ret.append(feats[i])
#     else:
#       non_no.append(i)
#       ret.append(avg)
#
#   print('# of items not in processed image features:', len(non_no))
#   # assert len(ret) == len(item2id)
#   np.save(to_path, np.array(ret))

if __name__ == '__main__':
    # transform review data into inter data
    # preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Clothing_Shoes_and_Jewelry_5.json.gz',to_path='../datasets/amazon_review/cloth_inter.csv')
    # preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Cell_Phones_and_Accessories_5.json.gz',
    #            to_path='../datasets/amazon_review/phone/phone_inter.csv')
    # preprocess(json_file_path='/data/lwang9/datasets/amazon/review_texts/reviews_Sports_and_Outdoors_5.json.gz',
    #            to_path='../datasets/amazon_review/sport/sport_inter.csv')
    #validate the lack number of visual data and meta data
    # df = pd.read_csv('../datasets/amazon_review/cloth_inter.csv')
    # df = pd.read_csv('../datasets/amazon_review/beauty_inter.csv')
    # items = df['item_id'].unique()#cloth: 23033   beauty: 12101
    # # img_data = readImageFeatures('/data/lwang9/datasets/amazon/image_features/image_features_Clothing_Shoes_and_Jewelry.b')
    # img_data = readImageFeatures('/data/lwang9/datasets/amazon/image_features/image_features_Beauty.b')
    # logger.info(len(items))
    # num = 0 #cloth: visual: 22879 meta data: 23033   beauty: visual: 12009 meta data: 12101
    # # for d in img_data:
    # #     if d[0] in items:
    # #         num += 1
    # # logger.info(num)
    #
    # # g = gzip.open('/data/lwang9/datasets/amazon/meta_features/meta_Clothing_Shoes_and_Jewelry.json.gz', 'rb')
    # g = gzip.open('/data/lwang9/datasets/amazon/meta_features/meta_Beauty.json.gz', 'rb')
    # for line in g:
    #     d = eval(line)
    #     if d['asin'] in items:
    #         num+=1
    # logger.info(num)

    # df_cloth = pd.read_csv('../datasets/amazon_review/cloth_inter.csv')
    # df_beauty = pd.read_csv('../datasets/amazon_review/beauty_inter.csv')
    # df_health = pd.read_csv('../datasets/amazon_review/health_inter.csv')
    # health_users = list(df_health['user_id'].unique())
    # cloth_users = list(df_cloth['user_id'].unique())
    # beauty_users = list(df_beauty['user_id'].unique())

    # common_users = list((set(health_users) & set(cloth_users)) | (set(health_users) & set(beauty_users)) | (
    #             set(cloth_users) & set(beauty_users)))
    # print(len(common_users))
    # df_beauty_1 = df_beauty[~df_beauty['user_id'].isin(common_users)]
    # df_cloth_1 = df_cloth[~df_cloth['user_id'].isin(common_users)]
    # df_health_1 = df_health[~df_health['user_id'].isin(common_users)]
    # logger.info(f'beauty all data :{len(df_beauty_1)}')
    # logger.info(f"beauty user : {len(df_beauty_1['user_id'].unique())}")  # user:
    # logger.info(f"beauty item: {len(df_beauty_1['item_id'].unique())}")  # item:
    # logger.info(f'cloth all data :{len(df_cloth_1)}')  #
    # logger.info(f"cloth user: {len(df_cloth_1['user_id'].unique())}")  # user:
    # logger.info(f"cloth item :{len(df_cloth_1['item_id'].unique())}")  # item:
    # logger.info(f'health all data: {len(df_health_1)}')  #
    # logger.info(f"health user: {len(df_health_1['user_id'].unique())}")  # user:
    # logger.info(f"health item: {len(df_health_1['item_id'].unique())}")  # item:
    # df_cloth_1.to_csv('../datasets/amazon_review/cloth_inter_non_overlap.csv',index = False)
    # df_beauty_1.to_csv('../datasets/amazon_review/beauty_inter_non_overlap.csv',index = False)
    # df_health_1.to_csv('../datasets/amazon_review/health_inter_non_overlap.csv',index = False)


    df_cloth = pd.read_csv('../datasets/amazon_review/health_cloth_beauty/cloth/cloth_inter.csv')
    df_phone = pd.read_csv('../datasets/amazon_review/phone_sport_cloth/phone/phone_inter.csv')
    df_sport = pd.read_csv('../datasets/amazon_review/phone_sport_cloth/sport/sport_inter.csv')
    phone_users = list(df_phone['user_id'].unique())
    cloth_users = list(df_cloth['user_id'].unique())
    sport_users = list(df_sport['user_id'].unique())
    # common_users = list(
    #     set(df_beauty['user_id']).intersection((set(df_cloth['user_id']))).intersection((set(df_health['user_id']))))
    common_users = list((set(phone_users) & set(cloth_users)) | (set(phone_users) & set(sport_users)) | (
                set(cloth_users) & set(sport_users)))
    print(len(common_users))
    df_phone_1 = df_phone[~df_phone['user_id'].isin(common_users)]
    df_cloth_1 = df_cloth[~df_cloth['user_id'].isin(common_users)]
    df_sport_1 = df_sport[~df_sport['user_id'].isin(common_users)]
    logger.info(f'phone all data :{len(df_phone_1)}')
    logger.info(f"phone user : {len(df_phone_1['user_id'].unique())}")  # user:
    logger.info(f"phone item: {len(df_phone_1['item_id'].unique())}")  # item:
    logger.info(f'cloth all data :{len(df_cloth_1)}')  #
    logger.info(f"cloth user: {len(df_cloth_1['user_id'].unique())}")  # user:
    logger.info(f"cloth item :{len(df_cloth_1['item_id'].unique())}")  # item:
    logger.info(f'sport all data: {len(df_sport_1)}')  #
    logger.info(f"sport user: {len(df_sport_1['user_id'].unique())}")  # user:
    logger.info(f"sport item: {len(df_sport_1['item_id'].unique())}")  # item:
    df_cloth_1.to_csv('../datasets/amazon_review/phone_sport_cloth/cloth/cloth_inter_non_overlap.csv',index = False)
    df_phone_1.to_csv('../datasets/amazon_review/phone_sport_cloth/phone/phone_inter_non_overlap.csv',index = False)
    df_sport_1.to_csv('../datasets/amazon_review/phone_sport_cloth/sport/sport_inter_non_overlap.csv',index = False)
