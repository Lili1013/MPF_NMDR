import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np

def process_text_data(source_path,to_path):
    df = pd.read_csv(source_path)
    df.sort_values(by=['item_id'], inplace=True)
    sentences = []
    lack_index = []
    for index,each_line in df.iterrows():
        sen = each_line['texts']
        sen = sen.replace('\n', ' ')
        sentences.append(sen)

    logger.info('start transform')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    # assert sentence_embeddings.shape[0] == df_text.shape[0]
    # fill_list = [0] * 384
    # for each_index in lack_index:
    #     sentence_embeddings = np.insert(sentence_embeddings,each_index,fill_list,axis=0)
    np.save(to_path, sentence_embeddings)
    logger.info('done!')

if __name__ == '__main__':
    # process_text_data(source_path='../datasets/amazon_review/cloth_multi_modal_data_filter.csv',
    #                   to_path='../datasets/amazon_review/cloth_text_features_filter.npy')
    # process_text_data(source_path='../datasets/amazon_review/phone_sport_cloth/phone/phone_multi_modal_data_filter.csv',
    #                   to_path='../datasets/amazon_review/phone_sport_cloth/phone/phone_text_features_filter.npy')
    # process_text_data(source_path='../datasets/amazon_review/phone_sport_cloth/sport/sport_multi_modal_data_filter.csv',
    #                   to_path='../datasets/amazon_review/phone_sport_cloth/sport/sport_text_features_filter.npy')
    process_text_data(source_path='../datasets/amazon_review/phone_sport_cloth/cloth/cloth_multi_modal_data_filter.csv',
                      to_path='../datasets/amazon_review/phone_sport_cloth/cloth/cloth_text_features_filter.npy')