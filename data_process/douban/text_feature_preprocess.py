from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from loguru import logger

# 确保GPU可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的SBERT模型
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

def encode_reviews(reviews, batch_size=32):
    embeddings = []  # 用于存储所有嵌入向量
    for i in range(0, len(reviews), batch_size):
        logger.info(i)
        batch = reviews[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=True)
        embeddings.append(batch_embeddings)
        logger.info(i)

    # 将所有批次的嵌入向量合并为一个tensor
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.cpu().numpy()  # 将嵌入向量转移到CPU并转为NumPy数组



if __name__ == '__main__':
    # dataset = 'music'
    # df = pd.read_csv(f'../datasets/douban_review/{dataset}/{dataset}_multi_modal_data.csv')
    # # 调用函数处理评论数据
    # review_embeddings = encode_reviews(df['labels'].tolist(), batch_size=64)
    #
    # # 将嵌入向量保存为numpy文件，也可以选择其他方式保存，如直接附加到DataFrame
    # np.save(f'../datasets/douban_review/{dataset}/{dataset}_text_features.npy', review_embeddings)

    dataset = 'music'
    df = np.load(f'../datasets/douban_review/{dataset}/{dataset}_text_features.npy')
    print(len(df))
