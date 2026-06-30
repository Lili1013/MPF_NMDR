import array

import pandas as pd
import numpy as np


def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10).decode('UTF-8')
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()
def process_visual_feat(source_image_path,to_path,item_ids):
  # df = pd.read_csv(source_path)
  # df.sort_values(by=['item_id'], inplace=True)
  # item2id = df['item_id'].unique().tolist()
  img_data = readImageFeatures(source_image_path)
  # item2id = dict(zip(df['asin'], df['itemID']))
  feats = {}
  avg = []
  for d in img_data:
    # print('gg')
    if d[0] in item_ids:
      feats[d[0]] = d[1]
      avg.append(d[1])
  avg = np.array(avg).mean(0).tolist()

  ret = []
  non_no = []
  for i in item_ids:
    if i in feats:
      ret.append(feats[i])
    else:
      non_no.append(i)
      ret.append(avg)

  print('# of items not in processed image features:', len(non_no))
  # assert len(ret) == len(item2id)
  np.save(to_path, np.array(ret))

if __name__ == '__main__':
  df_beauty = pd.read_csv('../datasets/amazon_review/health_cloth_beauty/cloth/cloth_inter_non_overlap_filter.csv')
  df_beauty.sort_values(by=['item_id'], inplace=True)
  item_ids = df_beauty['item_id'].unique().tolist()
  process_visual_feat(source_image_path='/data/lwang9/datasets/amazon/image_features/image_features_Clothing_Shoes_and_Jewelry.b',
                      to_path='../datasets/amazon_review/health_cloth_beauty/cloth/cloth_image_features_filter.npy', item_ids=item_ids)
  # process_visual_feat(source_path='datasets/elec/elec_inter_1.csv',
  #                     source_image_path='datasets/image_features_Electronics.b',
  #                     to_path='datasets/elec/elec_visual_feat_1.npy')
  # with open('../datasets/amazon_phone/phone_visual_feat.npy', 'rb') as f:
  #   visual_feat = np.load(f)
  # print('gg')

