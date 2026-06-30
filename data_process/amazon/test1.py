import pandas as pd
import gzip

def process_multi_modal_data(meta_data_path,items,to_path):
    data = gzip.open(meta_data_path,'rb')
    item_ids = []
    texts = []
    image_urls = []
    for line in data:
        d = eval(line)
        if d['asin'] in items:
            try:
                item_id = d['asin']
            except:
                item_id = ''
            try:
                im_url = d['imUrl']
            except:
                im_url = ''
            try:
                description = d['description']
            except:
                description = ''
            try:
                categories = ','.join(d['categories'][0])
            except:
                categories = ''
            try:
                title = d['title']
            except:
                title = ''
            item_ids.append(item_id)
            texts.append(title+" "+","+categories+" "+description)
            image_urls.append(im_url)
    df = pd.DataFrame({
        'item_id': item_ids,
        'texts': texts,
        'image_url': image_urls
    })
    df.to_csv(to_path,index=False)
if __name__ == '__main__':
    # df_inter = pd.read_csv('../datasets/amazon_review/beauty_inter_non_overlap_filter.csv')
    # item_ids = list(df_inter['item_id'].sort_values())
    # process_multi_modal_data(meta_data_path='/data/lwang9/datasets/amazon/meta_features/meta_Beauty.json.gz',items = item_ids,to_path = '../datasets/amazon_review/beauty_multi_modal_data_filter.csv')

    # df_inter = pd.read_csv('../datasets/amazon_review/health_inter_non_overlap_filter.csv')
    # item_ids = list(df_inter['item_id'].sort_values())
    # process_multi_modal_data(meta_data_path='/data/lwang9/datasets/amazon/meta_features/meta_Health_and_Personal_Care.json.gz',
    #                          items=item_ids, to_path='../datasets/amazon_review/health_multi_modal_data_filter.csv')
    #
    # df_inter = pd.read_csv('../datasets/amazon_review/cloth_inter_non_overlap_filter.csv')
    # item_ids = list(df_inter['item_id'].sort_values())
    # process_multi_modal_data(
    #     meta_data_path='/data/lwang9/datasets/amazon/meta_features/meta_Clothing_Shoes_and_Jewelry.json.gz',
    #     items=item_ids, to_path='../datasets/amazon_review/cloth_multi_modal_data_filter.csv')

    df_inter = pd.read_csv('../datasets/amazon_review/phone_sport_cloth/sport/sport_inter_non_overlap_filter.csv')
    item_ids = list(df_inter['item_id'].sort_values())
    process_multi_modal_data(
        meta_data_path='/data/lwang9/datasets/amazon/meta_features/meta_Sports_and_Outdoors.json.gz',
        items=item_ids, to_path='../datasets/amazon_review/phone_sport_cloth/sport/sport_multi_modal_data_filter.csv')

