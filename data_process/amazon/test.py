import pandas as pd

df_health = pd.read_csv('../datasets/amazon_review/health_cloth_beauty/health/health_inter_non_overlap_filter.csv')
df_health['domain_id'] = '0'
df_cloth = pd.read_csv('../datasets/amazon_review/health_cloth_beauty/cloth/cloth_inter_non_overlap_filter.csv')
df_cloth['domain_id'] = '1'
df_beauty = pd.read_csv('../datasets/amazon_review/health_cloth_beauty/beauty/beauty_inter_non_overlap_filter.csv')
df_beauty['domain_id'] = '2'
df = pd.concat([df_beauty,df_cloth,df_health])
health_users = list(df[df['domain_id']=='0']['user_id'].unique())
cloth_users = list(df[df['domain_id']=='1']['user_id'].unique())
beauty_users = list(df[df['domain_id']=='2']['user_id'].unique())
common_users = set(health_users).intersection(set(cloth_users)).intersection(set(beauty_users))
print(len(common_users))
# df_overlap = df.groupby('user_id').filter(lambda x:len(x)>1)
# print(len(df_overlap))