import pandas as pd

offline = pd.read_csv('ccf_offline_stage1_train.csv')
data = offline.copy()


# 数据预处理
data['Distance'].fillna(-1, inplace=True)
data['Coupon_id'].fillna('nan', inplace=True)
data['Date_received'].fillna('nan', inplace=True)
data['Date'].fillna('nan', inplace=True)
# 将优惠券领取时间转换为时间类型
data['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
data['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
# 将满减型的转换为折扣率
data['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0])-float(str(x).split(':')[1]))/(float(str(x).split(':')[0])))
data['discount_rate'].fillna(0, inplace=True)
# 判断是否为满减
data['ismanjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 星期几领券
data['weekday_Receive'] = data['date_received'].apply(lambda x: x.isoweekday())
# 几月份领券
data['receive_month'] = data['date_received'].apply(lambda x: x.month)
data['date_month'] = data['date'].apply(lambda x: x.month)
# 打标
data['label'] = list(map(lambda x, y: 1 if (x-y).total_seconds()/(60*60*24) <= 15 else 0,
                         data['date'], data['date_received']))


# 基本特征
# data['Coupon_id'] = data['Coupon_id'].map(int)
# data['Date_received'] = data['Date_received'].map(int)
data['cnt'] = 1
feature = data.copy()

keys = ['User_id']
prefixs = '_'.join(keys)+'_'
pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs+'receive_cnt'}).reset_index()
feature = pd.merge(feature, pivot, on=keys, how='left')
feature.fillna(0, downcast='infer', inplace=True)

pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs+'receive_not_consume_cnt'}).reset_index()
feature = pd.merge(feature, pivot, on=keys, how='left')
feature.fillna(0, downcast='infer', inplace=True)

pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs+'receive_and_consume_cnt'}).reset_index()
feature = pd.merge(feature, pivot, on=keys, how='left')
feature.fillna(0, downcast='infer', inplace=True)

feature[prefixs + 'receive_and_consume_rate'] = list(map(
    lambda x, y: x/y if y != 0 else 0,
    feature[prefixs + 'receive_and_consume_cnt'],
    feature[prefixs+'receive_cnt']
))

pivot1 = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='discount_rate', aggfunc=sum)
pivot2 = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
feature[prefixs+'ave_discount_rate'] = list(map(
    lambda x, y: x/y if y!= 0 else 0,
    pivot1['discount_rate'], pivot2['cnt']
))

pivot1 = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='Distance', aggfunc=sum)
pivot2 = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
feature[prefixs+'ave_distance'] = list(map(
    lambda x, y: x/y if y!= 0 else 0,
    pivot1['Distance'], pivot2['cnt']
))

pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
pivot = pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs+'receive_differ_Merchant_cnt'}).reset_index()
feature = pd.merge(feature, pivot, on=keys, how='left')
feature.fillna(0, downcast='infer', inplace=True)

pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
pivot = (pd.DataFrame(pivot).rename(columns={'Merchant_id': prefixs+'receive_and_consume_differ_Merchant_cnt'})
         .reset_index())
feature = pd.merge(feature, pivot, on=keys, how='left')
feature.fillna(0, downcast='infer', inplace=True)

feature[prefixs + 'receive_and_consume_differ_Merchant_rate'] = list(map(
    lambda x, y: x/y if y != 0 else 0,
    feature[prefixs+'receive_and_consume_differ_Merchant_cnt'],
    feature[prefixs+'receive_differ_Merchant_cnt']
))

feature.to_csv('特征.csv')
