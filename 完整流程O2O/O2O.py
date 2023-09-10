import pandas as pd
import numpy as np

# 读取数据集
off_train = pd.read_csv('ccf_offline_stage1_train.csv')
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')

# 数据预处理

#   查看数据集中的时间最值，确定范围，便于划分数据集
print(off_train['Date_received'].dropna().min())
print(off_train['Date_received'].dropna().max())
print(off_train['Date'].dropna().min())
print(off_train['Date'].dropna().max())
print(off_test['Date_received'].dropna().min())
print(off_test['Date_received'].dropna().max())

print('############')

#   查看id值的最小值，防止后续填充空值时覆盖掉数据
print(off_train['Coupon_id'].dropna().min())
print(off_train['Merchant_id'].dropna().min())

print('############')

#   填充空值，将时间空值用0代替，便于比较大小进行划分
off_train['Date_received'].fillna(0, inplace=True)
off_train['Date'].fillna(0, inplace=True)
off_test['Date_received'].fillna(0, inplace=True)

off_train['Distance'].fillna(-1, inplace=True)
off_train['Coupon_id'].fillna(0, inplace=True)

# print(off_train.info())
# print(off_train.head(5))

###########################################
# 数据划分

dataset1 = off_train[(off_train['Date_received'] >= 20160415) &
                     (off_train['Date_received'] <= 20160514)]
feature1 = off_train[((off_train['Date'] >= 20160101) & (off_train['Date'] <= 20160414)) |
                     ((off_train['Date'] == 0) & (off_train['Date_received'] >= 20160101) &
                     (off_train['Date_received'] <= 20160414))]

dataset2 = off_train[(off_train['Date_received'] >= 20160515) &
                     (off_train['Date_received'] <= 20160615)]
feature2 = off_train[((off_train['Date'] >= 20160201) & (off_train['Date'] <= 20160514)) |
                     ((off_train['Date'] == 0) & (off_train['Date_received'] >= 20160201) &
                     (off_train['Date_received'] <= 20160514))]

dataset3 = off_test
feature3 = off_train[((off_train['Date'] >= 20160301) & (off_train['Date'] <= 20160630)) |
                     ((off_train['Date'] == 0) & (off_train['Date_received'] >= 20160301) &
                     (off_train['Date_received'] <= 20160630))]

# print(feature1.info())
# print(feature1.head(5))

#######################################
# 特征工程


def discount_rate(a):
    a = str(a)
    a = a.split(':')
    if len(a) == 1:
        return float(a[0])
    else:
        return (float(a[0])-float(a[1]))/float(a[0])


def get_man(a):
    a = str(a)
    a = a.split(':')
    if len(a) == 1:
        return 'null'
    else:
        return int(a[0])


def get_jian(a):
    a = str(a)
    a = a.split(':')
    if len(a) == 1:
        return 'null'
    else:
        return int(a[1])


def is_man_jian(a):
    a = str(a)
    a = a.split(':')
    if len(a) == 1:
        return 0
    else:
        return 1


def get_user_feature(feature):
    user = feature.copy()
    t = user[['User_id']].copy()
    t.drop_duplicates(inplace=True)

    # 用户购买商品数量
    t1 = user[user['Date'] != 0][['User_id', 'Merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1['Merchant_id'] = 1
    t1 = t1.groupby('User_id').agg('sum').reset_index()
    t1.rename(columns={'Merchant_id': 'Count_merchant'}, inplace=True)

    # 用户优惠券购买距离最小值
    t2 = user[(user['Date'] != 0) & (user['Coupon_id'] != 0)][['User_id', 'Distance']]
    t2['Distance'] = t2['Distance'].astype('int')
    t2.replace(-1, np.nan, inplace=True)
    t3 = t2.groupby('User_id').agg('min').reset_index()
    t3.rename(columns={'Distance': 'user_min_distance'}, inplace=True)

    # 距离最大值
    t4 = t2.groupby('User_id').agg('max').reset_index()
    t4.rename(columns={'Distance': 'user_max_distance'}, inplace=True)

    # 距离平均值
    t5 = t2.groupby('User_id').agg('mean').reset_index()
    t5.rename(columns={'Distance': 'user_mean_distance'}, inplace=True)

    # 距离中间值
    t6 = t2.groupby('User_id').agg('median').reset_index()
    t6.rename(columns={'Distance': 'user_median_distance'}, inplace=True)

    # 用户使用优惠券次数
    t7 = user[(user['Date'] != 0) & (user['Coupon_id'] != 0)][['User_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('User_id').agg('sum').reset_index()

    # 用户领取券总数
    t8 = user[user['Coupon_id'] != 0][['User_id']]
    t8['coupon_received'] = 1
    t8 = t8.groupby('User_id').agg('sum').reset_index()

    # 领券到消费时间间隔
    t9 = user[(user['Date_received'] != 0) & user['Date'] != 0][['User_id', 'Date_received', 'Date']]
    t9['Date'] = pd.to_datetime(t9['Date'], format='%Y%m%d')
    t9['Date_received'] = pd.to_datetime(t9['Date_received'], format='%Y%m%d')
    t9['user_date_received_date_gap'] = t9['Date']-t9['Date_received']
    t9 = t9[['User_id', 'user_date_received_date_gap']]

    # 平均时间间隔
    t10 = t9.groupby('User_id').agg('mean').reset_index()
    t10.rename(columns={'user_date_received_date_gap': 'ave_user_date_received_date_gap'}, inplace=True)

    # 最小时间间隔
    t11 = t9.groupby('User_id').agg('min').reset_index()
    t11.rename(columns={'user_date_received_date_gap': 'min_user_date_received_date_gap'}, inplace=True)

    # 最大时间间隔
    t12 = t9.groupby('User_id').agg('max').reset_index()
    t12.rename(columns={'user_date_received_date_gap': 'max_user_date_received_date_gap'}, inplace=True)

    user_feature = pd.merge(t, t1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t5, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t6, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t7, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t8, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t9, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t10, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t11, on='User_id', how='left')
    user_feature = pd.merge(user_feature, t12, on='User_id', how='left')
    user_feature['buy_use_coupon'].replace(np.nan, 0, inplace=True)
    user_feature['buy_use_coupon_rate'] = user_feature['buy_use_coupon']/user_feature['Count_merchant']
    user_feature['Count_merchant'].replace(np.nan, 0, inplace=True)
    user_feature['user_coupon_transfer_rate'] = user_feature['buy_use_coupon']/user_feature['coupon_received']

    return user_feature


# get_user_feature(feature1)


def get_merchant_feature(feature):
    merchant = feature[['Merchant_id', 'Coupon_id', 'Distance', 'Date_received', 'Date']].copy()
    t = merchant[['Merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    # 每个商家卖出总数量
    t1 = merchant[merchant['Date'] != 0][['Merchant_id']].copy()
    t1['total_sale'] = 1
    t1 = t1.groupby('Merchant_id').agg('sum').reset_index()

    # 使用优惠券消费的商品总数
    t2 = merchant[(merchant['Date'] != 0) & (merchant['Coupon_id'] != 0)][['Merchant_id']].copy()
    t2['sale_use_coupon'] = 1
    t2 = t2.groupby('Merchant_id').agg('sum').reset_index()

    # 商品优惠券总数量
    t3 = merchant[merchant['Coupon_id'] != 0][['Merchant_id']].copy()
    t3['total_coupon'] = 1
    t3 = t3.groupby('Merchant_id').agg('sum').reset_index()

    # 优惠券使用和距离的关系
    t4 = merchant[(merchant['Date'] != 0) & (merchant['Coupon_id'] != 0)][['Merchant_id', 'Distance']].copy()
    t4.replace(-1, np.nan, inplace=True)

    # 最小距离
    t5 = t4.groupby('Merchant_id').agg('min').reset_index()
    t5.rename(columns={'Distance': 'merchant_min_distance'}, inplace=True)

    # 最大距离
    t6 = t4.groupby('Merchant_id').agg('max').reset_index()
    t6.rename(columns={'Distance': 'merchant_max_distance'}, inplace=True)

    # 平均距离
    t7 = t4.groupby('Merchant_id').agg('mean').reset_index()
    t7.rename(columns={'Distance': 'merchant_mean_distance'}, inplace=True)

    # 中间距离
    t8 = t4.groupby('Merchant_id').agg('median').reset_index()
    t8.rename(columns={'Distance': 'merchant_median_distance'}, inplace=True)

    merchant_feature = pd.merge(t, t1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t5, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t6, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t7, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, t8, on='Merchant_id', how='left')
    merchant_feature['sale_use_coupon'] = merchant_feature['sale_use_coupon'].replace(np.nan, 0)
    merchant_feature['merchant_coupon_transfer_rate'] = (
            merchant_feature['sale_use_coupon']/merchant_feature['total_coupon'])
    merchant_feature['coupon_rate'] = merchant_feature['sale_use_coupon']/merchant_feature['total_sale']
    merchant_feature['total_coupon'] = merchant_feature['total_coupon'].replace(np.nan, 0)

    return merchant_feature


# get_merchant_feature(feature1)


def get_coupon_feature(dataset, feature):
    dataset_c = dataset.copy()
    dataset_c['Date_received'] = pd.to_datetime(dataset_c['Date_received'], format='%Y%m%d')
    t = feature[feature['Date'] != 0][['Date']].max().copy()
    t = pd.to_datetime(t, format='%Y%m%d')
    # print(t)

    # 星期几
    dataset_c['day_of_week'] = dataset_c['Date_received'].apply(lambda x: x.isoweekday())
    # 几月
    dataset_c['day_of_month'] = dataset_c['Date_received'].apply(lambda x: x.month)
    # 领券时期到最大日期之间的天数
    dataset_c['days_distance'] = dataset_c['Date_received']-t
    # 显示满了多少钱后开始减
    dataset_c['discount_man'] = dataset_c['Discount_rate'].apply(get_man)
    # 显示满减的减少的钱
    dataset_c['discount_jian'] = dataset_c['Discount_rate'].apply(get_jian)
    # 返回优惠券是否是满减券
    dataset_c['is_man_jian'] = dataset_c['Discount_rate'].apply(is_man_jian)
    # 显示打折力度
    dataset_c['discount_rate'] = dataset_c['Discount_rate'].apply(discount_rate)
    d = dataset_c[['Coupon_id']].copy()
    d['coupon_count'] = 1
    # 显示每一种优惠券的数量
    d = d.groupby('Coupon_id').agg('sum').reset_index()
    dataset_c = pd.merge(dataset_c, d, on='Coupon_id', how='left')

    return dataset_c


# f1 = get_coupon_feature(dataset1, feature1)
# print(f1.head(5))


def get_user_with_merchant_feature(feature):
    user_merchant = feature[['User_id', 'Merchant_id']].copy()
    user_merchant.drop_duplicates(inplace=True)

    # 一个客户在一个商家一共买的次数
    t = feature[['User_id', 'Merchant_id', 'Date']].copy()
    t = t[t['Date'] != 0][['User_id', 'Merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()

    # 一个客户在一个商家一共收到的优惠券
    t1 = feature[['User_id', 'Merchant_id', 'Coupon_id']]
    t1 = t1[t1['Coupon_id'] != 0][['User_id', 'Merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()

    # 一个客户在一个商家使用优惠券购买的次数
    t2 = feature[['User_id', 'Merchant_id', 'Date', 'Date_received']]
    t2 = t2[(t2['Date'] != 0) & (t2['Date_received'] != 0)][['User_id', 'Merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()

    # 一个客户在一个商家浏览的次数（领过优惠券或者买过商品）
    t3 = feature[['User_id', 'Merchant_id']].copy()
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()

    # 一个客户在一个商家没有使用优惠券购买的次数
    t4 = feature[['User_id', 'Merchant_id', 'Date', 'Coupon_id']]
    t4 = t4[(t4['Date'] != 0) & (t4['Coupon_id'] == 0)
            ][['User_id', 'Merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['User_id', 'Merchant_id']).agg('sum').reset_index()

    user_merchant = pd.merge(user_merchant, t, on=['User_id', 'Merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t1, on=['User_id', 'Merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t2, on=['User_id', 'Merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t3, on=['User_id', 'Merchant_id'], how='left')
    user_merchant = pd.merge(user_merchant, t4, on=['User_id', 'Merchant_id'], how='left')
    user_merchant['user_merchant_buy_use_coupon'] = user_merchant['user_merchant_buy_use_coupon'].replace(np.nan, 0)
    user_merchant['user_merchant_buy_common'] = user_merchant['user_merchant_buy_common'].replace(np.nan, 0)
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant['user_merchant_buy_use_coupon'].astype(
        'float') / user_merchant['user_merchant_received'].astype('float')
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant['user_merchant_buy_use_coupon'].astype(
        'float') / user_merchant['user_merchant_buy_total'].astype('float')
    user_merchant['user_merchant_rate'] = user_merchant['user_merchant_buy_total'].astype(
        'float') / user_merchant['user_merchant_any'].astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant['user_merchant_buy_common'].astype(
        'float') / user_merchant['user_merchant_buy_total'].astype('float')

    return user_merchant


# f1 = get_user_with_merchant_feature(feature1)
# print(f1.head(5))


'''def get_other_feature(dataset):
    other_feature = dataset[['User_id', 'Coupon_id']].copy()

    # 每个用户收到优惠券的数量
    t = dataset[['User_id']].copy()
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('User_id').agg('sum').reset_index()

    # 用户领取指定优惠券的数量
    t1 = dataset[['User_id', 'Coupon_id']].copy()
    t1['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()

    # 领券最大、最小时间
    t2 = dataset[['User_id', 'Coupon_id', 'Date_received']].copy()
    t2['Date_received'] = pd.to_datetime(t2['Date_received'], format='%Y%m%d')
    t2 = t2.groupby(['User_id', 'Coupon_id'])['Date_received'].max().reset_index()
    t2.rename(columns={'Date_received': 'max_date'}, inplace=True)
    print(t2.info())

    t3 = t2 = dataset[['User_id', 'Coupon_id', 'Date_received']].copy()
    t3['Date_received'] = pd.to_datetime(t2['Date_received'], format='%Y%m%d')
    t3 = t2.groupby(['User_id', 'Coupon_id'])['Date_received'].min().reset_index()
    t3.rename(columns={'Date_received': 'min_date'}, inplace=True)

    # 一天领券数量
    t4 = dataset[['User_id', 'Date_received']].copy()
    t4['this_day_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['User_id', 'Date_received']).agg('sum').reset_index()
    t4 = t4.drop(['Date_received'], axis=1)

    # 当天所接收到相同优惠券的数量
    t5 = dataset[['User_id', 'Coupon_id', 'Date_received']].copy()
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['User_id', 'Coupon_id', 'Date_received']).agg('sum').reset_index()
    t5 = t5.drop(['Date_received'],axis=1)

    other_feature = pd.merge(other_feature, t, on=['User_id'])
    other_feature = pd.merge(other_feature, t1, on=['User_id', 'Coupon_id'])
    other_feature = pd.merge(other_feature, t2, on=['User_id', 'Coupon_id'])
    other_feature = pd.merge(other_feature, t3, on=['User_id', 'Coupon_id'])
    other_feature = pd.merge(other_feature, t4, on=['User_id'])
    other_feature = pd.merge(other_feature, t5, on=['User_id', 'Coupon_id'])
    print(other_feature.head(5))
    print(other_feature.info())

    #return other_feature


f1 = get_other_feature(dataset1)
#print(f1.head(5))'''


def data_process(dataset0, feature, train_flag):
    merchant = get_merchant_feature(feature)
    user = get_user_feature(feature)
    user_merchant = get_user_with_merchant_feature(feature)
    coupon = get_coupon_feature(dataset0, feature)

    dataset = pd.merge(coupon, merchant, on='Merchant_id', how='left')
    dataset = pd.merge(dataset, user, on='User_id', how='left')
    dataset = pd.merge(dataset, user_merchant, on=['User_id', 'Merchant_id'], how='left')
    dataset.drop_duplicates(inplace=True)

    if train_flag:
        dataset['Date'].replace(0, 'nan', inplace=True)
        dataset['Date_received'] = pd.to_datetime(dataset['Date_received'], format='%Y%m%d')
        dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y%m%d')
        dataset['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
                                    dataset['Date'], dataset['Date_received']))

    return dataset


ProcessDataSet1 = data_process(dataset1, feature1, True)
ProcessDataSet1.to_csv('ProcessDataSet1.csv')
print('---------------ProcessDataSet1 done-------------------')
ProcessDataSet2 = data_process(dataset2, feature2, True)
ProcessDataSet2.to_csv('ProcessDataSet2.csv')
print('---------------ProcessDataSet2 done-------------------')
# 3是测试集，所以不标记
ProcessDataSet3 = data_process(dataset3,feature3,False)
ProcessDataSet3.to_csv('ProcessDataSet3.csv')
print('---------------ProcessDataSet3 done-------------------')
