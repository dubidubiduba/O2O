import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost.callback import EarlyStopping, EvaluationMonitor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

dataset1 = pd.read_csv('ProcessDataSet1.csv')
dataset2 = pd.read_csv('ProcessDataSet2.csv')
dataset3 = pd.read_csv('ProcessDataSet3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset12 = pd.concat([dataset1, dataset2], axis=0)
dataset12_y = dataset12['label']

dataset12_x = dataset12.drop(['User_id', 'label',  'Coupon_id', 'Discount_rate', 'user_date_received_date_gap',
                              'max_user_date_received_date_gap', 'min_user_date_received_date_gap', 'Date',
                              'ave_user_date_received_date_gap'], axis=1)
dataset12_x['Date_received'] = pd.to_datetime(dataset12_x['Date_received']).dt.strftime('%Y%m%d').astype(np.float64)
# dataset12_x['Date'] = pd.to_datetime(dataset12_x['Date']).dt.strftime('%Y%m%d').astype(np.float64)
# print(dataset12_x.info())
# print(dataset12_x['ave_user_date_received_date_gap'].head(5))

dataset3.drop_duplicates(inplace=True)
dataset3_preds = dataset3[['User_id', 'Coupon_id', 'Date_received']]
dataset3_preds['Date_received'] = pd.to_datetime(dataset3_preds['Date_received']).dt.strftime('%Y%m%d').astype(np.int64)
dataset3_x = dataset3.drop(['User_id', 'Coupon_id', 'Discount_rate', 'user_date_received_date_gap',
                              'max_user_date_received_date_gap', 'min_user_date_received_date_gap',
                              'ave_user_date_received_date_gap'], axis=1)
dataset3_x['Date_received'] = pd.to_datetime(dataset3_x['Date_received']).dt.strftime('%Y%m%d').astype(np.float64)
# print(dataset3_x.info())

dataTrain = xgb.DMatrix(dataset12_x, label=dataset12_y)
dataTest = xgb.DMatrix(dataset3_x)


'''params = {'booster': 'gbtree',
           'objective': 'binary:logistic',
           'eval_metric': 'auc',
           'silent': 1,
           'gamma': 0,
           'min_child_weight': 1,
           'max_depth': 5,
           'lambda': 1,
           'subsample': 0.9,
           'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
           'eta': 0.01,
           'tree_method': 'exact',
           'base_score':0.11,
           'scale_pos_weight': 1
           }


watchlist = [(dataTrain,'train')]
model1 = xgb.train(params, dataTrain, num_boost_round=1200, evals=watchlist)
model1.save_model('xgb_model6')
print('------------------------train done------------------------------')'''

'''params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
            #   'tree_method': 'gpu_hist',
            #   'n_gpus': '-1',
              'seed': 0,
            #   'predictor': 'gpu_predictor'
              }

# 使用xgb.cv优化num_boost_round参数
cvresult = xgb.cv(params, dataTrain, num_boost_round=10000, nfold=2, metrics='auc', seed=0, callbacks=[
    EvaluationMonitor(),
    EarlyStopping(rounds=50)
])
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

# 使用优化后的num_boost_round参数训练模型
watchlist = [(dataTrain, 'train')]
model = xgb.train(params, dataTrain, num_boost_round=num_round_best, evals=watchlist)

model.save_model('xgb_model7')'''

'''gbdtmodel = GradientBoostingClassifier(n_estimators=1900,
                                       learning_rate=0.01,
                                       min_samples_split=200,
                                       min_samples_leaf=50,
                                       random_state=0,
                                       subsample=0.8,
                                       max_depth=9,
                                       verbose=1,
                                       n_iter_no_change=50)
imp = SimpleImputer(missing_values= np.nan, strategy='mean', verbose=0, copy=True)
imp_dataset12_x = imp.fit_transform(dataset12_x)
gbdtmodel.fit(imp_dataset12_x, dataset12_y)
s = pickle.dumps(gbdtmodel)
f = open('gbdt_model1', 'wb')
f.write(s)
f.close()'''

pred = pd.DataFrame()

model = xgb.Booster()
model.load_model('xgb_model2')
pred['pred1'] = model.predict(dataTest)
pred['pred1'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(pred['pred1'].values.reshape(-1, 1))

model = xgb.Booster()
model.load_model('xgb_model3')
pred['pred2'] = model.predict(dataTest)
pred['pred2'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(pred['pred2'].values.reshape(-1, 1))

model = xgb.Booster()
model.load_model('xgb_model4')
pred['pred3'] = model.predict(dataTest)
pred['pred3'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(pred['pred3'].values.reshape(-1, 1))

model = xgb.Booster()
model.load_model('xgb_model5')
pred['pred4'] = model.predict(dataTest)
pred['pred4'] = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(pred['pred4'].values.reshape(-1, 1))

f1 = open('gbdt_model1', 'rb')
s1 = f1.read()
gbdtmodel = pickle.loads(s1)
f1.close()
imp = SimpleImputer(missing_values=np.nan, strategy='mean',
              verbose=0, copy=True)
imp_datatest = imp.fit_transform(dataset3_x)
pred['pred5'] = gbdtmodel.predict_proba(imp_datatest)[:, 1]

dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = (pred['pred4'].values.copy()*0.45 + pred['pred3'].values.copy() * 0.1 +
                            pred['pred2'].values.copy()*0.25 + pred['pred1'].values.copy() * 0.1 +
                            pred['pred5'].values.copy()*0.1)

dataset3_preds1.sort_values(by=['User_id', 'label'], inplace=True)
dataset3_preds1.to_csv("xgb_preds.csv", index=None, header=None)
print(dataset3_preds1.describe())
