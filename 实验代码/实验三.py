import pandas as pd

offline = pd.read_csv('ccf_offline_stage1_test_revised.csv')

offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)

offline['discount_rate'] = (offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
                            (float(str(x).split(':')[0])-float(str(x).split(':')[1]))/(float(str(x).split(':')[0]))))

offline['min_cost_of_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' not in str(x) else
                                 int(str(x).split(':')[0]))

offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')

offline['Distance'].fillna(-1, inplace=True)

print(offline.isnull().any())

print(offline.isnull().sum()/len(offline))


