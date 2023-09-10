import pandas as pd
from pyecharts.charts import Bar
from pyecharts.charts import Pie
from pyecharts import options as opts
import collections
# 读取数据集
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')

# 统计记录数
record_count = off_test.shape[0]
# 统计优惠券领取记录数
received_count = off_test['Date_received'].count()
# 统计优惠券种类数
coupon_count = len(off_test['Coupon_id'].value_counts())
# 统计用户数
user_count = len(off_test['User_id'].value_counts())
# 统计商家数
merchant_count = len(off_test['Merchant_id'].value_counts())
# 最早领券时间
min_received = str(int(off_test['Date_received'].min()))
# 最晚领券时间
max_received = str(int(off_test['Date_received'].max()))
# 创建DataFrame类型用于输出观察
df = pd.DataFrame([record_count, received_count, coupon_count, user_count, merchant_count, min_received, max_received],
                  index=['记录数', '领取记录数', '优惠券种类数', '用户数', '商家数', '最早领券时间', '最晚领券时间'])
# 转换为csv文件
df.to_csv('统计结果.csv')

# 复制数据集
offline = off_test.copy()
# 将距离空值置为-1
offline['Distance'].fillna(-1, inplace=True)
# 将优惠券领取时间转换为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
# 将满减型的转换为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0])-float(str(x).split(':')[1]))/(float(str(x).split(':')[0])))
# 判断是否为满减
offline['isManjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 星期几领券
offline['weekday_Receive'] = offline['date_received'].apply(lambda x: x.isoweekday())
# 几月份领券
offline['receive_month'] = offline['date_received'].apply(lambda x: x.month)

offline.to_csv('统计结果2.csv')

# 领券数量柱状图
df_1 = offline[offline['Date_received'].notna()]
tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()

bar_1 = (
    Bar(init_opts=opts.InitOpts(width="1500px", height="600px"))
    .add_xaxis(list(tmp['Date_received']))
    .add_yaxis('', list(tmp['Coupon_id']))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="每天被领券的数量"),
        legend_opts=opts.LegendOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1)
    )
    .set_series_opts(
        opts.LabelOpts(is_show=False),
        markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="max", name="最大值")])
    )
)

bar_1.render('pyecharts_save/bar_1.html')

# 消费距离柱状图
dis = offline[offline['Distance'] != -1]['Distance'].values
dis = dict(collections.Counter(dis))

x1 = list(dis.keys())
y1 = list(dis.values())

bar_2 = (
    Bar()
    .add_xaxis(x1)
    .add_yaxis('', y1)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户消费距离记录")
    )
)

bar_2.render('pyecharts_save/bar_2.html')

# 各类优惠券占比饼图
v1 = ['满减', '折扣']
v2 = list(offline[offline['Distance'].notna()]['isManjian'].value_counts(True))
print(v2)
pie_1 = (
    Pie()
    .add("", [list(v) for v in zip(v1, v2)])
    .set_global_opts(title_opts={"text":"各类优惠券占比饼图"})
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
)

pie_1.render('pyecharts_save/pie_1.html')
