import numpy as np                 # 引入numpy库，pandas库，以及matplotlib库的pyplot子模块
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.DataFrame(np.random.rand(10, 5), index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                   columns=['a', 'b', 'c', 'd', 'e'])  # 使用numpy库的random模块中的rand方法生成均匀分布的十行五列数组
# 分别给行列命名，然后使用pandas库的DataFrame方法将其转换为DataFrame类型
df1.plot.bar()  # 使用DataFrame的plot方法绘图，bar表示绘制柱状图
df1.plot.scatter(x='a', y='b')  # scatter表示绘制散点图，x，y由DataFrame类型的列指定，映射到散点图的x，y轴上
df1.plot.line(figsize=(20, 5))  # 绘制折线图
plt.xticks(range(len(df1.index)), df1.index, rotation=0)  # 设置x轴的标签显示
df1.plot.area()  # 绘制面积图
plt.xticks(range(len(df1.index)), df1.index, rotation=0)

plt.show()  # pandas库中没有显示图形的方法，要显示图形需要用到pyplot模块的show方法
