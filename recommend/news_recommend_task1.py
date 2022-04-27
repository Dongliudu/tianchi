#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File:  news_recommend_task1.py    
@Modify Time   @Author   @Version   @Desciption
------------   -------   --------   -----------
2022/1/18 4:01 下午   ghj      1.0         数据分析
"""
# 导包
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import logging, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc, re, warnings, sys

plt.rc('font', family='SimHei', size=13)
warnings.filterwarnings("ignore")

"""数据分析
数据分析的价值主要在于熟悉了解整个数据集的基本情况包括每个文件里有哪些数据，具体的文件中的每个字段表示什么实际含义，以及数据集中特征之间的相关性，
在推荐场景下主要就是分析用户本身的基本属性，文章基本属性，以及用户和文章交互的一些分布，这些都有利于后面的召回策略的选择，以及特征工程。

建议：当特征工程和模型调参已经很难继续上分了，可以回来在重新从新的角度去分析这些数据，或许可以找到上分的灵感"""

# 读取数据
path = '/Users/a58/PycharmProjects/data/'
trn_click = pd.read_csv(path + 'train_click_log.csv')
item_df = pd.read_csv(path + 'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  # 重命名，方便后续match
item_emb_df = pd.read_csv(path + 'articles_emb.csv')
#####test
tst_click = pd.read_csv(path + 'testA_click_log.csv')

# 数据预处理¶
# 对每个用户的点击时间戳进行排序
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)

# 计算用户点击文章的次数，并添加新的一列count
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')

trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
print(trn_click.head())

"""train_click_log.csv文件数据中每个字段的含义
user_id: 用户的唯一标识
click_article_id: 用户点击的文章唯一标识
click_timestamp: 用户点击文章时的时间戳
click_environment: 用户点击文章的环境
click_deviceGroup: 用户点击文章的设备组
click_os: 用户点击文章时的操作系统
click_country: 用户点击文章时的所在的国家
click_region: 用户点击文章时所在的区域
click_referrer_type: 用户点击文章时，文章的来源"""

# 用户点击日志信息
print(trn_click.info())
print(trn_click.describe())

# 训练集中的用户数量为20w
print(trn_click.user_id.nunique())

print(trn_click.groupby('user_id')['click_article_id'].count().min())  # 训练集里面每个用户至少点击了两篇文章

plt.figure()
plt.figure(figsize=(15, 20))
i = 1
for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os',
            'click_country',
            'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
    plot_envs = plt.subplot(5, 2, i)
    i += 1
    v = trn_click[col].value_counts().reset_index()[:10]
    fig = sns.barplot(x=v['index'], y=v[col])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.title(col)
plt.tight_layout()
plt.show()

print(trn_click[
          'click_environment'].value_counts())  # 从点击环境click_environment来看，仅有2102次（占0.19%）点击环境为1；仅有25894次（占2.3%）点击环境为2；剩余（占97.6%）点击环境为4。

print(trn_click['click_deviceGroup'].value_counts())  # 从点击设备组click_deviceGroup来看，设备1占大部分（61%），设备3占36%。

# 测试集用户点击日志
tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
print(tst_click.head())
print(tst_click.describe())

"""我们可以看出训练集和测试集的用户是完全不一样的

训练集的用户ID由0 ~ 199999，而测试集A的用户ID由200000 ~ 249999。

因此，也就是我们在训练时，需要把测试集的数据也包括在内，称为全量数据。

!!!!!!!!!!!!!!!后续将对训练集和测试集合并分析!!!!!!!!!!!"""

# 测试集中的用户数量为5w
print(tst_click.user_id.nunique())

# 新闻文章信息数据表
print(item_df.head().append(item_df.tail()))

print(item_df['words_count'].value_counts())

print(item_df['category_id'].nunique())  # 461个文章主题
item_df['category_id'].hist()

print(item_df.shape)  # 364047篇文章

# 新闻文章embedding向量表示
print(item_emb_df.head())
print(item_emb_df.shape)

# 数据分析
# 用户重复点击
#####merge
user_click_merge = trn_click.append(tst_click)
# 用户重复点击
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg(
    {'count'}).reset_index()
print(user_click_count[:10])
print(user_click_count[user_click_count['count'] > 7])
print(user_click_count['count'].unique())
# 用户点击新闻次数
print(user_click_count.loc[:, 'count'].value_counts())
"""可以看出：有1605541（约占99.2%）的用户未重复阅读过文章，仅有极少数用户重复点击过某篇文章。 这个也可以单独制作成特征"""


# 用户点击环境变化分析

def plot_envs(df, cols, r, c):
    plt.figure()
    plt.figure(figsize=(10, 5))
    i = 1
    for col in cols:
        plt.subplot(r, c, i)
        i += 1
        v = df[col].value_counts().reset_index()
        fig = sns.barplot(x=v['index'], y=v[col])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()


# 分析用户点击环境变化是否明显，这里随机采样10个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=10, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
for _, user_df in sample_users.groupby('user_id'):
    plot_envs(user_df, cols, 2, 3)

"""可以看出绝大多数数的用户的点击环境是比较固定的。思路：可以基于这些环境的统计特征来代表该用户本身的属性"""

# 用户点击新闻数量的分布
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
plt.plot(user_click_item_count)

"""可以根据用户的点击文章次数看出用户的活跃度"""

# 点击次数在前50的用户
plt.plot(user_click_item_count[:50])
"""点击次数排前50的用户的点击次数都在100次以上。思路：我们可以定义点击次数大于等于100次的用户为活跃用户，这是一种简单的处理思路， 判断用户活跃度，
更加全面的是再结合上点击时间，后面我们会基于点击次数和点击时间两个方面来判断用户活跃度。"""

# 点击次数排名在[25000:50000]之间
plt.plot(user_click_item_count[25000:50000])

"""可以看出点击次数小于等于两次的用户非常的多，这些用户可以认为是非活跃用户"""

# 新闻点击次数分析
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
plt.plot(item_click_count)
plt.plot(item_click_count[:100])

"""可以看出点击次数最多的前100篇新闻，点击次数大于1000次"""

plt.plot(item_click_count[:20])
"""点击次数最多的前20篇新闻，点击次数大于2500。思路：可以定义这些新闻为热门新闻， 这个也是简单的处理方式，后面我们也是根据点击次数和时间进行文章热度的一个划分。"""

plt.plot(item_click_count[3500:])

"""可以发现很多新闻只被点击过一两次。思路：可以定义这些新闻是冷门新闻"""

# 新闻共现频次：两篇新闻连续出现的次数

tmp = user_click_merge.sort_values('click_timestamp')
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x: x.shift(-1))
union_item = tmp.groupby(['click_article_id', 'next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values(
    'count', ascending=False)
print(union_item[['count']].describe())

"""由统计数据可以看出，平均共现次数3.18，最高为2202。

说明用户看的新闻，相关性是比较强的。"""

# 画个图直观地看一看
x = union_item['click_article_id']
y = union_item['count']
plt.scatter(x, y)

plt.plot(union_item['count'].values[40000:])

"""大概有75000个pair至少共现一次"""

# 新闻文章信息

# 不同类型的新闻出现的次数
plt.plot(user_click_merge['category_id'].value_counts().values)
# 出现次数比较少的新闻类型, 有些新闻类型，基本上就出现过几次
plt.plot(user_click_merge['category_id'].value_counts().values[150:])
# 新闻字数的描述性统计
user_click_merge['words_count'].describe()
plt.plot(user_click_merge['words_count'].values)

# 用户点击的新闻类型的偏好
# 此特征可以用于度量用户的兴趣是否广泛。
plt.plot(sorted(user_click_merge.groupby('user_id')['category_id'].nunique(), reverse=True))
# 从上图中可以看出有一小部分用户阅读类型是极其广泛的，大部分人都处在20个新闻类型以下。
print(user_click_merge.groupby('user_id')['category_id'].nunique().reset_index().describe())

# 用户查看文章的长度的分布
# 通过统计不同用户点击新闻的平均字数，这个可以反映用户是对长文更感兴趣还是对短文更感兴趣。
print(plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)))

"""从上图中可以发现有一小部分人看的文章平均词数非常高，也有一小部分人看的平均文章次数非常低。

大多数人偏好于阅读字数在200-400字之间的新闻。"""

# 挑出大多数人的区间仔细看看
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)[1000:45000])
"""可以发现大多数人都是看250字以下的文章"""

# 更加详细的参数
print(user_click_merge.groupby('user_id')['words_count'].mean().reset_index().describe())

# 用户点击新闻的时间分析
# 为了更好的可视化，这里把时间进行归一化操作
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])

user_click_merge = user_click_merge.sort_values('click_timestamp')
print(user_click_merge.head())


def mean_diff_time_func(df, col):
    df = pd.DataFrame(df, columns={col})
    df['time_shift1'] = df[col].shift(1).fillna(0)
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()


# 点击时间差的平均值
mean_diff_click_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(
    lambda x: mean_diff_time_func(x, 'click_timestamp'))

plt.plot(sorted(mean_diff_click_time.values, reverse=True))

"""从上图可以发现不同用户点击文章的时间差是有差异的"""

# 前后点击文章的创建时间差的平均值
mean_diff_created_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(
    lambda x: mean_diff_time_func(x, 'created_at_ts'))
plt.plot(sorted(mean_diff_created_time.values, reverse=True))

"""从图中可以发现用户先后点击文章，文章的创建时间也是有差异的"""


# 需要注意这里模型只迭代了一次
def trian_item_word2vec(click_df, embed_size=16, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, epochs=10)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v[k] for k in click_df['click_article_id']}

    return item_w2v_emb_dict


item_w2v_emb_dict = trian_item_word2vec(user_click_merge)
# 随机选择5个用户，查看这些用户前后查看文章的相似性
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

print(sub_user_info.head())


# 上一个版本，这个函数使用的是赛题提供的词向量，但是由于给出的embedding并不是所有的数据的embedding，所以运行下面画图函数的时候会报keyerror的错误
# 为了防止出现这个错误，这里修改为使用word2vec训练得到的词向量进行可视化
def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list) - 1):
        emb1 = item_w2v_emb_dict[str(item_list[i])]  # 需要注意的是word2vec训练时候使用的是str类型的数据
        emb2 = item_w2v_emb_dict[str(item_list[i + 1])]
        sim_list.append(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2))))
    sim_list.append(0)
    return sim_list


for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)

# 这里由于对词向量的训练迭代次数不是很多，所以看到的可视化结果不是很准确，可以训练更多次来观察具体的现象。


"""总结
通过数据分析的过程， 我们目前可以得到以下几点重要的信息， 这个对于我们进行后面的特征制作和分析非常有帮助：

训练集和测试集的用户id没有重复，也就是测试集里面的用户模型是没有见过的
训练集中用户最少的点击文章数是2， 而测试集里面用户最少的点击文章数是1
用户对于文章存在重复点击的情况， 但这个都存在于训练集里面
同一用户的点击环境存在不唯一的情况，后面做这部分特征的时候可以采用统计特征
用户点击文章的次数有很大的区分度，后面可以根据这个制作衡量用户活跃度的特征
文章被用户点击的次数也有很大的区分度，后面可以根据这个制作衡量文章热度的特征
用户看的新闻，相关性是比较强的，所以往往我们判断用户是否对某篇文章感兴趣的时候， 在很大程度上会和他历史点击过的文章有关
用户点击的文章字数有比较大的区别， 这个可以反映用户对于文章字数的区别
用户点击过的文章主题也有很大的区别， 这个可以反映用户的主题偏好 
不同用户点击文章的时间差也会有所区别， 这个可以反映用户对于文章时效性的偏好
所以根据上面的一些分析，可以更好的帮助我们后面做好特征工程， 充分挖掘数据的隐含信息。
"""
