#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File:  news_recommend_task3.py.py    
@Modify Time   @Author   @Version   @Desciption
------------   -------   --------   -----------
2022/1/18 7:14 下午   ghj      1.0         特征处理
"""
"""
制作特征和标签， 转成监督学习问题¶
我们先捋一下基于原始的给定数据， 有哪些特征可以直接利用：

文章的自身特征， category_id表示这文章的类型， created_at_ts表示文章建立的时间， 这个关系着文章的时效性， words_count是文章的字数， 一般字数太长我们不太喜欢点击, 也不排除有人就喜欢读长文。
文章的内容embedding特征， 这个召回的时候用过， 这里可以选择使用， 也可以选择不用， 也可以尝试其他类型的embedding特征， 比如W2V等
用户的设备特征信息
上面这些直接可以用的特征， 待做完特征工程之后， 直接就可以根据article_id或者是user_id把这些特征加入进去。 但是我们需要先基于召回的结果， 构造一些特征，然后制作标签，形成一个监督学习的数据集。

构造监督数据集的思路， 根据召回结果， 我们会得到一个{user_id: [可能点击的文章列表]}形式的字典。 那么我们就可以对于每个用户， 每篇可能点击的文章构造一个监督测试集， 比如对于用户user1， 假设得到的他的召回列表{user1: [item1, item2, item3]}， 我们就可以得到三行数据(user1, item1), (user1, item2), (user1, item3)的形式， 这就是监督测试集时候的前两列特征。


构造特征的思路是这样， 我们知道每个用户的点击文章是与其历史点击的文章信息是有很大关联的， 比如同一个主题， 相似等等。 所以特征构造这块很重要的一系列特征是要结合用户的历史点击文章信息。我们已经得到了每个用户及点击候选文章的两列的一个数据集， 而我们的目的是要预测最后一次点击的文章， 比较自然的一个思路就是和其最后几次点击的文章产生关系， 这样既考虑了其历史点击文章信息， 又得离最后一次点击较近，因为新闻很大的一个特点就是注重时效性。 往往用户的最后一次点击会和其最后几次点击有很大的关联。 所以我们就可以对于每个候选文章， 做出与最后几次点击相关的特征如下：

候选item与最后几次点击的相似性特征(embedding内积） --- 这个直接关联用户历史行为
候选item与最后几次点击的相似性特征的统计特征 --- 统计特征可以减少一些波动和异常
候选item与最后几次点击文章的字数差的特征 --- 可以通过字数看用户偏好
候选item与最后几次点击的文章建立的时间差特征 --- 时间差特征可以看出该用户对于文章的实时性的偏好
还需要考虑一下 5. 如果使用了youtube召回的话， 我们还可以制作用户与候选item的相似特征

当然， 上面只是提供了一种基于用户历史行为做特征工程的思路， 大家也可以思维风暴一下，尝试一些其他的特征。 下面我们就实现上面的这些特征的制作， 下面的逻辑是这样：

我们首先获得用户的最后一次点击操作和用户的历史点击， 这个基于我们的日志数据集做
基于用户的历史行为制作特征， 这个会用到用户的历史点击表， 最后的召回列表， 文章的信息表和embedding向量
制作标签， 形成最后的监督学习数据集
"""
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import logging
import time
import lightgbm as lgb
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 节省内存的一个函数
# 减少内存
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

data_path = './data_raw/'
save_path = './temp_results/'

"""
数据读取¶
训练和验证集的划分
划分训练和验证集的原因是为了在线下验证模型参数的好坏，为了完全模拟测试集，我们这里就在训练集中抽取部分用户的所有信息来作为验证集。
提前做训练验证集划分的好处就是可以分解制作排序特征时的压力，一次性做整个数据集的排序特征可能时间会比较长。
"""


# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量
def trn_val_split(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)

    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)

    # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
    # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]  # 保证答案中出现的用户再验证集中还有
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans

#获取历史点击和最后一次点击¶
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df

#读取训练、验证及测试集¶
def get_trn_val_tst_data(data_path, offline=True):
    if offline:
        click_trn_data = pd.read_csv(data_path + 'train_click_log.csv')  # 训练集用户点击日志
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums=1000)
    else:
        click_trn = pd.read_csv(data_path + 'train_click_log.csv')
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None

    click_tst = pd.read_csv(data_path + 'testA_click_log.csv')

    return click_trn, click_val, click_tst, val_ans

#读取召回列表¶
# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    if multi_recall:
        return pickle.load(open(save_path + 'final_recall_items_dict.pkl', 'rb'))

    if single_recall_model == 'i2i_itemcf':
        return pickle.load(open(save_path + 'itemcf_recall_dict.pkl', 'rb'))
    elif single_recall_model == 'i2i_emb_itemcf':
        return pickle.load(open(save_path + 'itemcf_emb_dict.pkl', 'rb'))
    elif single_recall_model == 'user_cf':
        return pickle.load(open(save_path + 'youtubednn_usercf_dict.pkl', 'rb'))
    elif single_recall_model == 'youtubednn':
        return pickle.load(open(save_path + 'youtube_u2i_dict.pkl', 'rb'))


def trian_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(docs, size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=1)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v[k] for k in click_df['click_article_id']}
    pickle.dump(item_w2v_emb_dict, open(save_path + 'item_w2v_emb.pkl', 'wb'))

    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + 'item_content_emb.pkl'):
        item_content_emb_dict = pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
    else:
        print('item_content_emb.pkl 文件不存在...')

    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))
    else:
        item_w2v_emb_dict = trian_item_word2vec(all_click_df)

    if os.path.exists(save_path + 'item_youtube_emb.pkl'):
        item_youtube_emb_dict = pickle.load(open(save_path + 'item_youtube_emb.pkl', 'rb'))
    else:
        print('item_youtube_emb.pkl 文件不存在...')

    if os.path.exists(save_path + 'user_youtube_emb.pkl'):
        user_youtube_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
    else:
        print('user_youtube_emb.pkl 文件不存在...')

    return item_content_emb_dict, item_w2v_emb_dict, item_youtube_emb_dict, user_youtube_emb_dict

# 读取文章信息
def get_article_info_df():
    article_info_df = pd.read_csv(data_path + 'articles.csv')
    article_info_df = reduce_mem(article_info_df)

    return article_info_df

# 读取数据
# 这里offline的online的区别就是验证集是否为空
click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=False)
click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)

if click_val is not None:
    click_val_hist, click_val_last = click_val, val_ans
else:
    click_val_hist, click_val_last = None, None

click_tst_hist = click_tst

"""
对训练数据做负采样
通过召回我们将数据转换成三元组的形式（user1, item1, label）的形式，观察发现正负样本差距极度不平衡，我们可以先对负样本进行下采样，下采样的目的一方面缓解了正负样本比例的问题，
另一方面也减小了我们做排序特征的压力，我们在做负采样的时候又有哪些东西是需要注意的呢？

只对负样本进行下采样(如果有比较好的正样本扩充的方法其实也是可以考虑的)
负采样之后，保证所有的用户和文章仍然出现在采样之后的数据中
下采样的比例可以根据实际情况人为的控制
做完负采样之后，更新此时新的用户召回文章列表，因为后续做特征的时候可能用到相对位置的信息。
其实负采样也可以留在后面做完特征在进行，这里由于做排序特征太慢了，所以把负采样的环节提到前面了。
"""


# 将召回列表转换成df的形式
def recall_dict_2_df(recall_list_dict):
    df_row_list = []  # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df


# 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值
def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]

    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data) / len(neg_data))

    # 分组采样函数
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)  # 保证最少有一个
        sample_num = min(sample_num, 5)  # 保证最多不超过5个，这里可以根据实际情况进行选择
        return group_df.sample(n=sample_num, replace=True)

    # 对用户进行负采样，保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # 对文章进行负采样，保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # 将上述两种情况下的采样数据合并
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')

    # 将正样本数据合并
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)

    return data_new


# 召回数据打标签
def get_rank_label_df(recall_list_df, label_df, is_test=False):
    # 测试集是没有标签了，为了后面代码同一一些，这里直接给一个负数替代
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df

    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']], \
                                           how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']

    return recall_list_df_


def get_user_recall_item_label_df(click_trn_hist, click_val_hist, click_tst_hist, click_trn_last, click_val_last,
                                  recall_list_df):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    # 训练数据打标签
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    # 训练数据负采样
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)

    if click_val is not None:
        val_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_user_item_label_df = get_rank_label_df(val_user_items_df, click_val_last, is_test=False)
        val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)
    else:
        val_user_item_label_df = None

    # 测试数据不需要进行负采样，直接对所有的召回商品进行打-1标签
    tst_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_tst_hist['user_id'].unique())]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)

    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df

# 读取召回列表
recall_list_dict = get_recall_list(save_path, single_recall_model='i2i_itemcf') # 这里只选择了单路召回的结果，也可以选择多路召回结果
# 将召回数据转换成df
recall_list_df = recall_dict_2_df(recall_list_dict)

# 给训练验证数据打标签，并负采样（这一部分时间比较久）
trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(click_trn_hist,
                                                                                                       click_val_hist,
                                                                                                       click_tst_hist,
                                                                                                       click_trn_last,
                                                                                                       click_val_last,
                                                                                                       recall_list_df)

print(trn_user_item_label_df.label)
