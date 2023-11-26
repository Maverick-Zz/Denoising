import pandas as pd
import numpy as np
import datetime
import time
import random
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import pickle

# from pandarallel import pandarallel

tqdm.pandas(desc='pandas bar')

# 使用了tqdm库，该库提供了在Python中为循环和迭代器添加进度条的功能。
# tqdm.pandas(desc='pandas bar')的作用是在使用pandas库进行操作时，显示一个带有描述（'pandas bar'）的进度条。

# pandarallel.initialize(progress_bar=True)
# 并行操作，windows下不支持

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)


# pd.set_option() 是 pandas 库中的一个函数，用于设置 pandas 的一些显示和行为选项。
# 设置pandas库的显示选项。第一行将显示的最大列数设置为无限制（None），这样可以显示所有列。
# 第二行将显示的最大行数设置为10，这样如果DataFrame有超过10行的数据，只会显示部分数据，以防止输出太过庞大。

def save_to_pickle(obj, path):
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)


def read_pickle(path):
    with open(path, 'rb') as fr:
        obj = pickle.load(fr)
        return obj


# 获取文件路径
user_file_path = '../data/ML_Pure/ml-100k/ml-100k.user'
item_file_path = '../data/ML_Pure/ml-100k/ml-100k.item'
inter_file_path = '../data/ML_Pure/ml-100k/ml-100k.inter'

# 读取数据
df_user = pd.read_csv(user_file_path, sep='\t', header=0)
df_item = pd.read_csv(item_file_path, sep='\t', header=0)
df_inter = pd.read_csv(inter_file_path, sep='\t', header=0)

# 给dateframe列名重命名
user_column_names = {'user_id:token': 'user_id', 'age:token': 'age', 'gender:token': 'gender',
                     'occupation:token': 'occupation', 'zip_code:token': 'zip_code'}
df_user.rename(columns=user_column_names, inplace=True)
item_column_names = {'item_id:token': 'item_id', 'movie_title:token_seq': 'movie_title',
                     'release_year:token': 'release_year', 'class:token_seq': 'class'}
df_item.rename(columns=item_column_names, inplace=True)
inter_column_names = {'user_id:token': 'user_id', 'item_id:token': 'item_id', 'rating:float': 'rating',
                      'timestamp:float': 'timestamp'}
df_inter.rename(columns=inter_column_names, inplace=True)


# 对三张表分别进行label_encoder
def preprocess_dataframe(data_frame, label_encoders=None):
    if label_encoders is None:
        label_encoders = {}

    # 对每一列进行处理
    for column in data_frame.columns:
        # 判断特征的数据类型
        # if data_frame[column].dtype == 'object':
        # 对非数字型特征进行Label Encoding
        label_encoder = LabelEncoder()
        data_frame[column] = label_encoder.fit_transform(data_frame[column])
        # 将LabelEncoder对象保存在字典中，以备后用
        label_encoders[column] = label_encoder

    # 返回处理后的DataFrame和LabelEncoder对象字典
    return data_frame, label_encoders


df_user, user_label_encoders = preprocess_dataframe(df_user)
print("Processed df_user")

df_item, item_label_encoders = preprocess_dataframe(df_item)
print("Processed df_item")

inter_label_encoders = {}

# 需要进行Label Encoding的特征
features_to_encode = ['user_id', 'item_id']

for feature in features_to_encode:
    label_encoder = LabelEncoder()
    df_inter[feature] = label_encoder.fit_transform(df_inter[feature])
    inter_label_encoders[feature] = label_encoder

# user_label_encoders 中保存了 user_id 和 item_id 的 Label Encoder 对象
print("Processed df_inter")

# 合并label_encoder之后的三张表为一张大表
df1 = pd.merge(pd.merge(df_inter, df_user), df_item)

userid_to_seq = {}  # 创建一个空字典，用于存储用户数据
full_user_id = list(set(df1['user_id']))  # 获取所有唯一的用户ID

for user_id in tqdm(full_user_id):  # 遍历所有的 user_id，使用tqdm显示循环进度条
    user_data = df1[df1['user_id'] == user_id]  # 从df1数据框架中选择当前用户的数据
    user_data.sort_values(by='timestamp', ascending=False)  # 按照’time_ms‘列降序排列用户数据
    userid_to_seq[user_id] = user_data  # 将排序后的用户数据存储到userid_to_seq字典中，键是用户ID，值是该用户的数据


def get_behavior(user_data, timestamp, item_id_col='item_id', behavior_type='pv', topk=50, item_attributes=[]):
    """
        user_data: 用户全部行为数据的 dataframe
        timestamp: 当前样本的 timestamp
        item_id_col: item_id 的列名
        behavior_type: 行为的类型, pv click like dislike
        topk: 行为序列的最大长度
        item_attributes: item 属性特征
    """

    # 先根据 behavior_type 选择出相应的序列数据
    user_data = user_data[user_data['timestamp'] < timestamp]
    if behavior_type == 'pv':
        seq_data = user_data[user_data['rating'] <= 3]
    elif behavior_type == 'click':
        seq_data = user_data[user_data['rating'] > 3]
    else:
        raise Exception("选择的 behavior_type 有误, 请检查！")

    # 只保留 topk 的数据
    seq_data = seq_data[: topk]

    seq_dict = {}  # 存储

    # item_id 的 seq
    item_id_seq_str = ""
    item_id_seq = list(seq_data[item_id_col])
    item_id_seq = item_id_seq[: topk]
    if len(item_id_seq) >= 1:
        item_id_seq = list(map(lambda x: str(x), item_id_seq))
        item_id_seq_str = ",".join(item_id_seq)
    seq_dict['id_seq_str'] = item_id_seq_str

    # 其他属性的 seq
    for feature in item_attributes:
        item_feature_seq_str = ""
        item_feature_seq = list(seq_data[feature])
        item_feature_seq = item_feature_seq[: topk]
        if len(item_id_seq) >= 1:
            item_feature_seq = list(map(lambda x: str(x), item_feature_seq))
            item_feature_seq_str = ",".join(item_feature_seq)
        seq_dict[feature + '_seq_str'] = item_feature_seq_str

    return seq_dict


# 生成序列
def append_behavior_to_sample(data):
    user_id = data['user_id']
    time_ms = data['timestamp']
    user_data = userid_to_seq[user_id]

    # item 属性特征
    item_attributes = ['item_id', 'release_year', 'class']

    # 获得了 pv_seq
    seq_dict = {}
    seq_dict['pv_seq_dict'] = get_behavior(user_data, time_ms, behavior_type='pv', topk=50,
                                           item_attributes=item_attributes)

    # 获得了 click_seq
    seq_dict['click_seq_dict'] = get_behavior(user_data, time_ms, behavior_type='click', topk=50,
                                              item_attributes=item_attributes)

    # 处理每个 seq
    for behavior_type in ['pv', 'click']:
        behavior_seq_dict = seq_dict[behavior_type + '_seq_dict']
        data[behavior_type + '_seq'] = behavior_seq_dict['id_seq_str']
        for feature in item_attributes:
            data[behavior_type + '_' + feature + '_seq'] = behavior_seq_dict[feature + '_seq_str']

    return data


# 多线程并行处理
data = df1.copy()

# 生成序列
data = data.progress_apply(append_behavior_to_sample, axis=1)

# 存储
data = data.sort_values(by='timestamp')
data['binary_rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')

save_to_pickle(data, "../data/ML_Pure/processed/more_info.pkl")
