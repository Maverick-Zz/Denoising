import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score
from mymodels.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from mymodels.models import *
import pandas as pd
import numpy as np
import datetime
import time
import pickle
from tqdm import tqdm

import tensorflow as tf
from keras.layers import Input, Dense, Multiply
from keras.models import Model

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler



tqdm.pandas(desc='pandas bar')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)


def save_to_pickle(obj, path):
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)


def read_pickle(path):
    with open(path, 'rb') as fr:
        obj = pickle.load(fr)
        return obj


# 读取数据
sample_path = "../data/KuaiRand_Pure/ml-100k/test.pkl"
data = read_pickle(sample_path)


def get_seq_length(seq_str):
    seq_str = seq_str.split(",")
    return len(seq_str)


data['pv_seq_length'] = data['pv_seq'].apply(get_seq_length)
data['click_seq_length'] = data['click_seq'].apply(get_seq_length)

#  通过使用 get_seq_length函数，将data数据框中'pv_seq'列中的逗号分隔字符串转换为长度，然后将结果存储在新的 'pv_seq_length' 列中。

# 定义 features
sparse_features = ["user_id", "age", "gender",
                   "occupation", "zip_code",

                   "item_id", "movie_title",  "genre"]

dense_features = ["age","release_year"]

seq_features = ['pv_seq', 'click_seq']

target = ['binary_rating']

pv_seq_key2index = {}
click_seq_key2index = {}



# 使用 map 函数，将 key_ans 列表中的每个元素通过 like_seq_key2index 字典进行映射，
# 并返回一个新的列表，其中包含了这些元素的索引值。
def pv_seq_split(x):
    key_ans = x.split(',')
    for key in key_ans:
        if key not in pv_seq_key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            pv_seq_key2index[key] = len(pv_seq_key2index) + 1
    return list(map(lambda x: pv_seq_key2index[x], key_ans))


def click_seq_split(x):
    key_ans = x.split(',')
    for key in key_ans:
        if key not in click_seq_key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            click_seq_key2index[key] = len(click_seq_key2index) + 1
    return list(map(lambda x: click_seq_key2index[x], key_ans))


# label encoding
for feat in sparse_features:
    print("LabelEncoder 开始处理: ", feat)
    lbe = LabelEncoder()  # lbe = LabelEncoder(): 创建了一个 LabelEncoder 类的实例对象，该对象用于执行标签编码。
    data[feat] = lbe.fit_transform(data[feat])
# data[feat] = lbe.fit_transform(data[feat]): 对特征 feat 中的所有值进行标签编码，
# 并将编码后的整数值替代原始数据框 data 中的该特征列。

# 在循环结束后，data 数据框中的所有稀疏特征都被成功地转换成了整数值

""" pv_seq """
print("------ start to process pv_seq -------")
pv_seq_list = list(map(pv_seq_split, data['pv_seq'].values))
# 使用 pv_seq_split 函数将 data['pv_seq'] 中的每个逗号分隔字符串转换为整数索引序列，并将这些序列组成一个列表
pv_seq_length = np.array(list(map(len, pv_seq_list)))
# 使用 map 函数计算 pv_seq_list 中每个序列的长度，并将这些长度组成一个 NumPy 数组 pv_seq_length。
# 这个数组表示了 pv_seq_list 中每个序列的长度。
pv_seq_max_len = max(pv_seq_length)
# max 函数找到 pv_seq_length 数组中的最大值，即所有序列中最长的序列的长度。
# pv_seq padding
pv_seq_list = pad_sequences(pv_seq_list, maxlen=pv_seq_max_len, padding='post', )
# 使用 pad_sequences 函数将 pv_seq_list 中的所有序列填充到相同的长度。


""" click_seq """
print("------ start to process click_seq -------")
click_seq_list = list(map(click_seq_split, data['click_seq'].values))
click_seq_length = np.array(list(map(len, click_seq_list)))
click_seq_max_len = max(click_seq_length)

# click_seq padding
click_seq_list = pad_sequences(click_seq_list, maxlen=click_seq_max_len, padding='post', )




# 生成 feature_columns 特征列
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=18)
                          for feat in sparse_features]


# 创建了一个包含了序列特征的 VarLenSparseFeat 对象的列表。
# VarLenSparseFeat 是 DeepCTR 库中定义的一个特征类，用于处理不定长的序列特征。
varlen_feature_columns = [VarLenSparseFeat(SparseFeat('pv_seq', vocabulary_size=len(pv_seq_key2index) + 1, embedding_dim=18),maxlen=pv_seq_max_len, length_name="pv_seq_length"),
                          VarLenSparseFeat(SparseFeat('click_seq', vocabulary_size=len(click_seq_key2index) + 1, embedding_dim=18),maxlen=click_seq_max_len, length_name="click_seq_length")]
#print(varlen_feature_columns)
# 这样创建的 varlen_feature_columns 列表包含了所有的序列特征，每个特征都被定义为一个 VarLenSparseFeat 对象，
# 准备用于构建深度学习模型的输入层。

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


# 将数据划分为训练集、验证集和测试集

split_date = 1045000000

train = data[data['timestamp'] < split_date]
test = data[data['timestamp'] >= split_date]


# 确保数据集大小
print(f"Train samples: {len(train)}, Test samples: {len(test)}")
#Validation samples: {len(val)}

# train 数据和 test 数据
seq_length_feautures_name = ['pv_seq_length', 'click_seq_length']
feature_names += seq_length_feautures_name
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}


# 根据 index 选择 train、val、test 数据的 seq
print("------- start to process seq -----------")
train_pv_seq_list = pv_seq_list[train.index]  # 根据索引从 pv_seq_list 中选择对应的序列数据，用于训练集 (train_pv_seq_list)。
test_pv_seq_list = pv_seq_list[test.index]

train_click_seq_list = click_seq_list[train.index]
test_click_seq_list = click_seq_list[test.index]

# 放到 input 字典中
train_model_input['pv_seq'] = train_pv_seq_list
test_model_input['pv_seq'] = test_pv_seq_list
train_model_input['click_seq'] = train_click_seq_list
test_model_input['click_seq'] = test_click_seq_list


# 去噪模块

# 假设 train_pv_seq_list 和 train_click_seq_list 是列表形式的序列数据
# 将它们转换为 NumPy 数组并进行填充以确保相同长度
max_len = max(pv_seq_max_len, click_seq_max_len)
train_pv_seq_array = pad_sequences(train_pv_seq_list, maxlen=max_len, padding='post')
train_click_seq_array = pad_sequences(train_click_seq_list, maxlen=max_len, padding='post')

# 定义去噪网络
def sequence_denoising_network(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(input_layer)
    #x = Dense(256, activation='relu')(x)# 一个中间层
    output_layer = Dense(input_dim, activation='sigmoid')(x)  # 输出层
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# 创建去噪网络
seq_denoising_net = sequence_denoising_network(max_len)

# 编译模型
seq_denoising_net.compile(optimizer='adam', loss='mse')  # 使用均方误差作为损失函数


# 合并 pv_seq 和 click_seq 数据用于训练
combined_data = np.concatenate([train_pv_seq_array, train_click_seq_array], axis=0)


# Robust Scaling
# scaler = RobustScaler()
# combined_data_normalized = scaler.fit_transform(combined_data)

# 标准化（Z-Score Normalization）
scaler = StandardScaler()
combined_data_normalized = scaler.fit_transform(combined_data)

# 训练和预测时使用 combined_data_normalized
# 使用序列数据本身作为监督信号训练去噪网络
seq_denoising_net.fit(combined_data_normalized, combined_data_normalized, epochs=10, batch_size=1024)

# 将 int32 类型的序列数据转换为 float32
train_pv_seq_array = tf.cast(train_pv_seq_array, tf.float32)
train_click_seq_array = tf.cast(train_click_seq_array, tf.float32)

# 使用训练好的去噪网络处理 pv_seq 和 click_seq
train_pv_seq_gate = seq_denoising_net.predict(train_pv_seq_array)
train_click_seq_gate = seq_denoising_net.predict(train_click_seq_array)

# 将分数与原始序列相乘
train_pv_seq_processed = Multiply()([train_pv_seq_array, train_pv_seq_gate])
train_click_seq_processed = Multiply()([train_click_seq_array, train_click_seq_gate])

# 更新 train_model_input
train_model_input['pv_seq'] = train_pv_seq_processed
train_model_input['click_seq'] = train_click_seq_processed





 # cuda
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'



model = DIN(dnn_feature_columns, target_feature_list=["item_id"], history_fc_names=["click_seq"],
            task='binary',
            l2_reg_embedding=1e-5, device=device)

model.compile("adagrad", "binary_crossentropy",
              metrics=["binary_crossentropy", "auc"], )


history = model.fit(train_model_input, train[target].values, batch_size=2048, epochs=50, verbose=2,
                    validation_data=(test_model_input, test[target].values))
