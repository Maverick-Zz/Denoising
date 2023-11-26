# -*- coding:utf-8 -*-


from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer


class DIN(BaseModel):
    """
        Deep Interest Network 

        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param history_feature_list: list,to indicate  sequence sparse field
        :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
        :param dnn_activation: Activation function to use in deep net
        :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
        :param att_activation: Activation function to use in attention net
        :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
        :return:  A PyTorch model instance.

    """
    # 一个类的初始化方法（__init__），用于构造DIN模型的实例.
    def __init__(self, dnn_feature_columns, target_feature_list, history_fc_names=None, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=True, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0.5, init_std=0.0028,
                 seed=1024, task='binary', device='cpu', gpus=None):
        # 调用DIN的父类的方法。
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        # 第一个为空列表 []：这可能表示一个特定类型的特征列，这里传递为空列表，意味着没有特定类型的特征列被直接传递。

        # 在类的初始化方法（__init__）中对 self.sparse_feature_columns 属性进行初始化。
        # 从 dnn_feature_columns 中筛选出所有类型为 SparseFeat 的元素，并将它们作为一个列表赋值给 self.sparse_feature_columns。
        # 如果 dnn_feature_columns 为空，则 self.sparse_feature_columns 被设置为一个空列表。
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.target_feature_list = target_feature_list

        self.history_feature_columns = []
        self.target_feature_columns = []
        self.sparse_varlen_feature_columns = []
        # self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
        self.history_fc_names = history_fc_names


        # 对self.varlen_sparse_feature_columns（可变长度稀疏特征列的列表）中的每个特征列进行分类。
        # 将它们分别添加到 self.history_feature_columns 或 self.sparse_varlen_feature_columns 列表中。
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()
        # 调用了类内部的一个方法 _compute_interest_dim 并将其返回值赋给变量 att_emb_dim。
        # 计算和设置用于注意力机制的嵌入维度。

        # 创建了一个 AttentionSequencePoolingLayer 的实例，
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        # 创建了一个DNN实例，作为模型的一部分。
        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        # 是一个构造线性（全连接）层。 self.dnn_linear是模型中的一个线性层，它接收前一个DNN层的输出，并将其转换为单一输出值。
        # dnn_hidden_units[-1] 表示取 dnn_hidden_units 数组的最后一个元素，这是DNN的最后一个隐藏层的单元数。这个值被用作线性层的输入特征数量。
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)



    # 输入 X 包含用户和商品的信息，如用户ID、商品ID、用户的历史购买序列等。
    # embedding_lookup 为用户ID和商品ID等提供嵌入向量。
    # varlen_embedding_lookup 和 get_varlen_pooling_list 处理用户的历史购买序列，将其转换为固定长度的表示。
    # self.attention 层确定哪些历史购买行为与当前商品最相关。
    # 通过 self.dnn 处理合并后的特征，生成商品的喜好预测。
    # y_pred 表示模型预测的用户对商品的喜好程度。

    def forward(self, X):
        # 用于接收函数返回的多个值，但忽略其中的一个或多个值。
        # self.input_from_feature_columns(...):
        # 这是一个方法调用，它处理输入数据 X 并根据 dnn_feature_columns 和 embedding_dict 提取相应的特征。
        # 这个方法返回两个值。第一个值可能是经过处理的特征数据（比如经过嵌入的特征），而第二个值 dense_value_list 是提取出的稠密特征值列表。
        # 单下划线 _ 通常用作一个临时或不关注的变量名。
        # 在这个上下文中，_ 用来接收 input_from_feature_columns 方法返回的第一个值，但这个值实际上并没有在后续代码中使用，因此使用 _ 表示对其内容不感兴趣。
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        # 定义模型如何处理输入数据 X 并向前传递以产生输出。

        # sequence pooling part

        # embedding_lookup 函数的目的是从给定的嵌入字典中查找并返回特定特征的嵌入向量。
        # 这个函数通常用于处理分类特征，它将这些特征的原始值（如ID或名称）映射到嵌入空间中的密集向量。
        # query_emb_list 接收 embedding_lookup 函数的返回值，它是一个包含了指定特征嵌入向量的列表。
        # 这个列表中的每个元素对应于 self.target_feature_list 中的一个特征的嵌入向量。
        # X: 这是输入数据，可能包含了多个特征。
        # self.embedding_dict: 一个字典，包含了特征名称到其对应嵌入矩阵的映射。
        # self.feature_index: 特征索引，用于标识输入数据 X 中每个特征的位置。
        # self.sparse_feature_columns: 包含稀疏特征列的列表。这些特征列定义了哪些特征需要进行嵌入查找。
        # return_feat_list=self.target_feature_list: 指定要返回的特征列表。这里使用 self.target_feature_list 来指定哪些特征的嵌入向量需要被检索。
        # to_list=True: 指示函数将返回的嵌入向量作为一个列表返回，而不是单个合并的张量。
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.target_feature_list, to_list=True)

        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)

        # dnn_input_emb_list 是函数返回的一个列表，其中包含了 self.sparse_feature_columns 中定义的每个特征的嵌入向量。
        # 这个列表用于后续的神经网络处理，特别是在构建深度神经网络（DNN）的输入时。
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        # 使用 varlen_embedding_lookup 函数来处理输入数据 X 中的可变长度稀疏特征列，并提取相应的嵌入向量。
        # varlen_embedding_lookup 是一个函数，用于处理可变长度的稀疏特征列。
        # 不同于普通的嵌入查找，这个函数专门处理那些具有不同长度的序列特征，如用户的历史行为序列。
        # self.sparse_varlen_feature_columns: 一个包含可变长度稀疏特征列的列表。这些特征列定义了需要进行嵌入查找的序列特征。
        # sequence_embed_dict:
        # sequence_embed_dict 是函数返回的字典，其中包含了每个可变长度稀疏特征列的嵌入向量。
        # 字典的键是特征名称，值是对应的嵌入向量。
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        # 使用 get_varlen_pooling_list 函数来处理 sequence_embed_dict（包含可变长度特征嵌入的字典），生成一个序列嵌入的池化列表。
        # sequence_embed_list 是函数返回的列表，其中包含了池化后的嵌入向量。
        # 每个元素对应于 self.sparse_varlen_feature_columns 中的一个特征，表示该特征的序列经过池化后的嵌入向量。
        # sequence_embed_dict: 包含了可变长度稀疏特征列的嵌入向量的字典。
        # X: 输入数据，包含了多个特征。
        # self.feature_index: 特征索引，用于在输入数据 X 中定位每个特征的位置。
        # self.sparse_varlen_feature_columns: 包含可变长度稀疏特征列的列表，这些特征列定义了需要进行嵌入查找和池化的序列特征。
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        # 将 sequence_embed_list 列表中的元素添加到 dnn_input_emb_list 列表中。
        dnn_input_emb_list += sequence_embed_list

        # 使用 PyTorch 的 torch.cat 函数来将 dnn_input_emb_list 列表中的嵌入向量沿着最后一个维度（dim=-1）拼接起来，生成一个新的张量 deep_input_emb。
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        # 从 self.varlen_sparse_feature_columns（可变长度稀疏特征列的列表）中提取特定属性，并创建一个新的列表 keys_length_feature_name。
        # 这个过程是使用列表推导式完成
        # keys_length_feature_name 是这个列表推导式的结果，它包含了所有具有 length_name 属性的可变长度稀疏特征列的 length_name 值。
        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]

        # 使用 maxlen_lookup 函数从输入数据 X 中提取特定特征的长度信息，并应用 torch.squeeze 函数以调整结果张量的维度。
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]


        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        # print("y_pred", y_pred.shape)

        return y_pred

    # 计算self.sparse_feature_columns 中属于 self.target_feature_list 的特征的嵌入维度总和。
    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.target_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


if __name__ == '__main__':
    pass
