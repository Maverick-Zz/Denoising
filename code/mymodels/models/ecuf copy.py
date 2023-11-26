# -*- coding:utf-8 -*-

import sys
sys.path.append("../")

from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.sequence import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)


class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        x = x + self.bias
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = nn.Linear(dim, 1, bias=False)
        self.fc_ev = nn.Linear(dim, 1, bias=False)
        self.fc_ve = nn.Linear(dim, 1, bias=False)
        self.fc_ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = Bias(dim)
        self.bias_e = Bias(dim)

        # self.fc_v = nn.Linear(dim, dim)
        # self.fc_e = nn.Linear(dim, dim)

    def forward(self, inputs):
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)

        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0,2,1)

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)

        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_intermediate = v_intermediate.view(-1, self.dim)
        e_intermediate = e_intermediate.view(-1, self.dim)

        v_output = self.bias_v(v_intermediate)
        e_output = self.bias_e(e_intermediate)


        return v_output, e_output






class ECUF(BaseModel):
    """Instantiates the Deep Interest Network architecture.

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

    def __init__(self, dnn_feature_columns, target_feature_list, 
                 pv_history_fc_names=None, click_history_fc_names=None, like_history_fc_names=None, dislike_history_fc_names=None, like_aug_history_fc_names=None, dislike_aug_history_fc_names=None, 
                 dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', gpus=None, args=None):
        super(ECUF, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.target_feature_list = target_feature_list


        # 行为序列
        self.pv_history_feature_columns = []
        self.click_history_feature_columns = []
        self.like_history_feature_columns = []
        self.dislike_history_feature_columns = []
        self.like_aug_history_feature_columns = []
        self.dislike_aug_history_feature_columns = []


        self.target_feature_columns = []
        self.sparse_varlen_feature_columns = []

        self.pv_history_fc_names = pv_history_fc_names
        self.click_history_fc_names = click_history_fc_names
        self.like_history_fc_names = like_history_fc_names
        self.dislike_history_fc_names = dislike_history_fc_names
        self.like_aug_history_fc_names = like_aug_history_fc_names
        self.dislike_aug_history_fc_names = dislike_aug_history_fc_names

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.pv_history_fc_names:
                self.pv_history_feature_columns.append(fc)

            elif feature_name in self.click_history_fc_names:
                self.click_history_feature_columns.append(fc)

            elif feature_name in self.like_history_fc_names:
                self.like_history_feature_columns.append(fc)

            elif feature_name in self.dislike_history_fc_names:
                self.dislike_history_feature_columns.append(fc)

            elif feature_name in self.like_aug_history_fc_names:
                self.like_aug_history_feature_columns.append(fc)

            elif feature_name in self.dislike_aug_history_fc_names:
                self.dislike_aug_history_feature_columns.append(fc)

            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()
        print("att_emb_dim is: ", att_emb_dim)


        """ 对比学习 """
        self.contrastive_type = args.contrastive_type
        self.alpha = args.alpha
        self.eps = args.eps


        """ multi-head target attention """
        self.d_model = att_emb_dim
        self.d_ff = args.d_ff
        self.dropout_rate = args.hidden_dropout_prob
        self.mhta_head = args.num_attention_heads
        self.seq_len = args.max_seq_length

        # pv mhta
        self.pv_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.pv_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.pv_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.pv_attn), self.pv_ff, self.dropout_rate)
        self.pv_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        # click mhta
        self.click_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.click_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.click_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.click_attn), self.click_ff, self.dropout_rate)
        self.click_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        # like mhta
        self.like_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.like_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.like_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.like_attn), self.like_ff, self.dropout_rate)
        self.like_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        # dislike mhta
        self.dislike_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.dislike_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.dislike_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.dislike_attn), self.dislike_ff, self.dropout_rate)
        self.dislike_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)


        """ denoise """
        self.args = args
        self.item_encoder = Encoder(args)
        self.click_item_encoder = Encoder(args)


        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns) - self.d_model,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

        """ interests fusion """
        self.fusion_query = nn.Linear(self.d_model, self.d_model)
        self.fusion_key = nn.Linear(self.d_model, self.d_model)
        self.fusion_value = nn.Linear(self.d_model, self.d_model)

        self.pv_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.click_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.like_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.dislike_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )

        self.pv_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.click_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.like_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )
        self.dislike_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.Sigmoid()
        )

        self.to(device)


    def get_mask(self, keys_emb, keys_length):
        """
            获得行为序列的 mask
            参数:
                keys_emb: 行为序列表征
                keys_length: 行为序列的长度
        """

        # denoise
        batch_size, max_length, _ = keys_emb.size()

        keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype).repeat(batch_size,
                                                                                                1)  # [B, T]
        keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
        keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]
        attention_mask = keys_masks.squeeze()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.target_feature_list, to_list=True)

        # pv_seq
        pv_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.pv_history_feature_columns,
                                         return_feat_list=self.pv_history_fc_names, to_list=True)
        
        # click_seq
        click_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.click_history_feature_columns,
                                         return_feat_list=self.click_history_fc_names, to_list=True)

        # like_seq
        like_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.like_history_feature_columns,
                                         return_feat_list=self.like_history_fc_names, to_list=True)

        # dislike_seq
        dislike_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dislike_history_feature_columns,
                                         return_feat_list=self.dislike_history_fc_names, to_list=True)
        
        # like_aug_seq
        like_aug_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.like_aug_history_feature_columns,
                                         return_feat_list=self.like_aug_history_fc_names, to_list=True)

        # dislike_aug_seq
        dislike_aug_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dislike_aug_history_feature_columns,
                                         return_feat_list=self.dislike_aug_history_fc_names, to_list=True)


        # 其他的输入
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        
        # pv attention
        pv_keys_emb = torch.cat(pv_keys_emb_list, dim=-1)                       # [B, T, E]
        click_keys_emb = torch.cat(click_keys_emb_list, dim=-1)                       # [B, T, E]
        like_keys_emb = torch.cat(like_keys_emb_list, dim=-1)                       # [B, T, E]
        dislike_keys_emb = torch.cat(dislike_keys_emb_list, dim=-1)                       # [B, T, E]
        like_aug_keys_emb = torch.cat(like_aug_keys_emb_list, dim=-1)                       # [B, T, E]
        dislike_aug_keys_emb = torch.cat(dislike_aug_keys_emb_list, dim=-1)                       # [B, T, E]



        # pv seq len
        pv_keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        pv_keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, pv_keys_length_feature_name), 1)  # [B, 1]

        # click seq len
        click_keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        click_keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, click_keys_length_feature_name), 1)  # [B, 1]

        # mask
        pv_extended_attention_mask = self.get_mask(pv_keys_emb, pv_keys_length)
        click_extended_attention_mask = self.get_mask(click_keys_emb, click_keys_length)



        """ denoise """
        pv_keys_emb = self.item_encoder(pv_keys_emb,
                                                pv_extended_attention_mask,
                                                output_all_encoded_layers=True)[-1]

        click_keys_emb = self.item_encoder(click_keys_emb,
                                                click_extended_attention_mask,
                                                output_all_encoded_layers=True)[-1]

  
        """ constrative loss """
        if self.contrastive_type == "adv":  # 对抗
            # 生成随机对抗的噪声, 把噪声数据添加到 query 表征中
            random_noise = torch.empty(like_keys_emb.shape).uniform_().to(like_keys_emb.device)
            like_keys_emb_contractive = like_keys_emb + torch.mul(torch.sign(like_keys_emb), torch.nn.functional.normalize(random_noise,p=2,dim=1)) * self.eps
            random_noise = torch.empty(dislike_keys_emb.shape).uniform_().to(dislike_keys_emb.device)
            dislike_keys_emb_contractive = dislike_keys_emb + torch.mul(torch.sign(dislike_keys_emb), torch.nn.functional.normalize(random_noise,p=2,dim=1)) * self.eps
        
        elif self.contrastive_type == 'shuffle': # 打乱
            like_idx = torch.randperm(like_keys_emb.shape[1])
            like_keys_emb_contractive = like_keys_emb[:, like_idx].view(like_keys_emb.size())
            dislike_idx = torch.randperm(dislike_keys_emb.shape[1])
            dislike_keys_emb_contractive = dislike_keys_emb[:, dislike_idx].view(dislike_keys_emb.size())

        elif self.contrastive_type == "dropout": 
            like_keys_emb_contractive = F.dropout(like_keys_emb, self.dropout_rate, True)
            dislike_keys_emb_contractive = F.dropout(dislike_keys_emb, self.dropout_rate, True)

        elif self.contrastive_type == 'aug':
            like_keys_emb_contractive = like_aug_keys_emb
            dislike_keys_emb_contractive = dislike_aug_keys_emb

        else:
            raise Exception('Unknown contrastive_type.')


        # 取 mean pooling
        like_keys_emb_mean = torch.mean(like_keys_emb, dim=1, keepdim=False)
        like_keys_emb_contractive_mean = torch.mean(like_keys_emb_contractive, dim=1, keepdim=False)
        dislike_keys_emb_mean = torch.mean(dislike_keys_emb, dim=1, keepdim=False)
        dislike_keys_emb_contractive_mean = torch.mean(dislike_keys_emb_contractive, dim=1, keepdim=False)

        # batch 内当负样本
        batch_size, seq_length, _ = like_keys_emb.shape
        contractive_dot_product = torch.matmul(like_keys_emb_mean, like_keys_emb_contractive_mean.t())  # [bs, bs]
        mask = torch.eye(batch_size).to(like_keys_emb_mean.device) # [bs, bs]
        like_contractive_loss = F.log_softmax(contractive_dot_product, dim=-1) * mask # 交叉熵损失函数
        like_contractive_loss = (-like_contractive_loss.sum(dim=1)).mean()

        contractive_dot_product = torch.matmul(dislike_keys_emb_mean, dislike_keys_emb_contractive_mean.t())  # [bs, bs]
        mask = torch.eye(batch_size).to(dislike_keys_emb_mean.device) # [bs, bs]
        dislike_contractive_loss = F.log_softmax(contractive_dot_product, dim=-1) * mask # 交叉熵损失函数
        dislike_contractive_loss = (-dislike_contractive_loss.sum(dim=1)).mean()

        # like dislike 对比损失相加
        contractive_loss = like_contractive_loss + dislike_contractive_loss

        self.add_auxiliary_loss(contractive_loss, self.alpha) # 辅助 loss




        """ multi-head target attention """
        # pv mhta
        pv_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        pv_positions = self.pv_embeddings_position(pv_positions)
        pv_transformer_output = self.pv_encoder_layer(pv_keys_emb + pv_positions, query_emb.squeeze()) # target attention
        pv_hist = torch.mean(pv_transformer_output, dim=1, keepdim=True)
      
        # click mhta
        click_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        click_positions = self.click_embeddings_position(click_positions)
        click_transformer_output = self.click_encoder_layer(click_keys_emb + click_positions, query_emb.squeeze()) # target attention
        click_hist = torch.mean(click_transformer_output, dim=1, keepdim=True)

        # like mhta
        like_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        like_positions = self.like_embeddings_position(like_positions)
        like_transformer_output = self.like_encoder_layer(like_keys_emb + like_positions, query_emb.squeeze()) # target attention
        like_hist = torch.mean(like_transformer_output, dim=1, keepdim=True)

        # dislike mhta
        dislike_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        dislike_positions = self.dislike_embeddings_position(dislike_positions)
        dislike_transformer_output = self.dislike_encoder_layer(dislike_keys_emb + dislike_positions, query_emb.squeeze()) # target attention
        dislike_hist = torch.mean(dislike_transformer_output, dim=1, keepdim=True)


        """ interests fusion """
        # 拼接四类兴趣
        input_tensor = torch.cat([pv_hist, click_hist, like_hist, dislike_hist], axis=1).to(self.device)

        # qkv
        mixed_query_layer = self.fusion_query(input_tensor)
        mixed_key_layer = self.fusion_key(input_tensor)
        mixed_value_layer = self.fusion_value(input_tensor)

        # attention 权重
        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        fusion_interests = torch.matmul(attention_probs, mixed_value_layer) # [batchsize, 4, dim]

        # gate fusion
        pv_interest = fusion_interests[:, 0, :]
        click_interest = fusion_interests[:, 1, :]
        like_interest = fusion_interests[:, 2, :]
        dislike_interest = fusion_interests[:, 3, :]

        pv_gate_output = self.pv_gate(pv_interest)
        click_gate_output = self.pv_gate(click_interest)
        like_gate_output = self.pv_gate(like_interest)
        dislike_gate_output = self.pv_gate(dislike_interest)

        final_fusion_interests = pv_interest * pv_gate_output + click_interest * click_gate_output + like_interest * like_gate_output + dislike_interest * dislike_gate_output
        final_fusion_interests = final_fusion_interests.unsqueeze(1)


        """ 输出层 """
        deep_input_emb = torch.cat((deep_input_emb, pv_hist, click_hist, like_hist, dislike_hist, final_fusion_interests), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred



    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.target_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


if __name__ == '__main__':
    pass
