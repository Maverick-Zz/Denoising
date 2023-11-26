# -*- coding:utf-8 -*-

import sys
sys.path.append("../")

from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.sequence import *






class DFN(BaseModel):
    """
        Deep Feedback Network 

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

    def __init__(self, dnn_feature_columns, target_feature_list, pv_history_fc_names=None, like_history_fc_names=None, click_history_fc_names=None, dislike_history_fc_names=None
                , dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', gpus=None):
        super(DFN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.target_feature_list = target_feature_list

        self.pv_history_feature_columns = []
        self.like_history_feature_columns = []
        self.click_history_feature_columns = []
        self.dislike_history_feature_columns = []

        self.target_feature_columns = []
        self.sparse_varlen_feature_columns = []

        self.pv_history_fc_names = pv_history_fc_names
        self.like_history_fc_names = like_history_fc_names
        self.click_history_fc_names = click_history_fc_names
        self.dislike_history_fc_names = dislike_history_fc_names

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.pv_history_fc_names:
                self.pv_history_feature_columns.append(fc)

            elif feature_name in self.like_history_fc_names:
                self.like_history_feature_columns.append(fc)

            elif feature_name in self.click_history_fc_names:
                self.click_history_feature_columns.append(fc)

            elif feature_name in self.dislike_history_fc_names:
                self.dislike_history_feature_columns.append(fc)

            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()
        print("att_emb_dim is: ", att_emb_dim)

        # multi-head target attention
        self.d_model = att_emb_dim
        self.d_ff = 32
        self.dropout_rate = 0.2
        self.mhta_head = 2
        self.seq_len = 50

        # pv transformer
        self.pv_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.pv_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.pv_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.pv_attn), self.pv_ff, self.dropout_rate)
        self.pv_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        # click transformer
        self.click_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.click_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.click_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.click_attn), self.click_ff, self.dropout_rate)
        self.click_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        # dislike transformer
        self.dislike_attn = MultiHeadedAttention(h=self.mhta_head, d_model=self.d_model)
        self.dislike_ff = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout_rate)
        self.dislike_encoder_layer = EncoderLayer(self.d_model, deepcopy(self.dislike_attn), self.dislike_ff, self.dropout_rate)
        self.dislike_embeddings_position  = nn.Embedding(self.seq_len + 1, self.d_model)

        
        # click unclick attention
        self.click_unclick_attention = DotProductAttention(self.d_model)

        # dislike unclick attention
        self.dislike_unclick_attention = DotProductAttention(self.d_model)

        # fm
        self.fm = FM()


        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)
        print("self.compute_input_dim(dnn_feature_columns)", self.compute_input_dim(dnn_feature_columns))


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

        # dislike_seq
        dislike_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dislike_history_feature_columns,
                                         return_feat_list=self.dislike_history_fc_names, to_list=True)

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
        dislike_keys_emb = torch.cat(dislike_keys_emb_list, dim=-1)                       # [B, T, E]


        # pv transformer
        pv_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        pv_positions = self.pv_embeddings_position(pv_positions)
        pv_transformer_output = self.pv_encoder_layer(pv_keys_emb + pv_positions, query_emb.squeeze()) # target attention
        pv_hist = torch.mean(pv_transformer_output, dim=1, keepdim=True)
      
        # click transformer
        click_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        click_positions = self.click_embeddings_position(click_positions)
        click_transformer_output = self.click_encoder_layer(click_keys_emb + click_positions, query_emb.squeeze()) # target attention
        click_hist = torch.mean(click_transformer_output, dim=1, keepdim=True)

        # dislike transformer
        dislike_positions = torch.arange(0, self.seq_len, 1, dtype=int, device=self.device)
        dislike_positions = self.dislike_embeddings_position(dislike_positions)
        dislike_transformer_output = self.dislike_encoder_layer(dislike_keys_emb + dislike_positions, query_emb.squeeze()) # target attention
        dislike_hist = torch.mean(dislike_transformer_output, dim=1, keepdim=True)


        # click and unclick attention
        click_unclick = self.click_unclick_attention(click_hist, pv_transformer_output)
        
        # dislike and unclick attention
        dislike_unclick = self.dislike_unclick_attention(dislike_hist, pv_transformer_output)

        # FM layer
        fm_emb = torch.cat(query_emb_list, dim=-1) # [B, N, dim]
        fm_cross = self.fm(fm_emb)

        # 拼接, deep_input_emb 作为模型中说的 wide 侧
        deep_input_emb = torch.cat((deep_input_emb, pv_hist, click_hist, dislike_hist, fm_cross, click_unclick, dislike_unclick), dim=-1)
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



class FM(nn.Module):
    """ FM """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class DotProductAttention(nn.Module):
    """
        Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


if __name__ == '__main__':
    pass
