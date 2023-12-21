# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from .basemodelv2 import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer


class DDINMultiV3(BaseModel):
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

    def __init__(self, dnn_feature_columns, target_feature_list, pv_history_fc_names=None, pv_item_id_history_fc_names=None
                 , pv_release_year_history_fc_names=None, pv_class_history_fc_names=None, click_history_fc_names=None
                , click_item_id_history_fc_names=None, click_release_year_history_fc_names=None, click_class_history_fc_names=None, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=True, l2_reg_dnn=0,
                 l2_reg_embedding=1e-6, dnn_dropout=0.3, init_std=0.00034,  #0.0000212
                 seed=1022, task='binary', device='cpu', gpus=None):
        super(DDINMultiV4, self).__init__([], dnn_feature_columns, l2_reg_linear=0.0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.target_feature_list = target_feature_list

        self.pv_history_feature_columns = []
        self.pv_item_id_history_feature_columns = []
        self.pv_release_year_history_feature_columns = []
        self.pv_class_history_feature_columns = []

        self.click_history_feature_columns = []
        self.click_item_id_history_feature_columns = []
        self.click_release_year_history_feature_columns = []
        self.click_class_history_feature_columns = []

        self.target_feature_columns = []
        self.sparse_varlen_feature_columns = []

        self.pv_history_fc_names = pv_history_fc_names
        self.pv_item_id_history_fc_names = pv_item_id_history_fc_names
        self.pv_release_year_history_fc_names = pv_release_year_history_fc_names
        self.pv_class_history_fc_names = pv_class_history_fc_names

        self.click_history_fc_names = click_history_fc_names
        self.click_item_id_history_fc_names = click_item_id_history_fc_names
        self.click_release_year_history_fc_names = click_release_year_history_fc_names
        self.click_class_history_fc_names = click_class_history_fc_names


        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.pv_history_fc_names:
                self.pv_history_feature_columns.append(fc)

            elif feature_name in self.pv_item_id_history_fc_names:
                self.pv_item_id_history_feature_columns.append(fc)

            elif feature_name in self.pv_release_year_history_fc_names:
                self.pv_release_year_history_feature_columns.append(fc)

            elif feature_name in self.pv_class_history_fc_names:
                self.pv_class_history_feature_columns.append(fc)

            elif feature_name in self.click_history_fc_names:
                self.click_history_feature_columns.append(fc)

            elif feature_name in self.click_item_id_history_fc_names:
                self.click_item_id_history_feature_columns.append(fc)

            elif feature_name in self.click_release_year_history_fc_names:
                self.click_release_year_history_feature_columns.append(fc)

            elif feature_name in self.click_class_history_fc_names:
                self.click_class_history_feature_columns.append(fc)

            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.pv_attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=64,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)


        self.click_attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=64,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)



        """ multi-head target attention """
        self.d_model = att_emb_dim

        self.pv_gate_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16, 128, bias=False),
                nn.LeakyReLU(),
                nn.Linear(128, 64, bias=False),
                nn.LeakyReLU(),
                nn.Linear(64, 32, bias=False),
                nn.LeakyReLU(),
                nn.Linear(32, 16, bias=False),
                nn.LeakyReLU(),
                nn.Sigmoid()

            ) for _ in range(4)  # 使用两个分支
        ])

        self.click_gate_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16, 128, bias=False),
                nn.LeakyReLU(),
                nn.Linear(128, 64, bias=False),
                nn.LeakyReLU(),
                nn.Linear(64, 32, bias=False),
                nn.LeakyReLU(),
                nn.Linear(32, 16, bias=False),
                nn.LeakyReLU(),
                nn.Sigmoid()

            ) for _ in range(4)  # 使用两个分支
        ])

        self.query_transform = nn.Linear(16, 64)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)


        # 输出的权重
        self.sample_weight_fc = DNN(inputs_dim=(self.compute_input_dim(dnn_feature_columns) + 1),
                       hidden_units=[64, 1],
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)

        self.to(device)
        print("self.compute_input_dim(dnn_feature_columns)", self.compute_input_dim(dnn_feature_columns))


    def forward(self, X, y=None):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.target_feature_list, to_list=True)

        # pv_seq
        pv_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.pv_history_feature_columns,
                                         return_feat_list=self.pv_history_fc_names, to_list=True)

        pv_item_id_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.pv_item_id_history_feature_columns,
                                            return_feat_list=self.pv_item_id_history_fc_names, to_list=True)

        pv_release_year_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.pv_release_year_history_feature_columns,
                                            return_feat_list=self.pv_release_year_history_fc_names, to_list=True)

        pv_class_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.pv_class_history_feature_columns,
                                            return_feat_list=self.pv_class_history_fc_names, to_list=True)


        # click_seq
        click_keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.click_history_feature_columns,
                                         return_feat_list=self.click_history_fc_names, to_list=True)

        click_item_id_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.click_item_id_history_feature_columns,
                                               return_feat_list=self.click_item_id_history_fc_names, to_list=True)

        click_release_year_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.click_release_year_history_feature_columns,
                                                    return_feat_list=self.click_release_year_history_fc_names, to_list=True)

        click_class_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.click_class_history_feature_columns,
                                             return_feat_list=self.click_class_history_fc_names, to_list=True)


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
        query_emb_transformed = self.query_transform(query_emb)

        #print("query_emb_transformed:", query_emb_transformed.shape)
        pv_keys_emb = torch.cat(pv_keys_emb_list, dim=-1)
        pv_item_id_emb = torch.cat(pv_item_id_emb_list, dim=-1)
        pv_release_year_emb = torch.cat(pv_release_year_emb_list, dim=-1)
        pv_class_emb = torch.cat(pv_class_emb_list, dim=-1)

        pv_keys_emb_processed = self.pv_gate_branches[0](pv_keys_emb)
        pv_item_id_emb_processed = self.pv_gate_branches[1](pv_item_id_emb)
        pv_release_year_emb_processed = self.pv_gate_branches[2](pv_release_year_emb)
        pv_class_emb_processed = self.pv_gate_branches[3](pv_class_emb)

        pv_keys_emb_processed = pv_keys_emb_processed * pv_keys_emb
        pv_item_id_emb_processed = pv_item_id_emb_processed * pv_item_id_emb
        pv_release_year_emb_processed = pv_release_year_emb_processed * pv_release_year_emb
        pv_class_emb_processed = pv_class_emb_processed * pv_class_emb

        click_keys_emb = torch.cat(click_keys_emb_list, dim=-1)
        click_item_id_emb = torch.cat(click_item_id_emb_list, dim=-1)
        click_release_year_emb = torch.cat(click_release_year_emb_list, dim=-1)
        click_class_emb = torch.cat(click_class_emb_list, dim=-1)

        click_keys_emb_processed = self.pv_gate_branches[0](click_keys_emb)
        click_item_id_emb_processed = self.pv_gate_branches[1](click_item_id_emb)
        click_release_year_emb_processed = self.pv_gate_branches[2](click_release_year_emb)
        click_class_emb_processed = self.pv_gate_branches[3](click_class_emb)

        click_keys_emb_processed = click_keys_emb_processed * click_keys_emb
        click_item_id_emb_processed = click_item_id_emb_processed * click_item_id_emb
        click_release_year_emb_processed = click_release_year_emb_processed * click_release_year_emb
        click_class_emb_processed = click_class_emb_processed * click_class_emb

        combined_pv_seq_emb = torch.cat([pv_keys_emb_processed, pv_item_id_emb_processed, pv_release_year_emb_processed, pv_class_emb_processed],dim=-1)
        combined_click_seq_emb = torch.cat([click_keys_emb_processed, click_item_id_emb_processed, click_release_year_emb_processed,click_class_emb_processed], dim=-1)


        # 行为序列其他部分
        pv_keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        pv_keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, pv_keys_length_feature_name), 1)  # [B, 1]


        pv_hist = self.pv_attention(query_emb_transformed, combined_pv_seq_emb, pv_keys_length)           # [B, 1, E]


        click_keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        click_keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, click_keys_length_feature_name), 1)  # [B, 1]

        click_hist = self.click_attention(query_emb_transformed, combined_click_seq_emb, click_keys_length)           # [B, 1, E]


        # deep part
        deep_input_emb2 = torch.cat((deep_input_emb, pv_hist, click_hist), dim=-1)
        deep_input_emb2 = deep_input_emb2.view(deep_input_emb2.size(0), -1)

        # print("deep_input_emb", deep_input_emb.shape) # [512, 96]
        dnn_input = combined_dnn_input([deep_input_emb2], dense_value_list)

        # print("dnn_input", dnn_input.shape) # [512, 96]

        dnn_output = self.dnn(dnn_input)

        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)


        # sample_weight = None
        # if y != None:
        #     sample_weight_input = torch.cat((deep_input_emb, pv_hist, click_hist), dim=-1)
        #     sample_weight_input = sample_weight_input.view(sample_weight_input.size(0), -1)
        #     sample_weight_input = combined_dnn_input([sample_weight_input], dense_value_list)
        #     sample_weight_input = torch.cat((sample_weight_input, y), dim=-1)  # 正负样本也输入其中
        #     sample_weight = self.sample_weight_fc(sample_weight_input)
        #
        # return y_pred, sample_weight



        sample_weight_adjusted = None
        if y is not None:
            y_expanded = y.view(-1, 1).float()
            sample_weight_input = torch.cat((deep_input_emb, pv_hist, click_hist), dim=-1)
            sample_weight_input = sample_weight_input.view(sample_weight_input.size(0), -1)
            sample_weight_input = torch.cat((sample_weight_input, y_expanded), dim=-1)
            sample_weight = self.sample_weight_fc(sample_weight_input)
         
            # 根据权重均值的偏差进行调整
            positive_mask = (y_expanded == 1).float()
            negative_mask = (y_expanded == 0).float()
            #print(positive_mask)
            positive_mean = (sample_weight * positive_mask).sum() / positive_mask.sum()
            negative_mean = (sample_weight * negative_mask).sum() / negative_mask.sum()

            # 计算调整因子以保持权重均值为1
            total_mean = (positive_mean + negative_mean) / 2
            adjustment = 1 - total_mean

            # 为保持均值为1，正样本和负样本均应用相同的调整
            sample_weight_adjusted = sample_weight + adjustment
            sample_weight_adjusted = torch.clamp(sample_weight_adjusted, 0, 2)  # 限制权重范围
            
        return y_pred, sample_weight_adjusted



    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.target_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


if __name__ == '__main__':
    pass