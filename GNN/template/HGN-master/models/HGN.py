from models.layers import *
from csr_mhqa.utils import count_parameters


class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.hidden_dim = config.hidden_dim

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        self.graph_blocks = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config))

        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)

        q_dim = self.hidden_dim if config.q_update else config.input_dim

        self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        # 使用 RoBERTa 完成编码后的 Q（query_encoding） & C（context_encoding） 作为输入
        # 这一层是 RoBERTa Encoder 后面紧跟的 bi-attention layer，需要继续处理
        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)

        # 经过 RoBERTa + bi-attention layer 得到完全编码的 Q & C

        # 以 编码后的 context representation C 作为输入
        # 1 经过 bi-attention-linear layer
        # 2 经过 LSTM
        # output 最终提取出来不同节点的representation（paragraph, sentence, entity三种不同的节点）
        # output p, s, e
        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])


        # 对Q使用了 mean-pooling 而非 论文中的 max-pooling
        # 输出 q
        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        # 拼接 H = [q, p, s, e]，作为 输入
        # 经过 GAT
        # 输出 hat(H), 也就是 graph_state
        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, input_state, query_vec)

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        # gate attention layer
        # 以 提取出来的[q, p, s, e] 和 图的输出 hat(H) 作为输入
        # 输出 最终的 node representation
        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))

        # 将最终的representation 放入简单的 MLP分类器中，得到预测的分类结果
        predictions = self.predict_layer(batch, input_state, sent_logits[-1], packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]
