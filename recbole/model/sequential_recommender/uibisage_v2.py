from cgi import test
from copy import deepcopy
from re import I

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import SAGEConv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder
from tqdm import tqdm

class UIBiSage2(SequentialRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(UIBiSage2, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num + 1
        self.n_items = dataset.item_num + 1

        ## graph config
        self.user_embedding_size = config['user_embedding_size']  # input dim user graph sage
        self.item_embedding_size = config['item_embedding_size']  # input dim item graph sage
        self.seq_aggregate = config['seq_aggregate']
        self.u_cutoff = config['u_cutoff'] ## user  cosine sim cut off
        self.i_cutoff = config['i_cutoff'] ## item cosine sim cut off
        self.graph_num_way = config['n_way']
        
        ui_matrix = self.seq_interaction_generation(dataset)
        if self.seq_aggregate:
            self.ui_matrix = ui_matrix
        else:
            self.ui_matrix = (ui_matrix!=0)
        
        user_adj = self.adjacency_generation(self.ui_matrix, self.u_cutoff) #0.75)
        item_adj = self.adjacency_generation(self.ui_matrix.T, self.i_cutoff) #0.75)
        self.user_graph = self.graph_generation(user_adj, 'user')
        self.item_graph = self.graph_generation(item_adj, 'item')
        # import pdb;pdb.set_trace()

        # transformer paramter
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.device = config['device']

        # define Graph sage layers
        self.I2U_SAGE = nn.ModuleList([SAGEConv(self.user_embedding_size, self.hidden_size, 'mean')])
        self.I2U_SAGE.extend([SAGEConv(self.hidden_size, self.hidden_size, 'mean') for i in range(self.graph_num_way - 1)])
        # SAGE layer for converting user2item vector
        self.U2I_SAGE = nn.ModuleList([SAGEConv(self.user_embedding_size, self.hidden_size, 'mean')])
        self.U2I_SAGE.extend([SAGEConv(self.hidden_size, self.hidden_size, 'mean') for i in range(self.graph_num_way - 1)])
        
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.user_embedding_size, padding_idx=0) # user id [1,2,...]
        self.item_embedding = nn.Embedding(self.n_items, self.item_embedding_size, padding_idx=0) # item id [1,2,...]
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.ui_concat_layer = nn.Linear(self.hidden_size*2, self.hidden_size)

        ## transformer layer modul
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads, \
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,\
                                              hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob,\
                                              attn_dropout_prob=self.attn_dropout_prob, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def adjacency_generation(self, ui_matrix, cutoff):
        cosine_sim = cosine_similarity(ui_matrix, Y=None, dense_output=True)
        norm_sim = (cosine_sim + 1) / 2
        
        norm_sim = (norm_sim > cutoff).astype(int)
        norm_sim = norm_sim-np.diag(np.diag(norm_sim))
        adjacency_list = []
        for i,row in enumerate(norm_sim):
            for j in np.where(row==1)[0]:
                adjacency_list.append([i+1,j+1]) # 0부터 시작하는 넘파이 인덱싱 문제를 피하기 위해 +1, 
        adjacency_list = np.array(adjacency_list).T
        return adjacency_list.tolist()

    def graph_generation(self, adjacency_list, ui_type):
        if ui_type == 'user':
            src_ids = torch.tensor(adjacency_list[0])
            des_ids = torch.tensor(adjacency_list[1])
            graph = dgl.graph((src_ids, des_ids), idtype=torch.int32, num_nodes=self.n_users, device=self.device)
        elif ui_type == 'item':
            src_ids = torch.tensor(adjacency_list[0])
            des_ids = torch.tensor(adjacency_list[1])
            graph = dgl.graph((src_ids, des_ids), idtype=torch.int32, num_nodes=self.n_items, device=self.device)
        return graph

    def seq_interaction_generation(self, dataset):
        ui_matrix = torch.zeros([self.n_users+1, self.n_items+1], dtype=torch.float32)
        for uid in tqdm(dataset[self.USER_ID].unique()):
            user_item = torch.gather(dataset[self.ITEM_ID], 0, torch.where(dataset[self.USER_ID] == uid)[0])
            ui_matrix[uid, user_item] = torch.arange(user_item.size(0), 0 ,-1, dtype=torch.float32)
        return ui_matrix
       
    def forward(self, user_seq, item_seq, item_seq_len):
        self.user_graph.ndata['h0'] = self.user_embedding(torch.LongTensor(range(self.user_graph.num_nodes())).to(self.device))
        self.item_graph.ndata['h0'] = self.item_embedding(torch.LongTensor(range(self.item_graph.num_nodes())).to(self.device))
        for w in range(self.graph_num_way):
            # import pdb;pdb.set_trace()
            self.user_graph.ndata['h'+str(w+1)] = F.leaky_relu(self.U2I_SAGE[w](self.user_graph, self.user_graph.ndata['h'+str(w)]))
            self.item_graph.ndata['h'+str(w+1)] = F.leaky_relu(self.I2U_SAGE[w](self.item_graph, self.item_graph.ndata['h'+str(w)]))
        
        user_emb = self.user_graph.ndata['h'+str(self.graph_num_way)][(user_seq).unsqueeze(1).repeat((1, self.max_seq_length))]
        item_emb = self.item_graph.ndata['h'+str(self.graph_num_way)][item_seq] # (B,T,H)  
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        input_emb = torch.cat([user_emb, item_emb], dim=2)
        input_emb = self.ui_concat_layer(input_emb)
        # import pdb;pdb.set_trace()
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        user_seq = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        seq_output = self.forward(user_seq, item_seq, item_seq_len)
        # import pdb;pdb.set_trace()
        if self.loss_type == 'BPR':
            neg_items= interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_graph.ndata['h'+str(self.graph_num_way)][pos_items]
            neg_items_emb = self.item_graph.ndata['h'+str(self.graph_num_way)][neg_items]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) # [batch_size]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) # [batch_size]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_graph.ndata['h'+str(self.graph_num_way)]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, pos_items)
            return loss

    def predict(self, interaction):
        user_seq = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(user_seq, item_seq, item_seq_len)
        test_item_emb = self.item_graph.ndata['h'+str(self.graph_num_way)][test_item]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user_seq = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(user_seq, item_seq, item_seq_len)
        test_items_emb = self.item_graph.ndata['h'+str(self.graph_num_way)]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # [batch_size, n_items]
        return scores