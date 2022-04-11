from cgi import test
from copy import deepcopy
from re import I

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import SAGEConv
import scipy.sparse as sp

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.layers import TransformerLayer
from tqdm import tqdm

class UIlightGCNSeq(SequentialRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(UIlightGCNSeq, self).__init__(config, dataset)
        ## graph config
        self.graph_embedding_size = config['embedding_size']  # input dim user graph sage
        self.graph_layers = config['graph_layers']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']
        
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
        self.s_attn_first = config['s_attn_first']
        
        # Self-Attention layers (Query: Item, Key(Value): Item)
        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim, padding_idx=0)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.restore_user_e = None
        self.restore_item_e = None
        
        ## transformer layer modul
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        self.reg_loss = EmbLoss()
        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

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

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
   
    def msg_passing(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
   
    def forward(self, user_seq, item_seq, item_seq_len):
        user_all_embeddings, item_all_embeddings = self.msg_passing()
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        user_emb = user_all_embeddings[user_seq]
        item_emb = item_all_embeddings[item_seq]
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output  # [B H]

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user_all_embeddings, item_all_embeddings = self.msg_passing()
        
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
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user_all_embeddings, item_all_embeddings = self.msg_passing()

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