import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn import Linear, ReLU, Sequential, LayerNorm, Sigmoid, Dropout, Module

from code.abstract.abstract_adj_mat_gnn import AbstractAdjMatGNN
from code.utils.torch_utils.xavier_linear import XavierLinear

class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_head = int(args.d_model / args.head)
        self.n_head = args.head
        self.d_model = args.d_model
        
        self.query = nn.Linear(args.d_model, args.d_model, bias = False)
        self.key = nn.Linear(args.d_model, args.d_model, bias = False)
        
        self.attn = nn.Linear(self.d_head * 2, self.n_head, bias = False)
        self.activation = nn.LeakyReLU(negative_slope = args.alpha, inplace = True)
        self.output = nn.Sequential(nn.Linear(args.d_model, args.d_model, bias = False), nn.Tanh())

        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, query, key, adj):
        '''
        m -> multi
        h -> head
        q -> query
        k -> key
        d -> dimension of each head
        n -> number
        len -> length
        mh -> multi-head
        '''
        bsz, qlen, klen = query.shape[0], query.shape[1], key.shape[1]
        head_list = list(range(self.n_head))
        mh_query, mh_key = self.query(query).view(bsz, qlen, self.n_head, self.d_head), self.key(key).view(bsz, klen, self.n_head, self.d_head)
        mh_attn = self.attn( \
                    torch.cat([mh_query[:, :, None, :, :].expand(-1, -1, klen, -1, -1), mh_key[:, None, :, :, :].expand(-1, qlen, -1, -1, -1)], dim = -1)) \
                                                                                                            [:, :, :, head_list, head_list] #(bsz, qlen, klen, n_head)
        mh_attn = self.activation(mh_attn)
        mh_attn = mh_attn.masked_fill((1 - adj)[:, :, :, None].expand_as(mh_attn).bool(), -1e-8)
        mh_attn = F.softmax(mh_attn, dim = -2)
        mh_attn = self.dropout(mh_attn)

        return query + self.output(torch.einsum('bqkn, bknd -> bqnd', mh_attn, key.view(bsz, klen, self.n_head, self.d_head)).reshape(bsz, qlen, self.d_model))

class Gate(nn.Module):
    def __init__(self, args):
        # hidden_dim: the dimension fo hidden vector
        super(Gate , self).__init__()
        self.d_model = args.d_model
        self.weight_proj = nn.Linear(2 * self.d_model, 1)
        self.tanh = nn.Tanh()

    def forward(self, featureA, featureB):
        feature = torch.cat([featureA, featureB], dim = -1)
        att = self.tanh(self.weight_proj(feature)) # (B, N, 1)
        gate_score = F.sigmoid(att)
        gate_score = gate_score.repeat(1, 1, self.d_model ) # (B, N, D)
        return gate_score * featureA + (1 - gate_score) * featureB

class Flow_GAT(nn.Module):
    def __init__(self, args):
        super(Flow_GAT, self).__init__()
        self.args = args
        self.__multihead_attention = nn.ModuleList([nn.ModuleList([MultiheadAttention(args) for _ in range(2)]) for _ in range(args.iteration)])
        # self.gate = nn.Parameter(torch.rand(2))
        # self.gate = Gate(args)

    def forward(self,
                nodes_emb = None, 
                graph_adj = None
                ):
        # graph_adj[:, 64, :]= 0
        # graph_adj[:, :,64] = 0
        #nodes_emb decompose into post_emb, cat_emb and word_emb.
        post_emb, cate_emb, word_emb = nodes_emb[:, :self.args.max_post], nodes_emb[:, self.args.max_post:self.args.max_post + self.args.liwc_num], nodes_emb[:, self.args.max_post + self.args.liwc_num:]
        adj_word2post, adj_cate2word, adj_word2cate, adj_post2word = graph_adj[:, self.args.max_post + self.args.liwc_num:, :self.args.max_post], graph_adj[:, self.args.max_post:self.args.max_post + self.args.liwc_num, self.args.max_post + self.args.liwc_num:], graph_adj[:, self.args.max_post + self.args.liwc_num:, self.args.max_post:self.args.max_post + self.args.liwc_num], graph_adj[:, :self.args.max_post, self.args.max_post + self.args.liwc_num:]
        # post -> word
        word_emb = self.__multihead_attention[0](query = word_emb, key = post_emb, adj = adj_word2post)

        # word -> post
        post_emb_1 = self.__multihead_attention[0](query = post_emb, key = word_emb, adj = adj_post2word)

        # post -> word
        # word_emb_2 = self.__multihead_attention[1](query = word_emb, key = post_emb, adj = adj_word2post)
        
        # word -> cat
        cate_emb = self.__multihead_attention[1](query = cate_emb, key = word_emb, adj = adj_cate2word)

        # cat -> word
        word_emb = self.__multihead_attention[1](query = word_emb, key = cate_emb, adj = adj_word2cate)

        # word -> post
        post_emb_2 = self.__multihead_attention[0](query = post_emb, key = word_emb, adj = adj_post2word)
        
        #gate_weight = F.softmax(self.gate)

        # post_emb = torch.einsum('k, bknd -> bnd', gate_weight, torch.stack((post_emb_1, post_emb_2), dim=1))
        post_emb = (post_emb_1 + post_emb_2) / 2

        new_nodes_emb = torch.cat((post_emb, cate_emb, word_emb), dim=1)

        return new_nodes_emb
        # return (post_emb_1 + post_emb_2) / 2

    def get_in_dim(self):
        return self.args.d_model

    def get_out_dim(self):
        return self.args.d_model    

    def get_latest_vertex_embeddings(self):
        return self.args.d_model        

class PdGNN(AbstractAdjMatGNN):

    def __init__(self, args):
        super(PdGNN, self).__init__() 

        self.n_layers = args.iteration
        self.n_relations = 1

        gnn_layers = []
        for layer in range(self.n_layers):
            gnn_layers.append(Flow_GAT(args))
                
    def is_adj_mat(self):
        return True
      
    def get_initial_layer_input(self, vertex_embeddings, mask):
        return vertex_embeddings * mask

    def process_layer(self, vertex_embeddings,
                      adj_mat,
                      gnn_layer,
                      message_scale,
                      message_replacement,
                      mask):

        return gnn_layer(vertex_embeddings, adj_mat)