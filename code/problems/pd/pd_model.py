import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from code.gnns.flow_gat import Flow_GAT

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.bert_config = AutoConfig.from_pretrained(args.bert_path)
        self.bert_config.output_hidden_states = True #question
        self.bert = AutoModel.from_pretrained(args.bert_path, config = self.bert_config)
        
    def forward(self,
            posts_ids = None,
            attn_mask = None, 
            wc_ids = None
            ):
        '''
        '''
        pad, d_model = self.args.pad, self.args.d_model
        bsz, max_node, max_idx_num = wc_ids.shape
        max_post, max_len = self.args.max_post, self.args.max_len

        #Semantic-rep of the last layer
        posts_ids = posts_ids.view(bsz * max_post, max_len)
        attn_mask = attn_mask.view(bsz * max_post, max_len)
        encoder_outputs = self.bert(input_ids = posts_ids, attention_mask = attn_mask)
        last_sematic_rep = encoder_outputs[1].reshape(bsz, max_post, d_model)

        #Wembedding of words and cats(wc)
        wc_ids = wc_ids.reshape(-1, max_idx_num)
        wc_num = (wc_ids != pad).float().sum(dim = -1)[:, None].expand(-1, d_model) #(bsz * max_node, 768)
        wc_emb = self.bert.embeddings.word_embeddings(wc_ids) #(bsz * max_node, max_idx_num, 768)
        wc_emb = wc_emb.masked_fill((wc_ids == pad)[:, :, None].expand_as(wc_emb), 0).sum(-2) / (wc_num + 1e-8)
        wc_emb = wc_emb.reshape(bsz, max_node, d_model) #(bsz, max_node, 768)

        #Creating node representation of heterogeneous graph
        layers_nodes_emb = []
        for bert_layer_output in encoder_outputs[2]:
            layers_nodes_emb.append(torch.cat([bert_layer_output[:, 0].reshape(bsz, max_post, d_model), wc_emb], dim = -2))

        return last_sematic_rep, layers_nodes_emb

class TrigNet(nn.Module):
    def __init__(self, args):
        super(TrigNet, self).__init__()
        self.args = args
        self.ptm_encoder = BERT(args)
        self.gat_encoder = Flow_GAT(args)
        self.liwc_embeddings = nn.Embedding(self.args.liwc_num, args.d_model)
        self.layer_param = nn.Parameter(torch.rand(3))

        self.classifiers = nn.ModuleList([nn.Linear(args.d_model, 2) for _ in range(args.num_labels)])
        self.dropout = nn.Dropout(args.dropout)

    def forward(self,
            posts_ids = None,
            posts_mask = None,
            attn_mask = None,
            graph_ids = None,
            graph_adj = None,
            labels = None,
            ):

        batch_size = posts_ids.shape[0]
        device = posts_ids.device

        # Bert-encoder
        last_sematic_rep, layers_nodes_emb = self.ptm_encoder(posts_ids, attn_mask, wc_ids = graph_ids)
        
        # Get 12 layers cls for post node embedding
        layers_nodes_emb = torch.stack(layers_nodes_emb, dim = 1) #{format,layers_nodes_emb} (list*tensor)->(tensor) （bsz, bert_layers, max_post, hid_size)
        layers_cls_emb = layers_nodes_emb[:, 1:, :self.args.max_post, :] # (bsz, 12, self.args.max_post, hidden) #{format, layers_nodes_emb} layers_cls_emb为12层post节点表示

        # Elmo_style post fusion
        layer_weight = F.softmax(self.layer_param, dim=-1)
        post_emb = torch.einsum('k, bknd -> bnd', layer_weight, layers_cls_emb[:, -3:, :, :]) #{operation, layers_cls_emb} 混合最后3层post节点表示(bsz, max_post, hid_size)
        # post_emb = layers_cls_emb[:, -1, :, :].squeeze(1)

        # LIWC_emb different learning rates
        liwc_emb = self.liwc_embeddings(torch.Tensor(list(range(self.args.liwc_num))).to(device).long()[None, :].expand(batch_size, -1))
        word_emb = layers_nodes_emb[:, 0, self.args.max_post + self.args.liwc_num:, :]
        nodes_emb = torch.cat([post_emb, liwc_emb, word_emb], dim = -2) #{format, nodes_emb} (bsz, max_node, hid_size)
        new_nodes_emb = self.gat_encoder(nodes_emb, graph_adj)    # (bsz, self.args.max_post, hidden)
        last_psychology_rep = new_nodes_emb[:, :self.args.max_post]

        # mean
        last_psychology_rep = last_psychology_rep.masked_fill((1 - posts_mask)[:, :, None].expand_as(last_psychology_rep).bool(), 0).sum(dim = -2) / (posts_mask.sum(dim = -1)[:, None].expand(-1, 768) + 1e-8)
        #bert_rep = post_emb.masked_fill((1 - posts_mask)[:, :, None].expand_as(post_emb).bool(), 0).sum(dim = -2) / (posts_mask.sum(dim = -1)[:, None].expand(-1, 768) + 1e-8)


        # Only psychology for output
        final_rep = last_psychology_rep

        #classification
        logits = torch.stack([classifier(final_rep) for classifier in self.classifiers], dim = 1) # b, 4, 2 

        #output
        outputs = {}
        outputs['loss'] = F.cross_entropy(logits.view(-1, 2), labels.view(-1), reduction = 'sum')
        outputs['pred'] = torch.argmax(logits.view(-1, 2), dim = -1).view(-1, 4)
        outputs['acc']  = (outputs['pred'] == labels).float().sum()
        #outputs['gate'] = E

    def get_gnn(self):
        return self.gat_encoder
