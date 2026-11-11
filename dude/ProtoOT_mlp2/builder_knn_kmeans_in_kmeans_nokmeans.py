import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F

class UCDIR(nn.Module):

    def __init__(self, base_encoder, dim=768, K_A=65536, K_B=65536,
                 m=0.999, T=0.2,epsilon=0.05,sink_iters=3):

        super(UCDIR, self).__init__()

        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T
        self.epsilon = epsilon
        self.sink_iters = sink_iters

        norm_layer = partial(SplitBatchNorm, num_splits=2)

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim,norm_layer=norm_layer)

      
        dim_mlp = self.encoder_q.fc.weight.shape[1]

        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queues
        self.register_buffer("queue_A", torch.randn(dim, K_A))
        self.queue_A = F.normalize(self.queue_A, dim=0)
        self.register_buffer("queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_B", torch.randn(dim, K_B))
        self.queue_B = F.normalize(self.queue_B, dim=0)
        self.register_buffer("queue_B_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("q_queue_A", torch.randn(dim, K_A))
        self.q_queue_A = F.normalize(self.q_queue_A, dim=0)
        self.register_buffer("q_queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("q_queue_B", torch.randn(dim, K_B))
        self.q_queue_B = F.normalize(self.q_queue_B, dim=0)
        self.register_buffer("q_queue_B_ptr", torch.zeros(1, dtype=torch.long))
        

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids,q_keys, q_key_ids, domain_id):

        if domain_id == 'A':
            self.queue_A.index_copy_(1, key_ids, keys.T)
        elif domain_id == 'B':
            self.queue_B.index_copy_(1, key_ids, keys.T)
        
        if domain_id == 'A':
            self.q_queue_A.index_copy_(1, q_key_ids, q_keys.T)
        elif domain_id == 'B':
            self.q_queue_B.index_copy_(1, q_key_ids, q_keys.T)



    @torch.no_grad()
    def _batch_shuffle_singlegpu(self, x):

        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):

        return x[idx_unshuffle]

    def forward(self, im_q_A=None, im_k_A=None, im_id_A=None,im_q_B=None,
                im_k_B=None, im_id_B=None, cates_A=None, cates_B=None,
                is_eval=False,criterion=None,cluster_result=None,
                conf_nearest_atob=None,conf_nearest_atoa=None,conf_nearest_btob=None,
                flag=None):

        im_q = torch.cat([im_q_A, im_q_B], dim=0)

        if is_eval:
            k = self.encoder_k(im_q)
            k = F.normalize(k, dim=1)

            k_A, k_B = torch.split(k, im_q_A.shape[0])
            return k_A, k_B


        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        q_A, q_B = torch.split(q, im_q_A.shape[0])

        im_k = torch.cat([im_k_A, im_k_B], dim=0)
        


        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
            
            k = self.encoder_k(im_k) 

            k = F.normalize(k, dim=1)

            k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)

            k_A, k_B = torch.split(k, im_k_A.shape[0])
            
        

        self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, q_A.detach(), im_id_A, 'A')
        self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, q_B.detach(), im_id_B, 'B')

       
        loss_inst = self.instance_contrastive_loss(q_A, k_A, im_id_A,q_B, k_B, im_id_B,criterion)

        loss_in_domain_near  = self.in_domain(q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result,criterion,conf_nearest_atoa,conf_nearest_btob)


        loss_cross_domain_proto = self.cross_domain(q_A,im_id_A,q_B,im_id_B,cluster_result,criterion,conf_nearest_atob)


        return   loss_inst, loss_in_domain_near, loss_cross_domain_proto
    


    def instance_contrastive_loss(self,
                                  q_A, k_A, im_id_A,
                                  q_B, k_B, im_id_B,
                                  criterion):

        l_pos_A = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_B = torch.einsum('nc,nc->n', [q_B, k_B]).unsqueeze(-1)

        l_all_A = torch.matmul(q_A, self.queue_A.clone().detach())
        l_all_B = torch.matmul(q_B, self.queue_B.clone().detach())

        mask_A = torch.arange(self.queue_A.shape[1]).cuda() != im_id_A[:, None]
        l_neg_A = torch.masked_select(l_all_A, mask_A).reshape(q_A.shape[0], -1)

        mask_B = torch.arange(self.queue_B.shape[1]).cuda() != im_id_B[:, None]
        l_neg_B = torch.masked_select(l_all_B, mask_B).reshape(q_B.shape[0], -1)

        logits_A = torch.cat([l_pos_A, l_neg_A], dim=1)
        logits_B = torch.cat([l_pos_B, l_neg_B], dim=1)

        logits_A /= self.T
        logits_B /= self.T

        labels_A = torch.zeros(logits_A.shape[0], dtype=torch.long).cuda()
        labels_B = torch.zeros(logits_B.shape[0], dtype=torch.long).cuda()

        loss_A = criterion(logits_A, labels_A)
        loss_B = criterion(logits_B, labels_B)

        return loss_A + loss_B


        
    def in_domain(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result,criterion,conf_nearest_atoa,conf_nearest_btob):

        all_losses = dict()

        
        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                k_feat = k_A
                queue = self.q_queue_A.clone().detach().T
               
                conf_nearest = conf_nearest_atoa.cuda()
        
            else:
                im_id = im_id_B
                q_feat = q_B
                k_feat = k_B
                queue = self.q_queue_B.clone().detach().T
               
                conf_nearest = conf_nearest_btob.cuda()

            conf_nearest_batch = torch.zeros((q_feat.shape[0],queue.shape[0])).cuda()
            for i,i_id in enumerate(im_id):
                conf_nearest_batch[i] = conf_nearest[i_id]
        
            all_score = torch.div(torch.matmul(q_feat, queue.T), self.T)
            exp_all_score = torch.exp(all_score)
            l_all = all_score - torch.log(exp_all_score.sum(1, keepdim=True))
            mean_l_pos = (conf_nearest_batch * l_all).sum(1) / (conf_nearest_batch.sum(1) + 1e-8)
            loss = - mean_l_pos.mean()
            all_losses['domain_' + domain_id + '_near'] = loss
            
            
        return all_losses['domain_A_near'] + all_losses['domain_B_near']
    

    
    
      
   
    
    def cross_domain(self,q_A,im_id_A,q_B,im_id_B,cluster_result,criterion,conf_nearest_atob):

        all_losses = dict()

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                queue = self.q_queue_A.clone().detach().T
                queue_cross = self.q_queue_B.clone().detach().T
                
                conf_nearest = conf_nearest_atob.cuda()

            else:
                im_id = im_id_B
                q_feat = q_B
                queue = self.q_queue_B.clone().detach().T
                queue_cross = self.q_queue_A.clone().detach().T
                
                conf_nearest = conf_nearest_atob.T.cuda()

            
            conf_nearest_batch = torch.zeros((q_feat.shape[0],queue_cross.shape[0])).cuda()
            for i,i_id in enumerate(im_id):
                conf_nearest_batch[i] = conf_nearest[i_id]
        
            all_score = torch.div(torch.matmul(q_feat, queue_cross.T), self.T)
            exp_all_score = torch.exp(all_score)
            l_all = all_score - torch.log(exp_all_score.sum(1, keepdim=True))
            mean_l_pos = (conf_nearest_batch * l_all).sum(1) / (conf_nearest_batch.sum(1) + 1e-8)
            loss = - mean_l_pos.mean()
            all_losses['domain_' + domain_id + '_proto'] = loss

        return all_losses['domain_A_proto'] + all_losses['domain_B_proto']
    



# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
