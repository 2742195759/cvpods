# python3 代码
import numpy as np
import torch
import os
import pandas as pd
from torchvision.models import resnet101, inception_v3
# using ClassifierChain
# from skmultilearn.problem_transform import 
# from torch import *
import torch.nn.init as init
from .xk_module.transformer import Encoder as TransformerEncoder

"""
    协议： XXX
    1. model 下面的类，不负责将input转化为相应的 device 上
"""



class DeepMultiTagPredictor(torch.nn.Module):
    """ @ mutable | tunable
        
        the module to use the high_feature to predict the tag.
        and get the loss for error-proporganda
        the module simply model the task as multi binary logistic regression classifiers
    """
    def __init__(self, aesthetic_feature_dim, label_num=14):
        self.aesthetic_feature_dim = aesthetic_feature_dim
        self.label_num = 14
        self.weight_w = torch.nn.Parameter(torch.Tensor(self.label_num, self.aesthetic_feature_dim))
        self.weight_w = torch.nn.init.normal_(self.wei_user, mean=0.0, std=1.0)

    def forward(self, high_feature):
        """
            use the high_feature to predict the label class
            
            @ high_feature: the high level feature wants to predict the tag
            @ return      : numpy([batch_size, labels_num]) #prob [0, 1]
        """
        output = torch.matmul(high_feature, self.weight_w.transpose())
        return torch.nn.Sigmoid(output) # simple


class Cond_LSTM(torch.nn.Module):
    """
        [U diag(Fs) V] . shape = (n_hidden, n_hidden) => n_F 为 
        
    """
    def __init__ (self, n_input, n_hidden, n_F, n_cond_dim):
        super(Cond_LSTM, self).__init__()
        self.wei_U = torch.nn.ParameterList()
        self.wei_V = torch.nn.ParameterList()
        self.wei_WI = torch.nn.ParameterList()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_cond_dim = n_cond_dim
        self.n_F = n_F
        self.wei_F = torch.nn.Parameter(torch.Tensor(n_F, n_cond_dim))
        for i in range(4): # responding to "i f g o" gates respectively
            self.wei_U.append(torch.nn.Parameter(torch.Tensor(n_hidden, n_F)))
            self.wei_V.append (torch.nn.Parameter(torch.Tensor(n_F, n_hidden)))
            self.wei_WI.append ( torch.nn.Parameter(torch.Tensor(n_hidden, n_input)))      # for input weight
        
        self.init_parameters()

    def init_parameters(self): # FIXME(是否有更加好的实现方式)
        self.wei_F     = init.normal_(self.wei_F, mean=0.0, std=1.0)
        for i in range(4):
            self.wei_U[i]  = init.normal_(self.wei_U[i], mean=0.0, std=1.0)
            self.wei_V[i]  = init.normal_(self.wei_V[i], mean=0.0, std=1.0)
            self.wei_WI[i] = init.normal_(self.wei_WI[i], mean=0.0, std=1.0)
    
    def forward(self, tnsr_input, tpl_h0_c0, tnsr_cond): # XXX 每个batch必须要cond相同
        """
        @ tnsr_input : (n_step, n_batch, n_feat_dim)
        @ tpl_h0_c0  : ((n_batch, n_hidden) , (n_batch, n_hidden)) 一个tuple
        @ tnsr_cond  : (n_cond_dim, ), the same for the whole batch
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        tnsr_cond = tnsr_cond.unsqueeze(1)
        wei_WH = [ self.wei_U[i].matmul((self.wei_F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ] # shape = (n_hidden, n_hidden)

        n_batch = tnsr_input.shape[1]
        n_step  = tnsr_input.shape[0]
        c = [ tpl_h0_c0[1] ]  # self.c[i].shape = (n_hidden, n_batch)
        assert (c[0].shape == (self.n_hidden, n_batch))
        h = [ tpl_h0_c0[0] ]  # self.c[i].shape = (n_hidden, n_batch)
        assert (h[0].shape == (self.n_hidden, n_batch))
        
        for t in range(n_step): # TODO add Bias, bi and bh
            it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input[t].t()) + wei_WH[0].matmul(h[t]))  # it.shape = (n_hidden, n_batch)
            ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input[t].t()) + wei_WH[1].matmul(h[t]))  
            gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input[t].t()) + wei_WH[2].matmul(h[t]))
            ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input[t].t()) + wei_WH[3].matmul(h[t]))
            
            c.append (ft * c[t] + it * gt)
            h.append (ot * torch.tanh(c[t+1]))
            
        assert (len(h) == len(c) and len(c) == n_step+1)
        assert (h[0].shape == (self.n_hidden, n_batch)) # 列向量
        return h, c

    # XXX 不要有多个batch，不要梯度
    def eval_start(self, tpl_h0_c0, tnsr_cond):
        """ eval_start 然后每个 eval_step 输出一个output
        @tpl_h0_c0 : (h0, c0)   type(h0|c0) =  torch.tensor ;  h0|c0.shape = (n_hidden,)
        @tnsr_cond : tensor shape=(n_cond)
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        tnsr_cond = tnsr_cond.unsqueeze(1)
        self.wei_WH = [ self.wei_U[i].matmul((self.wei_F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ]
        self._eval_c = tpl_h0_c0[1].unsqueeze(1)  # self.c[i].shape = (n_hidden, 1)
        assert (self._eval_c.shape == (self.n_hidden, 1))
        self._eval_h = tpl_h0_c0[0].unsqueeze(1)  # self.c[i].shape = (n_hidden, n_batch)
        assert (self._eval_h.shape == (self.n_hidden, 1))
        pass

    def eval_step (self, tnsr_input):
        """ eval_start 然后每个 eval_step 输出一个output

        @ tnsr_input : tensor shape=(input_feat_dims)
        """
        tnsr_input = tnsr_input.unsqueeze(1)  # make it bacame matrix
        it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input) + self.wei_WH[0].matmul(self._eval_h))  # it.shape = (n_hidden, 1)
        ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input) + self.wei_WH[1].matmul(self._eval_h))  
        gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input) + self.wei_WH[2].matmul(self._eval_h))
        ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input) + self.wei_WH[3].matmul(self._eval_h))
            
        self._eval_c = ft * self._eval_c + it * gt
        self._eval_h = ot * torch.tanh(self._eval_c)

        return self._eval_h.squeeze(), self._eval_c.squeeze()

class AestheticFeatureLayer(torch.nn.Module): # XXX (独立的 lr)
    def __init__(self):
        super(AestheticFeatureLayer, self).__init__()
        " ====== submodule"
        self.resnet1 = resnet101(pretrained=True)
        #self.resnet2 = resnet101(pretrained=True)
        self.inception = inception_v3(pretrained=True)

        " ====== feature vector "
        self.mlsp = [None] * 11
        self.features = None

        " ====== register hook"
        self.start_register_hook()

    def start_register_hook(self): # FIXME
        def resnet_avgpool(module, i, o):
            nonlocal self
            self.features = o.reshape((o.shape[0], -1))
        
        def register_helper(idx, name):
            def copy (module, i, o):
                nonlocal self
                self.mlsp[idx] = o
            getattr(self.inception, name).register_forward_hook(copy)
            

        self.resnet1._modules.get('avgpool').register_forward_hook(resnet_avgpool)
        reg_list = [
            (0, 'Mixed_5b'),
            (1, 'Mixed_5c'),
            (2, 'Mixed_5d'),
            (3, 'Mixed_6a'),
            (4, 'Mixed_6b'),
            (5, 'Mixed_6c'),
            (6, 'Mixed_6d'),
            (7, 'Mixed_6e'),
            (8, 'Mixed_7a'),
            (9, 'Mixed_7b'),
            (10,'Mixed_7c'),
        ]
        for idx, name in reg_list:
            register_helper(idx, name)

    def first_channel(self, images, patches):
        """
            @image : tnsr(N, 3, W, H)      # 
            @patches:tnsr(N, PN, 3, W, H)  # PN is patch number (pretained) 
            @ret   :
                tnsr(N, 2048), tnsr(N, PN, 2048)
        """
        self.resnet1(images)
        return (self.features, None)

    def MLSP_feature(self, images):
        """
            @image : (N, 3, W, H)
            @ret   : 
                (N, 10084)
        """
        import pdb
        #pdb.set_trace()
        self.inception(images)
        " N C W H  -> N C 1 1  -> N C -> concat"
        result = []
        for item in self.mlsp:
            gap = torch.nn.AvgPool2d((item.shape[2], item.shape[3]))
            result.append(gap(item).squeeze())
        self.mlsp = [None] * 11
        return torch.cat(result, dim=1)

    def get_feature_dim(self):
        return 10048 + 2048

    def forward(self, images, patches=None):
        t1, t2 = self.first_channel(images, patches)
        t3 = self.MLSP_feature    (images)
        return t1, t2, t3

class Word2vecBiLSTM(torch.nn.Module):
    def __init__(self, n_voc, n_word_dim, n_output=300):
        super(Word2vecBiLSTM, self).__init__()
        assert (n_output % 2== 0)
        self.n_voc = n_voc
        self.n_word_dim = n_word_dim
        self.n_hidden = int(n_output/2)
        self.word_emb = torch.nn.Embedding(self.n_voc, self.n_word_dim)
        self.bi_lstm = torch.nn.LSTM(n_word_dim, self.n_hidden, 1, True, False, 0., True)

    def forward(self, sents):
        """
        :sents: .shape=(n_batch, n_cap_len)
        :returns: words, sent_words, sents
            words means the vector of words: (n_batch, n_cap_len, n_word_dim)
            sent_words means the vector of words in lstm: (n_batch, n_cap_len, n_output)
            sents means the vector of sents: (n_batch, n_output)
        """
        #import pdb
        #pdb.set_trace()
        n_batch = sents.shape[0]
        words = self.word_emb(sents)
        lstm_input = words.transpose(0,1)  # (batch, n_cap_len, n_word_emb)
        output, (h_n, c_n) = self.bi_lstm(lstm_input)
        sents = h_n.transpose(0,1).reshape(n_batch, -1)
        sent_words = output.transpose(0, 1)
        return words, sent_words, sents
    
    def get_hidden_size(self):
        return self.n_hidden

class Indentity(torch.nn.Module): #恒等层
    def __init__(self):
        super(Indentity, self).__init__()

    def forward(self, input):
        return input

class SoftmaxPredictor(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SoftmaxPredictor, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.transform = None
        self.softmax = torch.nn.Softmax(dim=-1)
        if self.dim_in == self.dim_out : self.transform = Indentity()
        else                           : self.transform = torch.nn.Linear(dim_in, dim_out)

    def forward(self, input):
        assert (input.shape[1] == self.dim_in)
        return self.softmax(self.transform(input))

class MLP(torch.nn.Module):
    """ MLP method
        list[0] is the input dim
        list[-1] is the output dim
        output is activated
    """
    def __init__(self, dim_list, activate_module, last_activate=True):
        super(MLP, self).__init__()
        self.dim_list = dim_list
        self.dim_list = [ int(num) for num in dim_list ]
        self.activate = activate_module
        self.m_ls = torch.nn.Sequential()
        for i in range(len(dim_list)-1):
            self.m_ls.add_module(str(i), torch.nn.Linear(dim_list[i], dim_list[i+1]))
            if i == len(dim_list) - 2 and not last_activate: 
                continue 
            self.m_ls.add_module('_'+str(i), self.activate())
        
    def forward(self, input):
        """input is the  (batch, ndim), satisifies that ndim == self.dim_list[0]
        """
        assert( input.shape[-1] == self.dim_list[0] )
        output = self.m_ls(input)
        assert (output.shape[-1] == self.dim_list[-1] )
        return output

class BinaryMLP(torch.nn.Module):
    def __init__(self, dim_list, activate_module):
        super(BinaryMLP, self).__init__()
        assert (dim_list[-1] == 1 and len(dim_list) >= 2)
        self.dim_list = dim_list
        self.activate = activate_module
        self.mlp = MLP(dim_list[:-1], activate_module)
        self.pre = torch.nn.Linear(dim_list[-2], dim_list[-1])
        
    def forward(self, input):
        """input is the  (batch, ndim), satisifies that ndim == self.dim_list[0]
        """
        assert( input.shape[-1] == self.dim_list[0] )
        import pdb
        pdb.set_trace()
        output = self.pre(self.mlp(input))
        assert (output.shape[-1] == self.dim_list[-1] )
        return torch.nn.Sigmoid()(output)
    
class AttentionMLP(torch.nn.Module):
    def __init__(self, dim_inputA, dim_inputB, mlp_list, mlp_act):
        """ Attention for the inputA and inputB
            inputA : (K, dk) stands for ( -1, K, dk )
            inputB : (K, dl) stands for ( -1, L, dl )
            output : (K, L ) stands for ( -1, K, L )
                b stand for batch size, inputA is the global feature
                inputB is the vector sets to weighted sum up.
                output is the heatmap. and L dim sum to 1
        """
        super(AttentionMLP, self).__init__()
        self.dk = dim_inputA
        self.dl = dim_inputB
        self.mlp_list = mlp_list 
        self.mlp_act = mlp_act
        self.mlp = MLP(self.mlp_list[:-1], self.mlp_act)
        self.linear = torch.nn.Linear(self.mlp_list[-1], 1, False)
        self.softmax = torch.nn.Softmax(dim=2)
        assert (self.mlp_list[0 ] == self.dk + self.dl)
        
    def forward(self, inputA, inputB, nA, nB):
        """ 
            output = (batch, nA, nB)
        """
        self.K, self.L = nA, nB
        assert( input.shape[-1] == self.dim_list[0] )
        tA = inputA.unsqueeze(2).repeat(1,1,self.L,1)
        tB = inputB.unsqueeze(1).repeat(1,self.K,1,1)
        tC = torch.cat([tA, tB], dim=3)
        output = self.softmax(self.linear(self.m_ls(tC)).reshape(-1,self.K,self.L))
        return output

class BinaryCrossEntropyLoss(torch.nn.Module):
    """ 二分类的交叉熵
        labels = Shape:  1 or -1； 1 pos ， -1 neg
        logits = Shape == labels.Shape

        for every elements: 
            if (labels==1) : output = - ln(sigmoid(x))
            if (lables==-1): output = - ln(signmod(-x))

        return : 
            output : the same shape as losses
        """
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, logits, labels):
        """
        """
        assert ((labels == 1).sum() + (labels == -1).sum() == labels.nelement())
        lables = labels.float()
        logits = logits.float()
        return - self.logsigmoid(logits * labels)

class Word2vecTransformer(torch.nn.Module):
    """ Transformer for sequence input encoder / decoder
        This Module can replace the Word2vecBiLSTM ...
        
        Origin Code from densecap masked transformer
    """
    def __init__(self, n_voc, n_word_dim, n_output=300, n_hidden=150, n_layers=3, n_heads=5, drop_ratio=0.1):
        self.transformer_encoder = TransformerEncoder(n_output, n_hidden, n_layers, n_heads, drop_ratio)
    
    def forward(self): 
        pass

class MaxMarginLoss(torch.nn.Module):#{{{
    def __init__(self, margin=1, hard=False):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        self.hard = hard
        pass
    def forward(self, positive_scores, negative_scores, n_neg=None):
        raw_loss = (self.margin + negative_scores - positive_scores)
        raw_loss = torch.clamp(raw_loss, min=0)
        if self.hard and n_neg:
            raw_loss = raw_loss.reshape([-1, n_neg])
            raw_loss = raw_loss.max(dim=1)[0] * n_neg  # [batch, 1]
        return raw_loss.sum()#}}}
    
class BprLoss(torch.nn.Module):#{{{
    def __init__(self):
        super(BprLoss, self).__init__()
        #self.margin = margin
        #self.hard = hard
        pass

    def forward(self, positive_scores, negative_scores, n_neg=None):
        raw_loss = - torch.log(torch.nn.Sigmoid()(positive_scores - negative_scores))
        #raw_loss = raw_loss.reshape([-1, n_neg])
        #raw_loss = raw_loss.max(dim=1)[0] * n_neg  # [batch, 1]
        return raw_loss.sum()#}}}
