import os
import torch
import torch.nn as nn
# from torch.nn.modules.transformer import *
from torch.nn.modules.linear import *
import numpy as np
from collections import OrderedDict
# from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
gpu = torch.cuda.is_available()
from transformers import AutoModel


class G(nn.Module):
    def __init__(self, ptm_path):
        super().__init__()
        self.ptm = AutoModel.from_pretrained(ptm_path)

    def forward(self, shape):
        x = torch.randn(shape).to(self.ptm.device)
        x = self.ptm(inputs_embeds=x)[0]
        return x

class D(nn.Module):
    def __init__(self, ptm_path):
        super().__init__()
        self.ptm = AutoModel.from_pretrained(ptm_path)
        dim = self.ptm.embeddings.word_embeddings.weight.shape[-1]
        self.cls = nn.Linear(dim, 2)
    def forward(self, x):
        x = self.ptm(inputs_embeds=x)[0]
        ret = self.cls(x)
        return ret

class GelU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GelUNew(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LinearModel(nn.Module):
    """
    mlp models
    hid_dims : list, represent layers and hid sizes
    """

    def __init__(self, dims, dropout):
        super().__init__()
        self.dims = dims
        pre_size = dims[0]
        self.layers = []
        self.gelu = torch.nn.LeakyReLU(0.01)
        self.dropout = dropout
        for i, hsize in enumerate(dims):
            if i == 0:
                pre_size = hsize
                continue
            layer_name = 'layer%d' % i
            self.__setattr__(layer_name, nn.Linear(pre_size, hsize))
            self.layers.append((self.__getattr__(layer_name),
                                i == len(dims)-1 and len(dims) > 2))
            pre_size = hsize

    def forward(self, x):
        for layer, do_dropout in self.layers:
            if do_dropout:
                x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x)
            x = gelu(x)
        return x


class CNNModel2d(nn.Module):
    '''
    slot classfier
    return predicted slots' indexes
    https://github.com/xiayandi/Pytorch_text_classification/blob/master/cnn.py
    '''

    def __init__(self, num_filters=128, filter_sizes=None, emb_dim=100, hid_dim=[256], dropout_switches=[True], num_classes=35, channels=1):
        super().__init__()
        if filter_sizes is None:
            self.filter_sizes = [1, 2, 3, 4]
        self.num_filters = num_filters
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.hid_sizes = hid_dim
        self.encoders = nn.ModuleList()
        self.dropout_switches = dropout_switches
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=channels,
                                       out_channels=self.num_filters,
                                       kernel_size=(filter_size, self.emb_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))
        self.hid_layers = nn.ModuleList()
        ins = len(self.filter_sizes) * self.num_filters
        self.linear_trans = nn.Sequential(
            nn.Linear(ins, self.hid_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hid_sizes[-1], self.hid_sizes[-1])
            # nn.Linear(self.hid_sizes[-1], self.hid_sizes[-1]),
            # nn.Dropout(0.1)
        )
        # self.norm = LayerNorm(self.hid_sizes[-1])
        # text cnn
        # self.gen_encoding_hid = nn.Linear(ins, self.hid_sizes[0])
        # self.gen_encoding = nn.Linear(self.hid_sizes[0], self.hid_sizes[0])
        # for i, hid_size in enumerate(self.hid_sizes):
        #     hid_attr_name = "hid_layer_%d" % i
        #     self.__setattr__(hid_attr_name, nn.Linear(ins, hid_size))
        #     self.hid_layers.append(self.__getattr__(hid_attr_name))
        #     ins = hid_size
        # self.out = nn.Linear(ins, self.num_classes)

    def forward(self, x):
        """
        :param x:
                input x is in size of [N, C, H, W]
                N: batch size
                C: number of channel, in text case, this is 1
                H: height, in text case, this is the length of the text
                W: width, in text case, this is the dimension of the embedding
        :return:
                a tensor [N, L], where L is the number of classes
        """
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        # lookup table output size [N, H, W=emb_dim]
        # x = self.lookup_table(x)
        # expand x to [N, 1, H, W=emb_dim]
        # x = x.unsqueeze(c_idx)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = torch.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            k_w = 1
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, k_w))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        # each of enc_outs size [N, C]
        # len(self.filter_sizes) * self.num_filters  4*300
        encoding = torch.cat(enc_outs, 1)
        ret = self.linear_trans(encoding)
        return ret
        # text cnn 
        # encoding = self.bn(encoding)
        # hid_in = encoding
        # for hid_layer, do_dropout in zip(self.hid_layers, self.dropout_switches):
        #     hid_out = hid_layer(hid_in)
        #     # hid_out = lyn(hid_out)
        #     hid_out = torch.relu(hid_out)
        #     if do_dropout:
        #         hid_out = F.dropout(hid_out, training=self.training)
        #     hid_in = hid_out
        # # use as an encoder
        # return hid_in
        # pred_prob = torch.sigmoid(self.out(hid_in))
        # return pred_prob

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class CNNModel3d(nn.Module):
    '''
    slot classfier 
    conv on uttrs^T*uttrs(2-gram), embeddings are the sum of tach pair of words
    return predicted slots' indexes
    https://github.com/xiayandi/Pytorch_text_classification/blob/master/cnn.py
    '''

    def __init__(self, num_filters=100, filter_sizes=None, emb_dim=100, hid_dim=[256], dropout_switches=[True], num_classes=35):
        super().__init__()
        if filter_sizes is None:
            self.filter_sizes = [1, 2, 4, 8, 16]
        self.num_filters = num_filters
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.hid_sizes = hid_dim
        self.encoders = nn.ModuleList()
        self.dropout_switches = dropout_switches
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv3d(in_channels=1,
                                       out_channels=self.num_filters,
                                       kernel_size=(filter_size, filter_size, self.emb_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))
        self.hid_layers = nn.ModuleList()
        ins = len(self.filter_sizes) * self.num_filters
        self.gen_encoding_hid = nn.Linear(ins, self.hid_sizes[0])
        self.gen_encoding = nn.Linear(self.hid_sizes[0], self.hid_sizes[0])
        for i, hid_size in enumerate(self.hid_sizes):
            hid_attr_name = "hid_layer_%d" % i
            self.__setattr__(hid_attr_name, nn.Linear(ins, hid_size))
            self.hid_layers.append(self.__getattr__(hid_attr_name))
            ins = hid_size
        self.out = nn.Linear(ins, self.num_classes)

    def forward(self, x):
        """
        :param x:
                input x is in size of [N, C, H, W]
                N: batch size
                C: number of channel, in text case, this is 1
                H: height, in text case, this is the length of the text
                W: width, in text case, this is the dimension of the embedding
        :return:
                a tensor [N, L], where L is the number of classes
        """
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        d_idx = 4
        # lookup table output size [N, H, W=emb_dim]
        # x = self.lookup_table(x)
        # expand x to [N, 1, H, W=emb_dim]
        # x = x.unsqueeze(c_idx)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = gelu(encoder(x))
            enc_ = F.max_pool3d(enc_, kernel_size=(
                enc_.size()[h_idx], enc_.size()[w_idx], enc_.size()[d_idx]))
            enc_ = enc_.squeeze(d_idx)
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        # each of enc_outs size [N, C]
        # len(self.filter_sizes) * self.num_filters  4*300
        encoding = torch.cat(enc_outs, 1)
        # gen vec for slot encoder to calc attention
        hid_in = encoding
        for hid_layer, do_dropout in zip(self.hid_layers, self.dropout_switches):
            hid_out = gelu(hid_layer(hid_in))
            if do_dropout:
                hid_out = F.dropout(hid_out, training=self.training)
            hid_in = hid_out
        pred_prob = torch.sigmoid(self.out(hid_in))
        return pred_prob

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class RNNModel(nn.Module):
    '''
    uttrs encoder
    return encoder_outputs for each word
    '''
    def __init__(self, emb_dim, hid_dim, dropout=0.0, bid=True):
        super().__init__()
        self.dropout = dropout
        self.rnn = nn.RNN(emb_dim, hid_dim, bidirectional=bid,
                          batch_first=True, dropout=dropout)
        self.hid_dim = hid_dim

    def forward(self, inputs, lens):

        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(packed)
        recovered, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, padding_value=0., total_length=inputs.shape[1])
        hidden = hidden[0] + hidden[1]
        recovered = recovered[:, :, :self.hid_dim] + \
            recovered[:, :, self.hid_dim:]
        return recovered, hidden


class Generator(nn.Module):
    def __init__(self, config, embedding, slot_emb):
        super(Generator, self).__init__()
        self.dropout = config['dropout']
        self.hid_dim = config['rnn_hid']
        self.d_model = config['d_model']
        self.dropout_layer = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.d_model, self.hid_dim,
                          dropout=0, batch_first=True)
        self.W_ratio = nn.Linear(2 * self.hid_dim + self.d_model, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.embedding = embedding
        self.slot_emb = slot_emb
        self.vocab_size = config['vocab_size']
        self.batch_size = config['bsz']
        self.max_values_len = config['max_values_len']
        self.alias = nn.Linear(self.hid_dim, self.d_model)

    def forward(self, encoded_hidden, encoded_outputs, encoded_lens, uttrs, target_batches, use_teacher_forcing, slot):
        self.batch_size = encoded_outputs.shape[0]
        all_point_outputs = torch.zeros(
            self.batch_size, self.max_values_len, self.vocab_size)

        # Compute pointer output
        words_point_out = []
        hidden = encoded_hidden
        # print(hidden.shape)
        words = []
        curr_slot_emb = self.slot_emb(slot)
        decoder_input = self.dropout_layer(
            curr_slot_emb).expand(self.batch_size, self.d_model)
        for wi in range(self.max_values_len):
            dec_state, hidden = self.gru(
                decoder_input.unsqueeze(1).expand(hidden.shape[1], 1, -1), hidden)
            context_vec, logits, prob = self.attend(
                encoded_outputs, hidden.squeeze(0), encoded_lens)
            # p_vocab = self.attend_vocab(
            #     self.embedding.weight, self.alias(hidden.squeeze(0)))
            # print(p_vocab.shape)
            # p_gen_vec = torch.cat(
            #     [dec_state.squeeze(1), context_vec, decoder_input], -1)
            # vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(
                self.batch_size, self.vocab_size)
            p_context_ptr.scatter_add_(1, uttrs, prob)
            # final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
            #     vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
            pred_word = uttrs[torch.argmax(prob, dim=1)]
            words.append([w_idx.item() for w_idx in pred_word])
            all_point_outputs[:, wi, :] = p_context_ptr
            if use_teacher_forcing:
                decoder_input = self.embedding(
                    target_batches[:, wi])
            else:
                decoder_input = self.embedding(pred_word)
            decoder_input
        words_point_out.append(words)
        return all_point_outputs, words_point_out

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, emb_weight, dropout=0.1, finetune=False):
        super().__init__(*args)
        self.dropout = dropout
        new = self.weight.data.new
        self.weight.data.copy_(new(emb_weight))
        self.ft = finetune

    def forward(self, *args):
        out = super().forward(*args)
        if not self.ft:
            out.detach_()
        out = torch.layer_norm(out, (out.shape[-2], out.shape[-1]))
        return F.dropout(out, self.dropout, self.training)


# class ELMoEmbedding(nn.Module):
#     def __init__(self, ELMoModel):
#         self.emb = ELMoModel

#     def forward(self, text):
#         chars_ids = batch_to_ids(text)
#         encoded = torch.cuda.FloatTensor(
#             self.emb(chars_ids)['elmo_representations'][0])
#         return encoded


def masked_cross_entropy_for_value(logits, target, mask):
    # logits: b * |s| * m * |v|
    # target: b * |s| * m
    # mask:   b * |s|
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    log_probs_flat = torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s| * m
    loss = masking(losses, mask)
    return loss


def masking(losses, mask):
    mask_ = []
    batch_size = mask.size(0)
    max_len = losses.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if mask[:, si].is_cuda:
            seq_range_expand = seq_range_expand
        seq_length_expand = mask[:, si].unsqueeze(
            1).expand_as(seq_range_expand)
        mask_.append((seq_range_expand < seq_length_expand))
    mask_ = torch.stack(mask_)
    mask_ = mask_.transpose(0, 1)
    if losses.is_cuda:
        mask_ = mask_
    losses = losses * mask_.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class Projection(nn.Module):
    """
    inputs :
    cond : uttrs_emb
    intent: pre_intent in intents_cls and active_intent in slots cls

    This module is to model P(f(seq) | P(cond|intent))
    where f(deq) output a logits over classes
    """

    def __init__(self, hid_dim, num_classes):
        super().__init__()
        self.trans1_intent = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(hid_dim, hid_dim))
            # ('bn1', nn.BatchNorm1d(hid_dim))
        ]))
        self.trans2_intent = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(hid_dim * 2, hid_dim))
            # ('bn1', nn.BatchNorm1d(hid_dim))
        ]))
        self.trans_cat = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(hid_dim * 2, hid_dim))
            # ('bn1', nn.BatchNorm1d(hid_dim)) # 只有这个的时候比不加好
        ]))
        self.output = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(hid_dim, num_classes))
        ]))
        self.num_classes = num_classes

    def forward(self, seq, cond, intent, use_res=False, use_intent=False):

        batch_size = seq.shape[0]
        # use pre_intent
        # encode pre_intent
        if use_intent:
            res = intent
            intent = self.trans1_intent(intent)
            if use_res:
                intent += res
            intent = gelu(intent)
            # combine the uttrs
            cond_uttrs = torch.cat((intent, cond), 1)
            cond_uttrs = self.trans2_intent(cond_uttrs)
            res = cond  # res is uttrs
            if use_res:
                cond_uttrs += res
            cond_uttrs = gelu(cond_uttrs)
        else:
            # do not use the pre_intent
            cond_uttrs = self.trans1_intent(cond)
            cond_uttrs = gelu(cond_uttrs)
        # res is combined uttrs and pre_intent
        res = cond_uttrs.unsqueeze(1).expand_as(seq)
        tmp = torch.cat((res, seq), 2)
        tmp = tmp.view(-1, seq.shape[-1] * 2)
        tmp = self.trans_cat(tmp)
        if use_res:
            res = res.contiguous().view(-1, seq.shape[-1])
            tmp += res
        tmp = gelu(tmp)
        return self.output(tmp).view(batch_size, -1, self.num_classes)

class ContextAttn(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            # nn.SELU(),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Linear(hid_dim, 1)
        )
    def forward(self, seq, mask=None):
        scores = self.score(seq).squeeze(-1)
        if mask is not None:
            scores += mask
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context

class Attn(nn.Module):
    '''
    calc attention 
    '''

    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        
        self.trans = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim)
        )
    def forward(self, seq, cond, mask=None, dot=False):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        if dot:
            scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
            transed_seq = seq
        else:
            transed_cond = self.trans(cond)
            transed_seq = seq
            scores_ = transed_cond.unsqueeze(1).expand_as(transed_seq).mul(transed_seq).sum(2)
        # scores_ = (scores_ - scores_.min()) /(scores_.max() - scores_.min())
        if not mask is None:
            scores_ += mask
        scores = scores_
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(transed_seq).mul(seq).sum(1)
        return context, scores_, scores


class CustomedBCELoss(nn.Module):
    def __init__(self, weights=None, ones_weights=1.5):
        super().__init__()
        self.weights = weights
        self.ones_weights = ones_weights
        if not weights is None:
            self.weights = torch.softmax(torch.cuda.FloatTensor(weights), 0)
        self.eps = 1e-7

    def forward(self, pred, label):
        '''
        用于多标签分类，标签是一个01串的情况下，一定程度上解决标签稀疏的问题
        - (y*torch.log(a) + 0.5 * (1-y) * torch.log(1-a)).sum().mean()
        '''
        ones_weights = self.ones_weights
        if not self.weights is None:
            para = self.weights.expand_as(label)
            loss = (-ones_weights * label * para * torch.log(pred + self.eps) -
                    (1 - label) * para * torch.log(1-pred + self.eps)).mean()
        else:
            loss = (-ones_weights * label * torch.log(pred + self.eps) -
                    (1 - label) * torch.log(1-pred + self.eps)).mean()
        return loss


class CustomCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomCELoss, self).__init__()
        self.eps = 1e-16
        self.reduction = reduction
    # weights 是样本权重 一维batch size的张量

    def forward(self, outputs, targets, weights):
        # transform targets to one-hot vector
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)
        # loss = -(targets_onehot.float() * torch.log(outputs + self.eps) +\
        #      (1 - targets_onehot) * torch.log(1 - outputs + self.eps)).sum(1)
        loss = (-(targets_onehot.float() * torch.log(outputs + self.eps))).sum(1)
        # print('loss : {} outputs {}'.format(loss, torch.mean(outputs)))
        if self.reduction == 'mean':
            return torch.mean(weights * loss)
        elif self.reduction == 'sum':
            return torch.sum(weights * loss)
        else:
            return weights * loss

class WeightedCELoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1, weight=None):
        super(WeightedCELoss, self).__init__()
        self.reduction = reduction
        self.loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=weight)
    def forward(self, outputs, labels, sample_weights=None):

        loss = self.loss_func(outputs, labels)
        if sample_weights is not None:
            loss *= sample_weights
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss
def word_dropping(obj, word_drop, pad=3):
    word_drop = 1 - word_drop
    mask = Bernoulli(word_drop).sample(obj.shape).long().to(obj.device)
    obj = obj.mul(mask)
    obj.masked_fill_(obj == 0, pad)
    return obj
class FilterLoss(nn.Module):
    def __init__(self, beta1=0.1, beta2 = 0.5, k=2, reduction='mean'):
        """
        对于DST中的触发类别，使用此loss实现课程学习的效果，优先学习置信度（类别softmax之后的概率）高于beta的
        """
        super(FilterLoss, self).__init__()
        self.beta1 = beta1 # 置信度阈值
        self.beta2 = beta2 # 置信度阈值
        self.k = k # 阈值之上的样本是阈值之下的样本的权重倍数
        self.reduction = reduction
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
    def forward(self, preds, labels):
        '''
        preds: B* C
        labels: B
        '''
        preds_softmax = torch.softmax(preds, -1)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)).view(-1)
        # 置信度过滤
        prob_gate = (preds_softmax > self.beta1).long()
        gate = prob_gate.mul((preds_softmax < self.beta2).long())
        err_gate = (preds_softmax < self.beta1).long()
        other = 1 - gate - err_gate
        gate = gate * self.k + err_gate * 1.0/self.k + other
        loss = self.loss_func(preds, labels)
        loss = torch.mul(gate.float(), loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
class SoftFilterLoss(nn.Module):
    def __init__(self, beta=0.1, k=0.05, reduction='mean', ignore_index=-1):
        """
        对于DST中的值的分类，样本的权重是置信度的倒数，最大阈值为k
        """
        super(SoftFilterLoss, self).__init__()
        self.k = k
        self.beta = beta
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def forward(self, preds, labels):
        '''
        preds: B* C
        labels: B
        '''
        preds_softmax = torch.softmax(preds, -1)
        tmp_labels = labels + (labels == self.ignore_index).long()
        preds_softmax = preds_softmax.gather(1,tmp_labels.view(-1,1)).view(-1)
        prob_gate = (preds_softmax < self.beta).long()
        gate = torch.clamp(1 / (preds_softmax + self.k), min=1)
        gate = gate * (1-prob_gate) + prob_gate
        loss = self.loss_func(preds, labels)
        loss = torch.mul(gate.float(), loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, reduction='mean'):
        """
        https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        """
        super(FocalLoss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        # preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossenpty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
class FocalLossForValueExtraction(nn.Module):
    """
    Focal Loss for value extraction in dialogue state tracking, 
    in which number of calsses are dynamic,
    this also support sample weight like WeightedFocalLoss
    """
    def __init__(self, alpha = 0.25, gamma=2, reduction: str ='mean'):
        super(FocalLossForValueExtraction, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
    def forward(self, outputs, labels, sample_weights=None):
        alphas = torch.ones(outputs.shape[-1]).to(outputs.device) * (1-self.alpha)
        alphas[0] = self.alpha # AT_STR 都放在了第一个位置上
        preds_logsoft = F.log_softmax(outputs, dim=1)
        preds_softmax = torch.exp(preds_logsoft)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alphas = alphas.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alphas, loss.t())
        if sample_weights is not None:
            loss *= sample_weights
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction: str = 'mean'):
        super(WeightedFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_func = FocalLoss(reduction='none', alpha=alpha, gamma=gamma)
    def forward(self, outputs, labels, sample_weights=None):

        loss = self.loss_func(outputs, labels).squeeze()
        # print(loss.shape, sample_weights.shape)
        if sample_weights is not None:
            loss *= sample_weights
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss
def word_dropping(obj, word_drop, pad=3):
    word_drop = 1 - word_drop
    mask = Bernoulli(word_drop).sample(obj.shape).long().to(obj.device)
    obj = obj.mul(mask)
    obj.masked_fill_(obj == 0, pad)
    return obj


class MultiTaskLoss(torch.nn.Module):
    '''https://arxiv.org/abs/1705.07115'''

    def __init__(self, is_regression, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.ones(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression+1)*(stds**2))
        multi_task_losses = coeffs*losses + torch.log(stds)

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses
class CopyNet(nn.Module):
    """
    Inputs:
        inputs: token level representation of a sequence
        decoder: rnn
        input_features: first input for decoder
        max_output_len: steps the decoder should run
    Outputs:
        prob of each decode step, and its' context(softmax the prob then sum the inputs)
    """
    def __init__(self, inp_feature_dim, decoder_output_dim, max_output_len):
        super().__init__()
        self.inp_feature_dim = inp_feature_dim
        self.decoder_output_dim = decoder_output_dim
        # self.decoder = nn.GRU(inp_feature_dim, decoder_output_dim, 1, batch_first=True)
        self.decoder = nn.GRUCell(inp_feature_dim, decoder_output_dim)
        self.W1 = nn.Linear(decoder_output_dim, decoder_output_dim, bias=False)
        self.W2 = nn.Linear(decoder_output_dim, decoder_output_dim, bias=False)
        self.V = nn.Linear(decoder_output_dim, 1, bias=False)
        self.max_output_len = max_output_len
    def forward(self, inputs, inputs_mask, inputs_embeds, features, use_emb=False):
        if not use_emb:
            assert self.inp_feature_dim == self.decoder_output_dim
        batch_size = inputs.shape[0]
        probs = []
        dec_input = features
        hidden = torch.zeros(batch_size, self.decoder_output_dim).to(inputs.device)
        for step in range(self.max_output_len):
            hidden = self.decoder(dec_input, hidden)
            transed_inputs = self.W2(inputs)
            transed_dec_output = self.W1(hidden).unsqueeze(1).expand_as(transed_inputs)
            scores = torch.tanh(self.V(transed_dec_output + transed_inputs)).squeeze(-1)
            # print(scores.shape, inputs_mask.shape)
            _scores = scores + inputs_mask
            _scores = torch.softmax(_scores, 1)
            if step == 0:
                # prepare for candicate values
                context = _scores.unsqueeze(2).expand_as(inputs).mul(inputs).sum(1)
            probs.append(scores)
            candi_idx = torch.argmax(_scores, 1)
            for batch_id in range(batch_size):
                # 取embedding
                if use_emb:
                    dec_input[batch_id] = inputs_embeds[batch_id, candi_idx[batch_id]]
                # 取inputs
                else:
                    dec_input[batch_id] = inputs[batch_id, candi_idx[batch_id]]
        probs = torch.stack(probs, 1)
        return probs, context




class TestMyDataset():
    def __init__(self, data_path, data_device=torch.device('cuda:0')):
        """
        用于加载numpy的文件（*.npy）
        """
        file_list = os.listdir(data_path)
        self.data_len = 0
        for file_name in file_list:
            if not file_name.endswith('npy'):
                continue
            full_name = os.path.join(data_path, file_name)
            data = torch.from_numpy(np.load(full_name)).to(data_device)
            if data.dtype == torch.int32:
                data = data.long()
            self.data_len = max(self.data_len, data.shape[0])
            self.__setattr__(file_name.split('.')[0], data)

    def get_samples(self, start, ed):
        ret = dict()
        for key in self.__dict__:
            if isinstance(self.__getattribute__(key), torch.Tensor):
                if self.__getattribute__(key).shape[0] == self.data_len:
                    ret[key] = self.__getattribute__(key)[start:ed]
        return ret

class DataLoader():
    def __init__(self, data_root, batch_size):
        super().__init__()
        # possible batch start
        self.data_root = data_root
        self.batch_size = batch_size
        self.batch_num = 0
        self.batch_starts = 0
        self.tot_batchs = 0
        # self.dataset = MyDataset(data_root)
        self.dataset = None

    def batchs(self, start=0, batchs=1000000, shuffle=False):
        file_list = os.listdir(self.data_root)
        # np.random.shuffle(file_list)
        for sub_dir in file_list:
            self.dataset = TestMyDataset(os.path.join(self.data_root, sub_dir))
            self.batch_starts = list(
                range(0, self.dataset.data_len, self.batch_size))
            self.batch_starts[-1] = self.dataset.data_len - self.batch_size
            if batchs == 1000000 and shuffle:
                self.shuffle()  # each epoch
            for batch_start in self.batch_starts[start:batchs]:
                yield self.dataset.get_samples(batch_start, batch_start+self.batch_size)

    def shuffle(self):
        np.random.shuffle(self.batch_starts)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.linear = LinearModel([in_features, out_features, out_features], dropout=0.1)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj):
        support = self.linear(inp)
        output = torch.matmul(adj, support)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'