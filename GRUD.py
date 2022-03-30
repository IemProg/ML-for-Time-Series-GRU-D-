# -*- coding: utf-8 -*-
import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time

def test_model_gru_d(loader, model):
    correct, total, total_loss = 0, 0, 0
    model.eval()
    predictions, truths = [], []

    for data, label in loader:
        x, m, delta = data[:, 0], data[:, 1], data[:, 2]

        x = Variable(x.float()).cuda()
        delta = Variable(delta.float()).cuda()
        m = Variable(m.float()).cuda()

        output, hidden = gru_d(x, delta, m)

        label = label
        loss = loss_func(output, Variable(label.long()).cuda())

        total_loss += loss.item()
        predicted = (output.cpu().max(1)[1].data.long()).view(-1)
#         predicted = ((output.cpu().data > 0.5).long()).view(-1)
        predictions += list(predicted.numpy())
        truths += list(label.numpy())
        total += label.size(0)
        correct += (predicted == label).sum()

    model.train()

    return 100 * correct / total, roc_auc_score(truths, predictions), total_loss/len(loader)

class GRUD_CLEAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gru_dropout=0.3, decoder_dropout=0.5, batch_first=True):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
#         self.feature_means = Variable(torch.FloatTensor(feature_means), requires_grad=False)
        # initialize weights and biases
        self.W_r = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U_r = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V_r = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_r = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        self.W_z = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U_z = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V_z = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_z = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        self.W = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        # since W_gamma_x is diagonal, just initialize 1-d
        self.W_gamma_x = nn.Parameter(torch.FloatTensor(input_size).normal_(0, 0.02))
        self.b_gamma_x = nn.Parameter(torch.FloatTensor(input_size).zero_())

        self.W_gamma_h = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_gamma_h = nn.Parameter(torch.FloatTensor(hidden_size).zero_())
        self.gru_dropout = gru_dropout

        self.decoder = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)
        self.register_buffer('X_last_obs', torch.zeros(input_size))

    def forward(self, x, delta, m, h_t=None):
        """

        :param x: features input [batch_size, seq_len, num_features]
        :param delta: time interval of feature observation [batch_size, seq_len, num_features]
        :param m: masking vector {0, 1} of [batch_size, seq_len, num_features]
        :param x_forward: place to replace missing values with [b, seqlen, numf]
        :param h_t: initial hidden state [batch_size, seq_len, hidden_size]
        :return: output [batch_size, output_size], hidden_state [batch_size, hidden_size]
        """
        batch_size, seq_len, input_size = x.size()

        x_forward = getattr(self, 'X_last_obs')

        if h_t is None:
            # initialize to zero
            h_t = Variable(torch.FloatTensor(batch_size, self.hidden_size).zero_())
            if x.is_cuda:
                h_t = h_t.cuda()

        # compute decays
        decay_x = delta*self.W_gamma_x + self.b_gamma_x
        zeroes = Variable(torch.zeros(decay_x.size()))
        if decay_x.is_cuda:
            zeroes = zeroes.cuda()
        gamma_x_t = torch.exp(-torch.max(zeroes, decay_x))

        decay_h = torch.matmul(m, self.W_gamma_h) + self.b_gamma_h
        zeroes = Variable(torch.zeros(decay_h.size()))
        if decay_h.is_cuda:
            zeroes = zeroes.cuda()
        gamma_x_h = torch.exp(-torch.max(zeroes, decay_h))

        # replace missing values
        x_replace = decay_x * x_forward + (1-decay_x) * 0.001
        x[m.byte()] = x_replace[m.byte()]

        # dropout masks, one for each batch
        dropout_rate = self.gru_dropout if self.training else 0.

        W_dropout = Variable((torch.FloatTensor(self.W.size()).uniform_() > dropout_rate).float())
        U_dropout = Variable((torch.FloatTensor(self.U.size()).uniform_() > dropout_rate).float())
        V_dropout = Variable((torch.FloatTensor(self.V.size()).uniform_() > dropout_rate).float())

        if decay_h.is_cuda:
            W_dropout = W_dropout.cuda()
            U_dropout = U_dropout.cuda()
            V_dropout = V_dropout.cuda()

        for t in range(seq_len):
            # decay h
            update_range = Variable(torch.LongTensor(list(range(batch_size))))
            if decay_h.is_cuda:
                update_range = update_range.cuda()
            h_t = h_t.clone().index_copy_(0, update_range, gamma_x_h[:batch_size,t,:] * h_t[:batch_size])

            z_t = F.sigmoid(torch.matmul(x[:batch_size, t, :], self.W_z) + torch.matmul(h_t[:batch_size], self.U_z) + torch.matmul(1-m[:batch_size, t, :], self.V_z) + self.b_z)
            r_t = F.sigmoid(torch.matmul(x[:batch_size, t, :], self.W_r) + torch.matmul(h_t[:batch_size], self.U_r) + torch.matmul(1-m[:batch_size, t, :], self.V_r) + self.b_r)
            h_tilde_t = F.tanh(torch.matmul(x[:batch_size, t, :], self.W*W_dropout) + torch.matmul(h_t[:batch_size]*r_t, self.U*U_dropout) + torch.matmul(1-m[:batch_size, t, :], self.V*V_dropout) + self.b)
            h_t = h_t.clone()
            h_t = h_t.clone().index_copy_(0, update_range, (1 - z_t) * h_t[:batch_size] + z_t * h_tilde_t)

        if batch_size > 1:
            h_t = self.bn(h_t)

        output = F.log_softmax(self.decoder(self.decoder_dropout(h_t)), dim=-1)

        return output, h_t

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, output_size = 1, output_last = False):
        """
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.output_size = output_size
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))

        self.register_buffer('X_last_obs', torch.zeros(input_size))

        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size) #changed

        self.head = nn.Linear(self.hidden_size, self.output_size)

        self.output_last = output_last

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        batch_size = x.shape[0]
        dim_size = x.shape[1]

        # print("\t\t x_last_obsv.shape: ", x_last_obsv.shape)
        # print("\t\t x_mean.shape: ", x_mean.shape)
        # print("\t\t mask.shape: ", mask.shape)
        # print("\t\t x.shape: ", x.shape)
        # print("\t\t h.shape: ", h.shape)

        #delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_x = 1 - torch.exp(-self.gamma_x_l(delta))/2

        #delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))
        delta_h = torch.exp(-self.gamma_h_l(delta))/2
        # print("\t\t delta_x.shape: ", delta_x.shape)
        # print("\t\t delta_h.shape: ", delta_h.shape)

        #x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)  #There was a mistake here
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        combined = torch.cat((x, h, mask), 1)

        z = F.sigmoid(self.zl(combined))
        r = F.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = F.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, input):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        Hidden_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:,0,:,:])
        #X_last_obsv = torch.squeeze(input[:,1,:,:])
        X_last_obsv = getattr(self, 'X_last_obs')
        Mask = torch.squeeze(input[:,1,:,:])
        Delta = torch.squeeze(input[:,2,:,:])

        outputs = None
        x_last_obsv = torch.where(Mask>0, X, X_last_obsv)

        values_xt_ht = []
        for i in range(step_size):
            Hidden_State  = self.step(torch.squeeze(X[:,i:i+1,:])\
                                     , X_last_obsv\
                                     , torch.squeeze(self.X_mean[i:i+1])\
                                     , Hidden_State\
                                     , torch.squeeze(Mask[:,i:i+1,:])\
                                     , torch.squeeze(Delta[:,i:i+1,:]))
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        outputs = self.head(outputs)
        outputs = torch.sigmoid(outputs)

        if self.output_last:
            return outputs[:,-1,:]
        else:
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State
