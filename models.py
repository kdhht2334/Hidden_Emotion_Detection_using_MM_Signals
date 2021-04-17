#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kdh
@email: kdhht5022@gmail.com
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM, GRU

import torch 
import torch.nn as nn
from torch.autograd import Variable

import pretrainedmodels
# import pretrainedmodels.utils as utils

model_name = 'alexnet'  # 'bninception'
#resnext = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').cuda()
alexnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').cuda()


lstm_sell = (16+1) * 25  # 16
n_features = 444
time_steps = 8

def lstm_keras():
    
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    
    model = Sequential()
    model.add(LSTM(lstm_sell, return_sequences=True,
                   input_shape=(8, n_features), dropout=0.5))
                   #input_shape=(1,n_features), recurrent_dropout=0.5))
    model.add(LSTM(lstm_sell, return_sequences=True, dropout=0.5))
    model.add(LSTM(lstm_sell, return_sequences=True, dropout=0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(2, activation='tanh'))
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse'])

    model.summary()

    return model


def lstm_keras_prev():
    model = Sequential()
    model.add(LSTM(lstm_sell, return_sequences=True,
                   input_shape=(8, n_features), dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(lstm_sell, return_sequences=True, dropout=0.5, recurrent_dropout=0.3))
    model.add(LSTM(lstm_sell, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])

    model.summary()

    return model


def lstm_model_eeg():
    model = Sequential()
    model.add(LSTM(lstm_sell, return_sequences=True,
                   input_shape=(time_steps, n_features), dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(lstm_sell, return_sequences=True, dropout=0.5, recurrent_dropout=0.3))
    ##model.add(LSTM(lstm_sell, return_sequences=True))

    model.add(Flatten())
    ##model.add(Dense(8))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])

    model.summary()

    return model


class TwoLayerNet(torch.nn.Module):
    
    def __init__(self, D_in, H1, H2, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        
    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        h_relu3 = self.linear3(h_relu2)
        return h_relu3
    
    
class Encoder2(nn.Module):
    
    def __init__(self):
        super(Encoder2, self).__init__()
        
        self.features = alexnet._features

    def forward(self, x):
        x = self.features(x)
        return x
    
    
class Regressor(nn.Module):
    
    def __init__(self):
        super(Regressor, self).__init__()
        
        self.avgpool = alexnet.avgpool
        self.lin0 = alexnet.linear0
        self.lin1 = alexnet.linear1
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))
        
        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x


class Regressor_light(nn.Module):

    def __init__(self):
        super(Regressor_light, self).__init__()

        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 64)
        self.lin1 = nn.Linear(64, 8)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
#        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))

#        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x
    
    
class RNN(nn.Module):
    def __init__(self, fc_input_size, fc_output_size, num_layers, num_classes, mode, bidirectional, use_gpu, gpu_id):
        super(RNN, self).__init__()
        #self.hidden_size = hidden_size
        self.mode = mode
        self.num_layers = num_layers
        self.fc_input_size = fc_input_size
        self.fc_output_size = fc_output_size
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.num_layers = num_layers

        
        if len(fc_input_size) == 2:
            self.fc_input0 = nn.Linear(fc_input_size[0], fc_input_size[1] )
            self.dropout_fc_input0 = nn.Dropout(0.5)
            
        
        if len(fc_input_size) == 3:            
            self.fc_input0 = nn.Linear(fc_input_size[0], fc_input_size[1] )
            self.dropout_fc_input0 = nn.Dropout(0.5)    
            self.fc_input1 = nn.Linear(fc_input_size[1], fc_input_size[2] )            
            self.dropout_fc_input1 = nn.Dropout(0.5)
        

        if self.mode == 'lstm':
            self.lstm1 = nn.LSTM(fc_input_size[-1], fc_output_size[0], num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)
            # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)
            # self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)
        elif self.mode == 'gru':
            self.gru1 = nn.GRU(fc_input_size[-1], fc_output_size[0], num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)
            # self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)
            # self.gru3 = nn.GRU(hidden_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=bidirectional)

        
        if len(fc_output_size) == 1:
            self.fc_output0 = nn.Linear(fc_output_size[0], num_classes)
            self.dropout_fc_output0 = nn.Dropout(0.5)
        elif len(fc_output_size) == 2:
            self.fc_output0 = nn.Linear(fc_output_size[0], fc_output_size[1])
            self.dropout_fc_output0 = nn.Dropout(0.5)
            self.fc_output1 = nn.Linear(fc_output_size[1], num_classes)            
            self.dropout_fc_output1 = nn.Dropout(0.5)
            
            
        self.relu = nn.ReLU(True)
                   
            
    def forward(self, x):
        # Set initial states 
        if self.use_gpu == True:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.fc_output_size[0]).cuda(self.gpu_id)) 
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.fc_output_size[0]).cuda(self.gpu_id))
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.fc_output_size[0])) 
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.fc_output_size[0]))
        
        
        if len(self.fc_input_size) == 2:
            x = self.fc_input0(x)  #   x -> [1, 24, 1024]
            x = self.dropout_fc_input0(x) #   x -> [1, 24, 1024]
            x = self.relu(x) #   x -> [1, 24, 1024]
            
                    
        if len(self.fc_input_size) == 3:      
            x = self.fc_input0(x)
            x = self.dropout_fc_input0(x)
            x = self.relu(x)
            x = self.fc_input1(x)
            x = self.dropout_fc_input1(x)
            x = self.relu(x)
            
        # Forward propagate RNN
        if self.mode == 'lstm':
            x, _ = self.lstm1(x, (h0, c0))
        elif self.mode == 'gru':
            x, _ = self.gru1(x, h0)

        # Decode hidden state of last time step
        
        if len(self.fc_output_size) == 1:
            x = self.dropout_fc_output0(x)
            final_feature0 = x[:, 5, :]
            final_feature1 = x[:, 11, :]
            final_feature2 = x[:, 17, :]
            final_feature3 = x[:, 23, :]
            out0 = self.fc_output0(x[:, 5, :])
            out1 = self.fc_output0(x[:, 11, :])
            out2 = self.fc_output0(x[:, 17, :])
            out3 = self.fc_output0(x[:, 23, :])
             
            
        elif len(self.fc_output_size) == 2:
            x = self.dropout_fc_output0(x)
            x = self.fc_output0(x[:, -1, :])
            x = self.relu(x)
            x = self.dropout_fc_output1(x)
            final_feature = x
            out = self.fc_output1(x)
        
        #return [out0, out1,out2,out3], x
        return [out0, out1,out2,out3] , [final_feature0, final_feature1, final_feature2, final_feature3]
        
        
    def forward_lstm(self, x, h, c):
         

        
        x = self.fc_input0(x)
        x = self.dropout_fc_input0(x)
        x = self.relu(x)        
            
        # Forward propagate RNN
        
        out, hc = self.lstm1(x.view(1, 1, 256), (h, c))
        # Decode hidden state of last time step
        h = hc[0]
        c = hc[1]
        return out, h, c
    
    
    
    def forward_fc(self, x):
        x = self.dropout_fc_output0(x)
        out = self.fc_output0(x)
        
        return out  