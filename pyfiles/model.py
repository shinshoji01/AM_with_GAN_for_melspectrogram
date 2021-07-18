import warnings
warnings.filterwarnings("ignore")
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np

from util import *

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)
    
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2,3), keepdim=True)
    
class ConvolutionalBlock(nn.Module):
    
    def __init__(self, nch_in, nch_out, kernel, pooling="max", activation="ReLU"):
        super(ConvolutionalBlock, self).__init__()
        
        layers = [nn.Conv2d(nch_in, nch_out, kernel_size=kernel, stride=1, padding=int((kernel-1)/2))]
        
        if pooling=="max":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif pooling=="avg":
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        elif pooling=="none":
            pass
        
        layers.append(nn.BatchNorm2d(nch_out))
            
        if activation=="ReLU":
            layers.append(nn.ReLU())
        elif activation=="Tanh":
            layers.append(nn.Tanh())
        elif activation=="Sigmoid":
            layers.append(nn.Sigmoid())
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
class CNN_classifier_mel(nn.Module):
    
    def __init__(self, nch_input, nch_output, layer_num=4, nch=64):
        super(CNN_classifier_mel, self).__init__()
        
        self.pre_conv = ConvolutionalBlock(nch_input, nch, 11, "none")
        conv_layers = []
        in_nch = nch
        for i in range(layer_num):
            out_nch = in_nch * 2
            conv_layers.append(ConvolutionalBlock(in_nch, out_nch, 3, "max", "ReLU"))
            in_nch = out_nch
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.last_layer = nn.Sequential(
            GlobalAvgPool(),
            Flatten(),
            nn.Linear(nch*2**layer_num, nch_output)
        )
        
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.conv_layers(x)
        x = self.last_layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, nch_in, nch_hidden, nch_out, filter_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv1 = nn.Conv1d(nch_in, nch_hidden, filter_size, 1, 
                                       int((filter_size-1)/2)+int(dilation-1), dilation=dilation)
        self.dilated_conv2 = nn.Conv1d(nch_in, nch_hidden, filter_size, 1, 
                                       int((filter_size-1)/2)+int(dilation-1), dilation=dilation)
        self.onebyone_conv1 = nn.Conv1d(nch_hidden, nch_in, 1, 1, 0)
        self.onebyone_conv2 = nn.Conv1d(nch_hidden, nch_out, 1, 1, 0)
        
    def forward(self, data):
        res = data
        tanh_out = self.dilated_conv1(data)
        sigm_out = self.dilated_conv2(data)
        mul = torch.mul(tanh_out, sigm_out)
        res_out = self.onebyone_conv1(mul)
        skip_out = self.onebyone_conv2(mul)
        return torch.add(res_out, res), skip_out
    
class ResidualNet(nn.Module):
    def __init__(self, nch_in, nch_hidden, nch_out, filter_size_list, dilation_list):
        super(ResidualNet, self).__init__()
        resnet_list = []
        for i in range(len(dilation_list)):
            dilation = dilation_list[i]
            filter_size = filter_size_list[i]
            cnn = ResidualBlock(nch_in, nch_hidden, nch_out, filter_size, dilation)
            resnet_list.append(cnn)
        self.Resnets = nn.ModuleList(resnet_list)
        
    def forward(self, data):
        skip_out_list = []
        for resblock in self.Resnets:
            res_out, skip_out = resblock(data)
            skip_out_list.append(skip_out)
            data = res_out
        return skip_out_list

class WaveNet_classifier(nn.Module):
    def __init__(self, resnch_in, resnch_hidden, resnch_out, nch_hidden, nch_out,
                 filter_size_list, dilation_list, input_length, reduce, num_cls, 
                 pre_filter, nch_in, bidirectional=False,):
        super(WaveNet_classifier, self).__init__()
        
        self.preconv = nn.Conv1d(nch_in, resnch_in, pre_filter, 1, int((pre_filter-1)/2), dilation=1)
        ################### Parts of WaveNet ##################
        self.bidirectional = bidirectional
        self.ResNet = ResidualNet(resnch_in, resnch_hidden, resnch_out, filter_size_list, dilation_list)
        if bidirectional:
            self.ResNet_back = ResidualNet(resnch_in, resnch_hidden, resnch_out, filter_size_list, dilation_list)
        #######################################################
        
        ################### stack-CNN layers ##################
        cnn_layers = [nn.Sequential(
                        nn.Conv1d(resnch_out, nch_hidden, kernel_size=11, stride=1, padding=5),
                        nn.MaxPool1d(kernel_size=reduce, stride=reduce),
                        nn.BatchNorm1d(nch_hidden),
                        nn.ReLU())]
        for i in range(num_cls-1):
            cnn = nn.Sequential(
                nn.Conv1d(nch_hidden*2**i, nch_hidden*2**(i+1), kernel_size=7, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=reduce, stride=reduce),
                nn.BatchNorm1d(nch_hidden*2**(i+1)),
                nn.ReLU())
            cnn_layers.append(cnn)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        ########################################################
        
        self.output_layer = nn.Linear(int(input_length/reduce**num_cls*nch_hidden*2**(num_cls-1)), nch_out)
        
    def forward(self, data):
        data = self.preconv(data)
        skip_out_list = self.ResNet(data)
        out = torch.zeros(skip_out_list[0].shape).to(data.device)
        for i in range(len(skip_out_list)):
            out = torch.add(out, skip_out_list[i])
        if self.bidirectional:
            data_back = torch.flip(data, [2])
            skip_out_list_back = self.ResNet_back(data_back)
            for i in range(len(skip_out_list_back)):
                out = torch.add(out, skip_out_list_back[i])
        out = nn.ReLU()(out)
        for layer in self.cnn_layers:
            out = layer(out)
        out = Flatten()(out)
        out = self.output_layer(out)
        return out
    
class WaveNet_classifier_mel(nn.Module):
    def __init__(self, resnch_in, resnch_hidden, resnch_out, nch_hidden, nch_out,
                 filter_size_list, dilation_list, input_length, reduce, num_cls, 
                 pre_filter, nch_in, bidirectional=False,):
        super(WaveNet_classifier_mel, self).__init__()
        
        self.preconv = nn.Conv1d(nch_in, resnch_in, pre_filter, 1, int((pre_filter-1)/2), dilation=1)
        ################### Parts of WaveNet ##################
        self.bidirectional = bidirectional
        self.ResNet = ResidualNet(resnch_in, resnch_hidden, resnch_out, filter_size_list, dilation_list)
        if bidirectional:
            self.ResNet_back = ResidualNet(resnch_in, resnch_hidden, resnch_out, filter_size_list, dilation_list)
        #######################################################
        
        ################### stack-CNN layers ##################
        cnn_layers = [nn.Sequential(
                        nn.Conv1d(resnch_out, nch_hidden, kernel_size=11, stride=1, padding=5),
                        nn.MaxPool1d(kernel_size=reduce, stride=reduce),
                        nn.BatchNorm1d(nch_hidden),
                        nn.ReLU())]
        for i in range(num_cls-1):
            cnn = nn.Sequential(
                nn.Conv1d(nch_hidden*2**i, nch_hidden*2**(i+1), kernel_size=7, stride=1, padding=3),
                nn.MaxPool1d(kernel_size=reduce, stride=reduce),
                nn.BatchNorm1d(nch_hidden*2**(i+1)),
                nn.ReLU())
            cnn_layers.append(cnn)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        ########################################################
        
        self.output_layer = nn.Linear(int(input_length/reduce**num_cls*nch_hidden*2**(num_cls-1)), nch_out)
        
    def forward(self, data):
        shape = data.shape
        data = data.view(shape[0],shape[2],shape[3])
        data = self.preconv(data)
        skip_out_list = self.ResNet(data)
        out = torch.zeros(skip_out_list[0].shape).to(data.device)
        for i in range(len(skip_out_list)):
            out = torch.add(out, skip_out_list[i])
        if self.bidirectional:
            data_back = torch.flip(data, [2])
            skip_out_list_back = self.ResNet_back(data_back)
            for i in range(len(skip_out_list_back)):
                out = torch.add(out, skip_out_list_back[i])
        out = nn.ReLU()(out)
        for layer in self.cnn_layers:
            out = layer(out)
        out = Flatten()(out)
        out = self.output_layer(out)
        return out
    
class LSTM_classifier(nn.Module):
    def __init__(self, inputDim, outputDim, nch=64, bidirectional=True, num_cls=3, reduce=8, num_lstm_layers=5,
                 target_length=2**15, pre_filter=3, lstm_in=32):
        super(LSTM_classifier, self).__init__()
        
        self.preconv = nn.Conv1d(inputDim, nch, pre_filter, 1, int((pre_filter-1)/2), dilation=1)

        
        ################### stack-CNN layers ##################
        cnn_layers = [nn.Sequential(
                        nn.Conv1d(nch, nch*2, kernel_size=11, stride=1, padding=5),
                        nn.MaxPool1d(kernel_size=reduce, stride=reduce),
#                         nn.BatchNorm1d(nch*2),
                        nn.ReLU())]
        for i in range(1, num_cls):
            cnn = nn.Sequential(
                    nn.Conv1d(nch*(2**i), nch*2**(i+1), kernel_size=7, stride=1, padding=3),
                    nn.MaxPool1d(kernel_size=reduce, stride=reduce),
#                     nn.BatchNorm1d(nch*2**(i+1)),
                    nn.ReLU())
            cnn_layers.append(cnn)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        ########################################################
        
        self.lstm = nn.LSTM(input_size = nch*2**num_cls,
                            hidden_size = nch*2**num_cls,
                            batch_first = True,
                            bidirectional=bidirectional,
                            num_layers=num_lstm_layers
                          )
        
        self.output_layer = nn.Sequential(
            Flatten(),
            nn.Linear(int(target_length/reduce**num_cls*nch*2**num_cls*(1+int(bidirectional))), outputDim)
        )
            
    def forward(self, inputs, hidden0=None):
        output = inputs
        output = self.preconv(output)
        for layer in self.cnn_layers:
            output = layer(output)
        output = torch.transpose(output, 1, 2)
        output, (_, _) = self.lstm(output, hidden0)
        output = torch.transpose(output, 1, 2)
        
        output = self.output_layer(output)
        return output
    
class LSTM_classifier_mel(nn.Module):
    def __init__(self, inputDim, outputDim, nch=64, bidirectional=True, num_cls=3, reduce=8, num_lstm_layers=5,
                 target_length=2**15, pre_filter=3, lstm_in=32):
        super(LSTM_classifier_mel, self).__init__()
        
        self.preconv = nn.Conv1d(inputDim, nch, pre_filter, 1, int((pre_filter-1)/2), dilation=1)

        
        ################### stack-CNN layers ##################
        cnn_layers = [nn.Sequential(
                        nn.Conv1d(nch, nch*2, kernel_size=11, stride=1, padding=5),
                        nn.MaxPool1d(kernel_size=reduce, stride=reduce),
#                         nn.BatchNorm1d(nch*2),
                        nn.ReLU())]
        for i in range(1, num_cls):
            cnn = nn.Sequential(
                    nn.Conv1d(nch*(2**i), nch*2**(i+1), kernel_size=7, stride=1, padding=3),
                    nn.MaxPool1d(kernel_size=reduce, stride=reduce),
#                     nn.BatchNorm1d(nch*2**(i+1)),
                    nn.ReLU())
            cnn_layers.append(cnn)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        ########################################################
        
        self.lstm = nn.LSTM(input_size = nch*2**num_cls,
                            hidden_size = nch*2**num_cls,
                            batch_first = True,
                            bidirectional=bidirectional,
                            num_layers=num_lstm_layers
                          )
        
        self.output_layer = nn.Sequential(
            Flatten(),
            nn.Linear(int(target_length/reduce**num_cls*nch*2**num_cls*(1+int(bidirectional))), outputDim)
        )
            
    def forward(self, inputs, hidden0=None):
        shape = inputs.shape
        inputs = inputs.view(shape[0],shape[2],shape[3])
        output = inputs
        output = self.preconv(output)
        for layer in self.cnn_layers:
            output = layer(output)
        output = torch.transpose(output, 1, 2)
        output, (_, _) = self.lstm(output, hidden0)
        output = torch.transpose(output, 1, 2)
        
        output = self.output_layer(output)
        return output
    
class Discriminator_mel(nn.Module):
    
    def __init__(self, nch_input, nch_output, nch=64):
        super(Discriminator_mel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nch_input, nch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.Conv2d(nch, nch*2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch*2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.Conv2d(nch*2, nch*4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch*4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.Conv2d(nch*4, nch*8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch*8),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            
            nn.Conv2d(nch*8, nch_output, kernel_size=1, stride=1, padding=0)
              
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(2,3), keepdim=True)
#         x = torch.sigmoid(x)
        return x.squeeze()

class Generator_mel(nn.Module):
    
    def __init__(self, ndim, nch_output, nch=64):
        super(Generator_mel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ndim, nch*8, kernel_size=(5,10), stride=1, padding=0),
                nn.BatchNorm2d(nch*8),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*8, nch*4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch*4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*4, nch*2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch*2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*2, nch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch, nch_output, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(nch_output)
            ),
        ])
        
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        z = F.sigmoid(z)
        return z