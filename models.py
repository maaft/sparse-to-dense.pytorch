import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
from blocks import *

oheight, owidth = 256, 320
#oheight, owidth = 32, 40

class DenseBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, downsample=False, upsample=False):
        super(DenseBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        if downsample:
            self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                                   padding=1, stride=2, bias=False)
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(interChannels, growthRate, kernel_size=3,
                                   padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                                   padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlocks, downsample = False, upsample = False):
        num_inputs = nChannels
        layers = []
        layers.append(nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False))
        for i in range(int(nDenseBlocks)):
            if i < int(nDenseBlocks) - 1: # only down-/upsample on last layer
                layers.append(DenseBottleneck(num_inputs + 2, growthRate)) # + 2 for S1, S2 sparse-depth maps
            else:
                layers.append(DenseBottleneck(num_inputs + 2, growthRate, downsample, upsample))
            num_inputs += growthRate
        layers.append(nn.Conv2d(growthRate, nChannels, kernel_size=3, padding=1, stride=2, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x): # x = torch.cat((input, S1, S2), 1)
        return self.layers(x)

class D3(nn.Module):
    def __init__(self, in_channels, nChannels, growthRate, nDenseBlocks):
        super(D3, self).__init__()
        self.downconv = nn.Conv2d(1, 1, 1, stride=2)
        self.input_conv = nn.Conv2d(in_channels, nChannels, kernel_size=3, stride=2)

        for i in range(4):
            self.add_module("down{}".format(i), self._make_dense(nChannels, growthRate, nDenseBlocks, downsample=True))
            self.add_module("up{}".format(i), self._make_dense(nChannels, growthRate, nDenseBlocks, upsample=True))

        for i in range(5):
            self.add_module("dense{}".format(i), self._make_dense(nChannels, growthRate, nDenseBlocks))

        self.output_conv = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2)

    def forward(self, *input):
        S1 = input[1]
        S2 = input[2]

        s12 = []
        for i in range(5):
            S1 = self.downconv(S1)
            S2 = self.downconv(S2)
            s12.append(torch.cat((S1, S2), 1))

        out = self.input_conv(input)

        skip = []

        for i in range(3):
            out = torch.cat((out, s12[i]), 1)
            out = self.__getattr__("down{}".format(i))(out)
            skip.append(self.__getattr__("dense{}".format(i))(out))

        #out = torch.cat((out, s12[0]), 1)
        #out = self.__getattr__("down0")(out)
        #skip0 = self.__getattr__("dense0")(out)

        #out = torch.cat((out, s12[1]), 1)
        #out = self.__getattr__("down1")(out)
        #skip1 = self.__getattr__("dense1")(out)

        #out = torch.cat((out, s12[2]), 1)
        #out = self.__getattr__("down2")(out)
        #skip2 = self.__getattr__("dense2")(out)

        out = torch.cat((out, s12[3]), 1)
        out = self.__getattr__("down3")(out)

        out = self.__getattr__("dense3")(out)
        out = self.__getattr__("dense4")(out)

        out = self.__getattr__("up3")(out)

        for i in range(3):
            out = skip[2-i] + out
            out = self.__getattr__("up{}".format(2-i))(out)

        #out = skip2 + out
        #out = self.__getattr__("up2")(out)

        #out = skip1 + out
        #out = self.__getattr__("up1")(out)

        #out = skip0 + out
        #out = self.__getattr__("up0")(out)

        out = self.output_conv(out)

        out = out + S1

        return out


    # growthRate = k = 12
    # nDenseBlocks = L = 5
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, downsample = False, upsample = False):
        num_inputs = nChannels
        layers = []
        layers.append(nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False))
        for i in range(int(nDenseBlocks)):
            if i < int(nDenseBlocks) - 1: # only down-/upsample on last layer
                layers.append(DenseBottleneck(num_inputs + 2, growthRate)) # + 2 for S1, S2 sparse-depth maps
            else:
                layers.append(DenseBottleneck(num_inputs + 2, growthRate, downsample, upsample))
            num_inputs += growthRate
        layers.append(nn.Conv2d(growthRate, nChannels, kernel_size=3, padding=1, stride=2, bias=False))
        return nn.Sequential(*layers)

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding 
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU 
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d): 
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None: 
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()
        
        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
            
            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

#class FastUpConv(Decoder):
#    def upconv_module(self, in_channels):
#        conv_a = nn.Conv2d(in_channels, in_channels, kernel_size=(2,3))

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class UpProjModule(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels):
        super(UpProjModule, self).__init__()
        out_channels = in_channels//2
        self.unpool = Unpool(in_channels)
        self.upper_branch = nn.Sequential(collections.OrderedDict([
          ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm1', nn.BatchNorm2d(out_channels)),
          ('relu',      nn.ReLU()),
          ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
          ('batchnorm2', nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = nn.Sequential(collections.OrderedDict([
          ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x = x1 + x2
        x = self.relu(x)
        return x

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = UpProjModule(in_channels)
        self.layer2 = UpProjModule(in_channels//2)
        self.layer3 = UpProjModule(in_channels//4)
        self.layer4 = UpProjModule(in_channels//8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7 
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)

class RCNN(nn.Module):
    def __init__(self, layers, batchsize, decoder, in_channels, input_size=oheight, out_channels=1, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(RCNN, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        self.batchsize = batchsize

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            self.num_channels = 512
        elif layers >= 50:
            self.num_channels = 2048

        self.output = None
        self.hidden = None

        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.num_channels // 2)

        self.lstm = nn.LSTM(input_size=self.num_channels // 2 * 8 * 10, hidden_size=10, num_layers=2, bias=False)

        # decoding
        self.decoder = choose_decoder(decoder, self.num_channels // 2)
        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(self.num_channels // 32, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=(oheight, owidth), mode='bilinear')

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.output is not None:
            sequence = [self.output, x]
            self.output = x
            sequence = torch.cat(sequence).view(len(sequence), self.batchsize, -1)
            if self.hidden is None:
                self.hidden = (torch.randn(2, self.batchsize, 1000), torch.randn(2, self.batchsize, 1000))
            x, self.hidden = self.lstm(sequence)
            print(x.shape)
        else:
            self.output = x

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

class RefineNet(nn.Module):
    def __init__(self, layers, decoder, features, in_channels, input_size=oheight, out_channels=1, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(RefineNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        features = 256

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBlock(
            2 * features, (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(
            features, (2 * features, input_size // 32), (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(
            features, (features, input_size // 16), (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(
            features, (features, input_size // 8), (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features),
            #nn.BatchNorm2d(features),
            #nn.ReLU(inplace=True),
            ResidualConvUnit(features),
            #nn.BatchNorm2d(features),
            #nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(features, features // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features // 2)
        #self.decoder = choose_decoder(decoder, features // 2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(features // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=(oheight, owidth), mode='bilinear')

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        #self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        x = self.output_conv(path_1)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        #x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

class ResNet(nn.Module):
    def __init__(self, layers, decoder, in_channels=3, out_channels=1, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=(oheight, owidth), mode='bilinear')

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x
