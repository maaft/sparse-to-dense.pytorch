import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

oheight, owidth = 320, 320

class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_feats, *shapes):
        super(MultiResolutionFusion, self).__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module("resolve{}".format(i), nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                ))
            else:
                self.add_module(
                    "resolve{}".format(i),
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__("resolve{}".format(i))(x)

        return output


class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super(ChainedResidualPool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{}".format(i), nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):

    def __init__(self, feats):
        super(ChainedResidualPoolImproved, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module("block{}".format(i), nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):

    def __init__(self, features,
                 residual_conv_unit,
                 multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super(BaseRefineNetBlock, self).__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(
                residual_conv_unit(feats),
                residual_conv_unit(feats)
            ))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):

    def __init__(self, features, *shapes):
        super(RefineNetBlock, self).__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):

    def __init__(self, features, *shapes):
        super(RefineNetBlockImprovedPooling, self).__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

# class ResidualConvUnit(nn.Module):
#
#     def __init__(self, in_channels, n_filters=256, kernel_size=3):
#         super(ResidualConvUnit, self).__init__()
#
#         self.residual_conv_unit = nn.Sequential(collections.OrderedDict([
#             ('relu',      nn.ReLU()),
#             ('conv',      nn.Conv2d(in_channels, n_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)),
#             ('relu',      nn.ReLU()),
#             ('conv',      nn.Conv2d(in_channels, n_filters, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)),
#             ]))
#
#     def forward(self, x):
#         return self.residual_conv_unit.forward(x)
#
#
# class ChainedResidualPooling(nn.Module):
#
#     def __init__(self, in_channels, n_filters=256):
#         super(ChainedResidualPooling, self).__init__()
#
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1)
#         self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         net_relu = self.relu.forward(x)
#         net = self.max_pool.forward(net_relu)
#         net = self.conv.forward(net)
#         sum1 = net + net_relu
#
#         net = self.max_pool.forward(net)
#         net = self.conv.forward(net)
#
#         return net + sum1
#
# class MultiResolutionFusion(nn.Module):
#     def __init__(self, in_channels, n_filters=256):
#         super(MultiResolutionFusion, self).__init__()
#
#         self.in_channels = in_channels
#
#         self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#
#     def forward(self, *input):
#
#         if self.in_channels == 512:
#             low_inputs = input[0]
#
#             return self.conv(low_inputs)
#         else:
#             high_inputs = input[0]
#             low_inputs = input[1]
#
#             conv_low = self.conv(low_inputs)
#             conv_high = self.conv(high_inputs)
#
#             conv_low_up = self.upsample(conv_low)
#
#             return conv_low_up + conv_high
#
# class RefineBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(RefineBlock, self).__init__()
#
#         self.n_filters = n_filters
#         self.in_channels = in_channels
#
#         if(n_filters == 512):
#             self.res_conv_unit = ResidualConvUnit(in_channels, 512)
#             self.fuse = MultiResolutionFusion(512, 512)
#             self.fuse_pooling = ChainedResidualPooling(512, 512)
#         else:
#             self.res_conv_unit = ResidualConvUnit(in_channels, 256)
#             self.fuse = MultiResolutionFusion(256, 256)
#             self.fuse_pooling = ChainedResidualPooling(256, 256)
#
#     def forward(self, *input):
#
#         if(self.in_channels == 512):
#             high_inputs = input[0]
#             rcu_new_low = self.res_conv_unit(high_inputs)
#             x = self.res_conv_unit(rcu_new_low)
#
#             x = self.fuse(x)
#             x = self.fuse_pooling(x)
#             x = self.res_conv_unit(x)
#
#             return x
#         else:
#             high_inputs = input[0]
#             low_inputs = input[1]
#             rcu_high = self.res_conv_unit(high_inputs)
#             rcu_high = self.res_conv_unit(rcu_high)
#
#             x = self.fuse(rcu_high, low_inputs)
#             x = self.fuse_pooling(x)
#             x = self.res_conv_unit(x)
#
#             return x


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

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm 
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
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

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

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


class RefineNet(nn.Module):
    def __init__(self, layers, features, in_channels, size=oheight, out_channels=1, pretrained=True):

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

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        input_size = (in_channels, size)

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
            ResidualConvUnit(features),
            nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

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

        out = self.output_conv(path_1)

        return out

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
