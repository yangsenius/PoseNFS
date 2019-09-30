import torch
import torch.nn  as nn

## mode1 RELU-CONV-BatchNorm

## mode2 CONV-BatchNorm-RELU

_ZERO    = 'Zero'
_SKIP    = 'skip_connect'
_SCONV_3 = 'Sep_Conv_3x3'
_SCONV_5 = 'Sep_Conv_5x5'
_ACONV_3 = 'Atr_Conv_3x3'
_ACONV_5 = 'Atr_Conv_5x5'
_AVG_P_3 = 'avg_pool_3x3'
_MAX_P_3 = 'max_pool_3x3'

_CONV_3 = 'conv_3x3'

_BASIC_CONV_3 = 'Basic_Conv_3x3'
_BOTTLE_CONV_3 = 'Bottle_Conv_3x3'

_DOWN = 'reduce_connect'
_EQUAL = 'direct_connect'
_UP = 'upsampling_connect'

BN_MOMENTUM=0.1

OPS = {
    _ZERO:      lambda channel : Zero(),
    _SKIP:      lambda channel : Identity(),

    _ACONV_3:   lambda channel : Atr_Conv( channel,channel, 3, 1, 2, dilation = 2, affine = False),
    _ACONV_5:   lambda channel : Atr_Conv( channel,channel, 5, 1, 4, dilation = 2, affine = False),
    _SCONV_3:   lambda channel : Sep_Conv( channel,channel, 3, 1, 1,               affine = False),
    _SCONV_5:   lambda channel : Sep_Conv( channel,channel, 5, 1, 2,               affine = False),

    _AVG_P_3:   lambda channel : nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
    _MAX_P_3:   lambda channel : nn.MaxPool2d(3, stride=1, padding=1),

    _BASIC_CONV_3:   lambda channel : Basic_block( channel,channel, 3, 1, 1,    affine = False),
    _BOTTLE_CONV_3:  lambda channel : Bottle_block( channel,channel, 3, 1, 1,   affine = False),

    _CONV_3: lambda channel : nn.Conv2d(channel,channel,3,1,padding=1)


}


Connections = {
    _DOWN   :   lambda  channel : Downsample_Connection(channel//2, channel, 3, stride = 2, padding= 1 ),
    _EQUAL  :   lambda  channel : Identity(),
    _UP     :   lambda  channel : Upsample_Connection(2*channel, channel, factor=2)
}


class Zero(nn.Module):
    def __init__(self,):
        super(Zero,self).__init__()

    def forward(self, x):
        return x.mul(0.)

class Identity(nn.Module):

    def __init__(self,channel=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Downsample_Connection(nn.Module):

    def __init__(self,channel_in,channel_out,kernel_size,stride, padding):
        super(Downsample_Connection,self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, stride = 2, padding= 1 ),
            nn.BatchNorm2d(channel_out,momentum=0.1),)
            #nn.ReLU(inplace=True)) # addd

    def forward(self,x):

        return self.operation(x)

class Atr_Conv(nn.Module):
    """
    atrous_conv
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=False):

        super(Atr_Conv, self).__init__()

        self.operation = nn.Sequential(
            #nn.ReLU(inplace=True), #mode1
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True), #mode2
        )

    def forward(self, x):
        return self.operation (x)


class Sep_Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):

        super(Sep_Conv, self).__init__()

        self.operation = nn.Sequential(
            #nn.ReLU(inplace=True), #mode1
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True), #mode2
        )

    def forward(self, x):
        return self.operation(x)

class Upsample_Connection(nn.Module):

    def __init__(self,C_in,C_out,factor =2):
        super(Upsample_Connection,self).__init__()

        self.conv_1x1 = nn.Conv2d(C_in,C_out,1,1,0)
        self.bn = nn.BatchNorm2d(C_out,momentum=0.1)
        assert C_out % 2 ==0
        self.factor = factor
        #self.ups = nn.UpsamplingBilinear2d(scale_factor=factor)
    def forward(self , x):
        #x = self.conv_1x1(x)
        #x = self.bn(x)
       
        #print('zzz',self.conv_1x1.__class__.__name__,self.conv_1x1.weight.device,x.device)
        
        x = torch.nn.functional.interpolate(self.bn(self.conv_1x1(x)),  scale_factor = self.factor,
                                               mode = 'bilinear',align_corners = True)
    
        
    
        return x

class ReLUConvBN(nn.Module):
    """
     relu-conv-bn
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):

        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            #nn.ReLU(inplace=True), #mode1
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True) #mode2
        )

    def forward(self, x):
        return self.op(x)

class Basic_block(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(C_in,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,bias=False),
            nn.BatchNorm2d(C_in,momentum = BN_MOMENTUM),
            )

    def forward(self, x):
        return self.operation(x)

class Bottle_block(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,bias=False),
            nn.BatchNorm2d(C_in,momentum = BN_MOMENTUM),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(C_in,momentum = BN_MOMENTUM),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,bias=False),
            nn.BatchNorm2d(C_in,momentum = BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.operation(x)

###### You can construct your basic operators ###################
