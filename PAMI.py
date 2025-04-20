import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

import numpy as np
class SEBlock_MA(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SEBlock_MA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B,2,-1,H,W)
        
        x1 = self.avg_pool(x_[:,0])
        x2 = self.max_pool(x_[:,1])
        y = torch.cat([x1,x2], dim=1)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    

class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()

        branches_1=self.conv3x3(x)
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1)
        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2



        feat1=torch.cat([fusion_1_2,branches_3],dim=1)
        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
        att_3=att1[:,1,:,:].unsqueeze(1)


        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax

class DWConv(nn.Module) :
    def __init__(self, in_channels , kernel_size =3) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, padding='same', groups=in_channels)
    
    def forward(self, x) :
        x = self.dwconv(x)
        return x

class GroupedAsymmetricConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", groups_num=2):
        super(GroupedAsymmetricConvolution, self).__init__()
        if padding == "same":
            pad_size = (kernel_size - 1) // 2
        elif padding == "None":
            pad_size = 0
        else:
            pad_size = padding
        self.conv1x3 = nn.Conv2d(in_channels, in_channels // groups_num, kernel_size=(1, kernel_size), padding=(0, pad_size), groups=groups_num)
        self.conv3x1 = nn.Conv2d(in_channels // groups_num, out_channels, kernel_size=(kernel_size, 1), padding=(pad_size, 0), groups=groups_num)

    def forward(self, x):
        x = self.conv1x3(x)
        x = self.conv3x1(x)
        return x

class UpConvBlock_LK_S_RES(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode, activation
                 , norm, scale_factor = 2, groups_num = 1, expand_ratio =1,
                 att = "se", att_type="m"):
        super().__init__()  
        
        # GAConv
        self.gaconvblock = GAConv_Block_RES(in_channels, out_channels, kernel_size, activation, norm, 
                                groups_num = groups_num, expand_ratio = expand_ratio,
                                att=att)
      
            
        if mode == "upsample":
            self.upscale = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
                nn.Conv2d(out_channels, out_channels//scale_factor, kernel_size=1, padding='same')
            )
            
        else :
            self.upscale = nn.ConvTranspose2d(out_channels, out_channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)
            
    def forward(self, x):
        x = self.gaconvblock(x)
        x = self.upscale(x)
        return x



class GAConv_Block_RES(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="gelu", norm="batch", 
                 groups_num = 1, expand_ratio = 1, att = "sema"):
        super().__init__()
        self.conv1  = GroupedAsymmetricConvolution(in_channels, out_channels*expand_ratio, kernel_size, padding='same', groups_num = groups_num)
        self.conv2  = GroupedAsymmetricConvolution(out_channels*expand_ratio, out_channels, kernel_size, padding='same', groups_num = groups_num)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            self.relu = nn.ReLU()
        elif activation == "leaky_relu":
            self.relu = nn.LeakyReLU()
        elif activation == "gelu":
            self.relu = nn.GELU()

        self.att = SEBlock_MA(out_channels)
        self.att1 = SEBlock_MA(out_channels)

    def forward(self, x):
        
        x = self.conv1(x) 
        x = self.norm1(x)  
        x = self.relu(x)
        x = self.att1(x)
        
        res_x = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        x = torch.add(res_x,x)
        
        x = self.att(x)
        
        
        
        return x

class SCA_Block(nn.Module):
    def __init__(self, in_channels, spatial_att = "sum", block_type = 1, res = 0):
        super().__init__()
        # spatial branch
        if spatial_att == "sum":
            self.spatial_conv = nn.Conv2d(1, 1, kernel_size=3, padding='same')
        elif spatial_att == "origin":
            # only mul
            self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding='same')
            # weight + bias
            # self.spatial_conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding='same')
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(1)
        # channel branch
        self.channel_conv = DWConv(in_channels = in_channels)
        self.block_type = block_type
        self.res = res

    def forward(self, x):
        # channel dimension
        x_channel = self.channel_conv(x)
        # spatial dimension
        if self.spatial_conv.in_channels == 1:
            if self.block_type ==1:
                x_spatial = torch.unsqueeze(torch.sum(x_channel, dim=1),dim=1)
            else :
                x_spatial = torch.unsqueeze(torch.sum(x, dim=1),dim=1)
        x_spatial_kernel = self.spatial_conv(x_spatial)
        x_spatial_kernel = self.norm(x_spatial_kernel)
        x_spatial_kernel = self.softmax(x_spatial_kernel)
        x_spatial = torch.mul(x_channel, x_spatial_kernel)
        if self.res ==1:
            x_spatial = torch.add(x_spatial, x)
        elif self.res ==2:
            x_spatial = torch.mul(x_spatial, x)
        return x_spatial


class Encoder_PAMI_RES(nn.Module):
    def __init__(self, input_img_channels, activation, norm, 
                 blocks_num = [64, 128, 256, 512], groups_num = 1, channel_reduction = True, 
                 kernel_size = 3, downsample = "conv", att="se"):
        if channel_reduction:
            blocks_num = [num//2 for num in blocks_num]
        super().__init__()
        
        # Normal Conv
        self.block1 = GAConv_Block_RES(in_channels = input_img_channels, out_channels = blocks_num[0], kernel_size = kernel_size, 
                                activation = activation, norm=norm, att=att)

        self.block2 = GAConv_Block_RES(in_channels = blocks_num[0], out_channels = blocks_num[1], kernel_size = kernel_size, 
                                activation = activation, norm=norm, groups_num = groups_num, att=att)
        self.block3 = GAConv_Block_RES(in_channels = blocks_num[1], out_channels = blocks_num[2], kernel_size = kernel_size, 
                                activation = activation, norm=norm, groups_num = groups_num,
                                att=att)
        self.block4 = GAConv_Block_RES(in_channels = blocks_num[2], out_channels = blocks_num[3], kernel_size = kernel_size, 
                                activation = activation, norm=norm, groups_num = groups_num,
                                att=att)

        if downsample == "maxpool":
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(2)
            self.pool3 = nn.MaxPool2d(2)
            
        elif downsample =="avgpool":
            self.pool1 = nn.AvgPool2d(2)
            self.pool2 = nn.AvgPool2d(2)
            self.pool3 = nn.AvgPool2d(2)
            
        elif downsample == "conv":
            self.pool1 = nn.Conv2d(in_channels=blocks_num[0], out_channels=blocks_num[0], kernel_size=2, stride=2)
            self.pool2 = nn.Conv2d(in_channels=blocks_num[1], out_channels=blocks_num[1], kernel_size=2, stride=2)
            self.pool3 = nn.Conv2d(in_channels=blocks_num[2], out_channels=blocks_num[2], kernel_size=2, stride=2)

    def forward(self, x):
        x_layer1 = self.block1(x)

        x_layer2 = self.pool1(x_layer1)
        x_layer2 = self.block2(x_layer2)

        x_layer3 = self.pool2(x_layer2)
        x_layer3 = self.block3(x_layer3)

        x_layer4 = self.pool3(x_layer3)
        x_layer4 = self.block4(x_layer4)

        return x_layer1, x_layer2, x_layer3, x_layer4

class Decoder_PAMI_RES(nn.Module):
    def __init__(self, up_mode, activation, norm, blocks_num = [512, 1024, 256, 128, 64, 32], 
                 groups_num = 1, shortcut = "sdwc", channel_reduction = True, kernel_size = 3,
                 channel_att = "multi", att="se", SDWC_block_type = 1, SDWC_res = 0,
                 skip_fusion = False, skip_conn=""):
        super().__init__()

        if channel_reduction:
            blocks_num = [num//2 for num in blocks_num]
        self.block4 = UpConvBlock_LK_S_RES(in_channels = blocks_num[0], out_channels = blocks_num[0], 
                                       kernel_size = kernel_size, mode=up_mode, 
                                       activation = activation, norm = norm, 
                                       groups_num = groups_num, att=att)
        self.block3 = UpConvBlock_LK_S_RES(in_channels = blocks_num[2], out_channels = blocks_num[2], 
                                       kernel_size = kernel_size, mode=up_mode, 
                                       activation = activation, norm = norm, 
                                       groups_num = groups_num, att=att)
        self.block2 = UpConvBlock_LK_S_RES(in_channels = blocks_num[3], out_channels = blocks_num[3], 
                                       kernel_size = kernel_size, mode=up_mode, 
                                       activation = activation, norm = norm, 
                                       groups_num = groups_num, att=att)
          
        self.block1 = GAConv_Block_RES(in_channels = blocks_num[4], out_channels = blocks_num[5], kernel_size = 3, 
                                activation = activation, norm = norm, att=att)

        self.sa4 = SCA_Block(in_channels=blocks_num[1]//4, 
                                              block_type=SDWC_block_type, res=SDWC_res)
        self.sa3 = SCA_Block(in_channels=blocks_num[0]//4, 
                                            block_type=SDWC_block_type, res=SDWC_res)
        self.sa2 = SCA_Block(in_channels=blocks_num[2]//4, 
                                            block_type=SDWC_block_type, res=SDWC_res)
        self.sa1 = SCA_Block(in_channels=blocks_num[3]//4, 
                                            block_type=SDWC_block_type, res=SDWC_res)
            
            

    def forward(self, x_layer1, x_layer2, x_layer3, x_layer4):
        # level 4
        # dwconv
        x_l4 = self.block4(x_layer4)
        x_l4 = self.sa4(x_l4)
        # level 3
        # add
        # For DCA1
        if x_layer3.shape[2]!=x_l4.shape[2]:
            x_l4 = torch.nn.functional.pad(x_l4, (0, 1, 0, 1))
        x_layer3 = torch.add(x_l4, x_layer3)
        x_l3 = self.block3(x_layer3)
        x_l3 = self.sa3(x_l3)
        
        # level 2
        # add
        x_layer2 = torch.add(x_l3, x_layer2)
        x_l2 = self.block2(x_layer2)
        x_l2 = self.sa2(x_l2)
        # level 1
         # dwconv
        # add
        x_layer1 = torch.add(x_l2, x_layer1)
        x_l1 = self.block1(x_layer1)
        x_l1 = self.sa1(x_l1)
        
        
        
        
        
        return x_l1, x_l2, x_l3, x_l4

class PAMI_RES(nn.Module):
    def __init__(self, input_img_channels = 3, output_ch = 1, 
                 up_mode = "upconv", downsample="conv",activation = "relu", norm = "batch", 
                 shortcut = "sc", blocks_num = [64, 32], groups_num = 1,
                 channel_reduction = True, kernel_size = 3, 
                 ff_rate = [0.6, 0.2, 0.1, 0.1], ff_upscale = "upconv", att="se",
                 SDWC_block_type = 1, SDWC_res = 0, skip_fusion = False,
                 skip_conn = "", local_upsample = "upconv"):
        super().__init__()


        if channel_reduction:
            blocks_num = [num//2 for num in blocks_num]
        self.encoder = Encoder_PAMI_RES(input_img_channels, activation = activation, norm = norm, 
                                    downsample=downsample, groups_num = groups_num, 
                                    channel_reduction = channel_reduction,
                                    kernel_size = kernel_size, att=att)
 
        self.decoder = Decoder_PAMI_RES(up_mode, activation = activation, norm = norm,
                                    groups_num = groups_num, shortcut = shortcut, channel_reduction = channel_reduction,
                                    kernel_size = kernel_size, att=att, 
                                    SDWC_block_type=SDWC_block_type, SDWC_res=SDWC_res,
                                    skip_fusion = skip_fusion, skip_conn=skip_conn)

        self.sigmoid = nn.Sigmoid()
        
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding='same')
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.conv_layer3_4 = nn.Conv2d(in_channels=32, out_channels=output_ch, kernel_size=1, padding='same')
        
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding='same')
        self.upscale_layer4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscale_layer3 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.compress = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16,1,1,1,"same"),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.outputCompress = nn.Conv2d(32,output_ch,1,1,"same")


        self.sap4 = SAPblock(256)
    def forward(self,x):
        x_layer1, x_layer2, x_layer3, x_layer4 = self.encoder(x)

        x_layer4 = self.sap4(x_layer4)
        x_layer1, x_layer2, x_layer3, x_layer4 = self.decoder(x_layer1, x_layer2, x_layer3, x_layer4)
        # dshff 
        x_layer4 = self.upscale_layer4(x_layer4)
        x_layer3 = self.upscale_layer3(x_layer3)      
        x_layer4 = self.conv_layer4(x_layer4)
        x_layer3 = self.conv_layer3(x_layer3)
        x_layer3 = torch.add(x_layer3, x_layer4)
        x_layer3 = self.conv_layer3_4(x_layer3)
        
        x_layer2 = self.conv_layer2(x_layer2)
        x_layer2_ = torch.add(x_layer2, x_layer1)
        x_layer2_ = self.compress(x_layer2_)
        x_layer2 = torch.mul(x_layer2,x_layer2_)
        x_layer1 = torch.mul(x_layer1,x_layer2_)
        x_layer1 = torch.cat([x_layer1,x_layer2], dim=1)
        x_layer1 = self.outputCompress(x_layer1)
 
        x_layer1 = torch.add(x_layer1,x_layer3)
        
        x = self.sigmoid(x_layer1)

        return x

if __name__ == "__main__":
    pic_SIZE = (1, 3, 256, 256)
    pic_SIZE = (1, 3, 512, 512)
    model = PAMI_RES()
    

    # output1 = model.forward(torch.rand(pic_SIZE))
    # print(output1.shape)

    device = "cuda"
    model.to(device)

    # import torchinfo 
    # torchinfo.summary(model, pic_SIZE )

    from thop import profile
    inputss = torch.rand(pic_SIZE).to(device)
    flops, params = profile(model, inputs=(inputss,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
