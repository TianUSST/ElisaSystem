import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    """
    双卷积块 - U-Net的基本构建单元
    
    功能说明：
    - 包含两个连续的3x3卷积层
    - 每个卷积后跟批归一化和ReLU激活
    - 用于特征提取和通道数调整
    """
    def __init__(self, in_ch, out_ch):
        """
        初始化双卷积块
        
        参数说明：
        - in_ch: 输入通道数
        - out_ch: 输出通道数
        """
        super(DoubleConv, self).__init__()
        # 双卷积序列：卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化 -> ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),    # 第一个3x3卷积
            nn.BatchNorm2d(out_ch),                    # 批归一化
            nn.ReLU(inplace=True),                     # ReLU激活
            nn.Conv2d(out_ch, out_ch, 3, padding=1),   # 第二个3x3卷积
            nn.BatchNorm2d(out_ch),                    # 批归一化
            nn.ReLU(inplace=True)                      # ReLU激活
        )

    def forward(self, input):
        """
        前向传播函数
        
        参数：
        - input: 输入特征图
        
        返回：
        - 经过双卷积处理的特征图
        """
        return self.conv(input)


class Unet(nn.Module):
    """
    经典U-Net模型
    
    模型结构说明：
    1. 编码器路径：4层下采样，每层使用双卷积块和最大池化
    2. 解码器路径：4层上采样，每层使用转置卷积和双卷积块
    3. 跳跃连接：将编码器各层特征与解码器对应层连接
    4. 最终输出：1x1卷积 + Sigmoid激活
    
    主要特点：
    - 经典的U-Net架构，适用于医学图像分割
    - 使用跳跃连接保留细节信息
    - 对称的编码器-解码器结构
    """
    def __init__(self, in_ch, out_ch):
        """
        初始化U-Net模型
        
        参数说明：
        - in_ch: 输入通道数
        - out_ch: 输出通道数
        """
        super(Unet, self).__init__()

        # 编码器路径 - 4层下采样
        self.conv1 = DoubleConv(in_ch, 32)      # 第1层：输入 -> 32通道
        self.pool1 = nn.MaxPool2d(2)            # 最大池化，尺寸减半
        self.conv2 = DoubleConv(32, 64)         # 第2层：32 -> 64通道
        self.pool2 = nn.MaxPool2d(2)            # 最大池化，尺寸减半
        self.conv3 = DoubleConv(64, 128)        # 第3层：64 -> 128通道
        self.pool3 = nn.MaxPool2d(2)            # 最大池化，尺寸减半
        self.conv4 = DoubleConv(128, 256)       # 第4层：128 -> 256通道
        self.pool4 = nn.MaxPool2d(2)            # 最大池化，尺寸减半
        self.conv5 = DoubleConv(256, 512)       # 第5层：256 -> 512通道（最深层）
        
        # 解码器路径 - 4层上采样
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)    # 上采样：512 -> 256通道
        self.conv6 = DoubleConv(512, 256)                       # 双卷积（包含跳跃连接）
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)    # 上采样：256 -> 128通道
        self.conv7 = DoubleConv(256, 128)                       # 双卷积（包含跳跃连接）
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)     # 上采样：128 -> 64通道
        self.conv8 = DoubleConv(128, 64)                        # 双卷积（包含跳跃连接）
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)      # 上采样：64 -> 32通道
        self.conv9 = DoubleConv(64, 32)                         # 双卷积（包含跳跃连接）
        
        # 最终输出层
        self.conv10 = nn.Conv2d(32, out_ch, 1)  # 1x1卷积，调整到输出通道数

    def forward(self, x):
        """
        前向传播函数
        
        参数：
        - x: 输入图像，形状为(B, in_ch, H, W)
        
        返回：
        - 分割结果，形状为(B, out_ch, H, W)
        """
        # 编码器路径 - 逐步下采样并提取特征
        c1 = self.conv1(x)        # 第1层卷积
        p1 = self.pool1(c1)       # 第1层池化
        c2 = self.conv2(p1)       # 第2层卷积
        p2 = self.pool2(c2)       # 第2层池化
        c3 = self.conv3(p2)       # 第3层卷积
        p3 = self.pool3(c3)       # 第3层池化
        c4 = self.conv4(p3)       # 第4层卷积
        p4 = self.pool4(c4)       # 第4层池化
        c5 = self.conv5(p4)       # 第5层卷积（最深层特征）
        
        # 解码器路径 - 逐步上采样并融合跳跃连接
        up_6 = self.up6(c5)                           # 上采样到第4层尺寸
        merge6 = torch.cat([up_6, c4], dim=1)         # 与第4层特征拼接
        c6 = self.conv6(merge6)                       # 第6层卷积
        up_7 = self.up7(c6)                           # 上采样到第3层尺寸
        merge7 = torch.cat([up_7, c3], dim=1)         # 与第3层特征拼接
        c7 = self.conv7(merge7)                       # 第7层卷积
        up_8 = self.up8(c7)                           # 上采样到第2层尺寸
        merge8 = torch.cat([up_8, c2], dim=1)         # 与第2层特征拼接
        c8 = self.conv8(merge8)                       # 第8层卷积
        up_9 = self.up9(c8)                           # 上采样到第1层尺寸
        merge9 = torch.cat([up_9, c1], dim=1)         # 与第1层特征拼接
        c9 = self.conv9(merge9)                       # 第9层卷积
        
        # 最终输出
        c10 = self.conv10(c9)     # 最终1x1卷积
        out = nn.Sigmoid()(c10)   # Sigmoid激活
        return out


nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    """
    解码器块 - ResNet34-U-Net的解码器组件
    
    功能说明：
    - 包含1x1卷积降维、转置卷积上采样、1x1卷积升维
    - 每个操作后都有批归一化和ReLU激活
    - 用于ResNet34-U-Net的解码器路径
    """
    def __init__(self, in_channels, n_filters):
        """
        初始化解码器块
        
        参数说明：
        - in_channels: 输入通道数
        - n_filters: 输出通道数
        """
        super(DecoderBlock, self).__init__()

        # 第一个1x1卷积 - 降维到1/4
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        # 转置卷积 - 上采样并保持通道数
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        # 第二个1x1卷积 - 升维到目标通道数
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        """
        前向传播函数
        
        参数：
        - x: 输入特征图
        
        返回：
        - 经过解码器块处理的特征图
        """
        x = self.conv1(x)      # 1x1卷积降维
        x = self.norm1(x)      # 批归一化
        x = self.relu1(x)      # ReLU激活
        x = self.deconv2(x)    # 转置卷积上采样
        x = self.norm2(x)      # 批归一化
        x = self.relu2(x)      # ReLU激活
        x = self.conv3(x)      # 1x1卷积升维
        x = self.norm3(x)      # 批归一化
        x = self.relu3(x)      # ReLU激活
        return x

class resnet34_unet(nn.Module):
    """
    ResNet34-U-Net模型
    
    模型结构说明：
    1. 编码器：使用预训练的ResNet34作为特征提取器
    2. 解码器：使用自定义的解码器块进行上采样
    3. 跳跃连接：将ResNet34各层特征与解码器对应层连接
    4. 最终输出：转置卷积 + 卷积 + Sigmoid激活
    
    主要特点：
    - 结合ResNet34的强大特征提取能力
    - 使用跳跃连接保留细节信息
    - 支持预训练权重加载
    """
    def __init__(self, num_classes=1, num_channels=3,pretrained=True):
        """
        初始化ResNet34-U-Net模型
        
        参数说明：
        - num_classes: 输出类别数，默认为1（二分类）
        - num_channels: 输入通道数，默认为3（RGB）
        - pretrained: 是否使用预训练权重，默认为True
        """
        super(resnet34_unet, self).__init__()

        # 定义各层通道数
        filters = [64, 128, 256, 512]
        
        # 加载预训练的ResNet34作为编码器
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1      # 第一个卷积层
        self.firstbn = resnet.bn1          # 第一个批归一化层
        self.firstrelu = resnet.relu       # 第一个ReLU激活
        self.firstmaxpool = resnet.maxpool # 第一个最大池化层
        self.encoder1 = resnet.layer1      # ResNet第1层
        self.encoder2 = resnet.layer2      # ResNet第2层
        self.encoder3 = resnet.layer3      # ResNet第3层
        self.encoder4 = resnet.layer4      # ResNet第4层

        # 解码器层 - 使用自定义解码器块
        self.decoder4 = DecoderBlock(512, filters[2])    # 第4层解码器
        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 第3层解码器
        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 第2层解码器
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 第1层解码器

        # 最终输出层
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 最终转置卷积
        self.finalrelu1 = nonlinearity                                    # ReLU激活
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)                # 最终卷积
        self.finalrelu2 = nonlinearity                                    # ReLU激活
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)       # 输出卷积

    def forward(self, x):
        """
        前向传播函数
        
        参数：
        - x: 输入图像，形状为(B, num_channels, H, W)
        
        返回：
        - 分割结果，形状为(B, num_classes, H, W)
        """
        # 编码器路径 - 使用ResNet34提取特征
        x = self.firstconv(x)      # 第一个卷积
        x = self.firstbn(x)        # 批归一化
        x = self.firstrelu(x)      # ReLU激活
        x = self.firstmaxpool(x)   # 最大池化
        e1 = self.encoder1(x)      # ResNet第1层
        e2 = self.encoder2(e1)     # ResNet第2层
        e3 = self.encoder3(e2)     # ResNet第3层
        e4 = self.encoder4(e3)     # ResNet第4层

        # 解码器路径 - 使用跳跃连接融合特征
        d4 = self.decoder4(e4) + e3    # 第4层解码 + 跳跃连接
        d3 = self.decoder3(d4) + e2    # 第3层解码 + 跳跃连接
        d2 = self.decoder2(d3) + e1    # 第2层解码 + 跳跃连接
        d1 = self.decoder1(d2)         # 第1层解码

        # 最终输出处理
        out = self.finaldeconv1(d1)    # 最终转置卷积
        out = self.finalrelu1(out)     # ReLU激活
        out = self.finalconv2(out)     # 最终卷积
        out = self.finalrelu2(out)     # ReLU激活
        out = self.finalconv3(out)     # 输出卷积

        return nn.Sigmoid()(out)       # Sigmoid激活