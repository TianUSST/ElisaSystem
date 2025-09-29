from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

class conv_block(nn.Module):
    """
    卷积块 - Attention U-Net的基本构建单元
    
    功能说明：
    - 包含两个连续的3x3卷积层
    - 每个卷积后跟批归一化和ReLU激活
    - 用于特征提取和通道数调整
    """
    def __init__(self,ch_in,ch_out):
        """
        初始化卷积块
        
        参数说明：
        - ch_in: 输入通道数
        - ch_out: 输出通道数
        """
        super(conv_block,self).__init__()
        # 双卷积序列：卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化 -> ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),    # 第一个3x3卷积
            nn.BatchNorm2d(ch_out),                                                   # 批归一化
            nn.ReLU(inplace=True),                                                    # ReLU激活
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),   # 第二个3x3卷积
            nn.BatchNorm2d(ch_out),                                                   # 批归一化
            nn.ReLU(inplace=True)                                                     # ReLU激活
        )
    def forward(self,x):
        """
        前向传播函数
        
        参数：
        - x: 输入特征图
        
        返回：
        - 经过卷积块处理的特征图
        """
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    上采样卷积块 - Attention U-Net的上采样组件
    
    功能说明：
    - 使用双线性插值进行上采样
    - 上采样后接3x3卷积进行特征调整
    - 包含批归一化和ReLU激活
    """
    def __init__(self,ch_in,ch_out):
        """
        初始化上采样卷积块
        
        参数说明：
        - ch_in: 输入通道数
        - ch_out: 输出通道数
        """
        super(up_conv,self).__init__()
        # 上采样序列：双线性插值 -> 3x3卷积 -> 批归一化 -> ReLU
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),                                              # 双线性插值上采样
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),      # 3x3卷积
		    nn.BatchNorm2d(ch_out),                                                   # 批归一化
			nn.ReLU(inplace=True)                                                     # ReLU激活
        )

    def forward(self,x):
        """
        前向传播函数
        
        参数：
        - x: 输入特征图
        
        返回：
        - 经过上采样处理的特征图
        """
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    注意力块 - Attention U-Net的核心组件
    
    功能说明：
    - 计算门控信号和局部特征的注意力权重
    - 使用1x1卷积进行特征变换
    - 通过Sigmoid生成注意力掩码
    - 对局部特征进行加权处理
    
    注意力机制原理：
    1. 将门控信号g和局部特征x分别通过1x1卷积
    2. 相加后通过ReLU激活
    3. 通过1x1卷积和Sigmoid生成注意力权重
    4. 用权重对局部特征进行加权
    """
    def __init__(self, F_g, F_l, F_int):
        """
        初始化注意力块
        
        参数说明：
        - F_g: 门控信号的通道数
        - F_l: 局部特征的通道数
        - F_int: 中间特征的通道数
        """
        super(Attention_block, self).__init__()
        
        # 门控信号处理分支 - 1x1卷积 + 批归一化
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积
            nn.BatchNorm2d(F_int)                                                   # 批归一化
        )

        # 局部特征处理分支 - 1x1卷积 + 批归一化
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # 1x1卷积
            nn.BatchNorm2d(F_int)                                                   # 批归一化
        )

        # 注意力权重生成分支 - 1x1卷积 + 批归一化 + Sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),    # 1x1卷积
            nn.BatchNorm2d(1),                                                      # 批归一化
            nn.Sigmoid()                                                            # Sigmoid激活
        )

        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        前向传播函数
        
        参数：
        - g: 门控信号（来自解码器的高层特征）
        - x: 局部特征（来自编码器的跳跃连接）
        
        返回：
        - 经过注意力加权的局部特征
        """
        # 处理门控信号 - 通过1x1卷积和批归一化
        g1 = self.W_g(g)
        # 处理局部特征 - 通过1x1卷积和批归一化
        x1 = self.W_x(x)
        # 特征融合 - 相加后通过ReLU激活
        psi = self.relu(g1 + x1)
        # 生成注意力权重 - 通过1x1卷积、批归一化和Sigmoid
        psi = self.psi(psi)
        # 应用注意力权重 - 对局部特征进行加权
        return x * psi


class AttU_Net(nn.Module):
    """
    注意力U-Net模型
    
    模型结构说明：
    1. 编码器路径：5层下采样，每层使用卷积块和最大池化
    2. 解码器路径：4层上采样，每层使用上采样卷积块
    3. 注意力机制：在跳跃连接处使用注意力块进行特征选择
    4. 最终输出：1x1卷积 + Sigmoid激活
    
    主要特点：
    - 在经典U-Net基础上引入注意力机制
    - 注意力块帮助模型关注重要的特征区域
    - 提高分割精度，特别是在细节区域
    """
    def __init__(self, img_ch=3, output_ch=1):
        """
        初始化注意力U-Net模型
        
        参数说明：
        - img_ch: 输入图像的通道数，默认为3（RGB）
        - output_ch: 输出图像的通道数，默认为1（灰度分割图）
        """
        super(AttU_Net, self).__init__()

        # 最大池化层 - 用于下采样
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器路径 - 5层卷积块
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)      # 第1层：输入 -> 64通道
        self.Conv2 = conv_block(ch_in=64, ch_out=128)         # 第2层：64 -> 128通道
        self.Conv3 = conv_block(ch_in=128, ch_out=256)        # 第3层：128 -> 256通道
        self.Conv4 = conv_block(ch_in=256, ch_out=512)        # 第4层：256 -> 512通道
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)       # 第5层：512 -> 1024通道

        # 解码器路径 - 4层上采样 + 注意力机制
        self.Up5 = up_conv(ch_in=1024, ch_out=512)            # 第5层上采样
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)  # 第5层注意力块
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)    # 第5层卷积块

        self.Up4 = up_conv(ch_in=512, ch_out=256)             # 第4层上采样
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)  # 第4层注意力块
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)     # 第4层卷积块

        self.Up3 = up_conv(ch_in=256, ch_out=128)             # 第3层上采样
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)   # 第3层注意力块
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)     # 第3层卷积块

        self.Up2 = up_conv(ch_in=128, ch_out=64)              # 第2层上采样
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)     # 第2层注意力块
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)      # 第2层卷积块

        # 最终输出层
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.sigmoid = nn.Sigmoid()                                                      # Sigmoid激活
    def forward(self, x):
        """
        前向传播函数
        
        参数：
        - x: 输入图像，形状为(B, img_ch, H, W)
        
        返回：
        - 分割结果，形状为(B, output_ch, H, W)
        """
        # 编码器路径 - 逐步下采样并提取特征
        x1 = self.Conv1(x)        # 第1层卷积

        x2 = self.Maxpool(x1)     # 第1层池化
        x2 = self.Conv2(x2)       # 第2层卷积

        x3 = self.Maxpool(x2)     # 第2层池化
        x3 = self.Conv3(x3)       # 第3层卷积

        x4 = self.Maxpool(x3)     # 第3层池化
        x4 = self.Conv4(x4)       # 第4层卷积

        x5 = self.Maxpool(x4)     # 第4层池化
        x5 = self.Conv5(x5)       # 第5层卷积（最深层特征）

        # 解码器路径 - 逐步上采样并应用注意力机制
        d5 = self.Up5(x5)                         # 第5层上采样
        x4 = self.Att5(g=d5, x=x4)               # 第5层注意力机制
        d5 = torch.cat((x4, d5), dim=1)          # 特征拼接
        d5 = self.Up_conv5(d5)                   # 第5层卷积

        d4 = self.Up4(d5)                         # 第4层上采样
        x3 = self.Att4(g=d4, x=x3)               # 第4层注意力机制
        d4 = torch.cat((x3, d4), dim=1)          # 特征拼接
        d4 = self.Up_conv4(d4)                   # 第4层卷积

        d3 = self.Up3(d4)                         # 第3层上采样
        x2 = self.Att3(g=d3, x=x2)               # 第3层注意力机制
        d3 = torch.cat((x2, d3), dim=1)          # 特征拼接
        d3 = self.Up_conv3(d3)                   # 第3层卷积

        d2 = self.Up2(d3)                         # 第2层上采样
        x1 = self.Att2(g=d2, x=x1)               # 第2层注意力机制
        d2 = torch.cat((x1, d2), dim=1)          # 特征拼接
        d2 = self.Up_conv2(d2)                   # 第2层卷积

        # 最终输出
        d1 = self.Conv_1x1(d2)    # 1x1卷积
        d1 = self.sigmoid(d1)     # Sigmoid激活

        return d1

