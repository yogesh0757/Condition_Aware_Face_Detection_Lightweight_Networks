import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out
        
class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 8, 320, 240
            conv_dw(8, 16, 1),   # 8, 320, 240, # 16, 320, 240
            conv_dw(16, 32, 2),  # 16, 160, 120, # 32, 160, 120
            conv_dw(32, 32, 1),  # 32, 160, 120, # 32, 160, 120
            conv_dw(32, 64, 2),  # 64, 80, 60, # 64, 80, 60
            conv_dw(64, 64, 1),  # 64, 80, 60, # 64, 80, 60
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

def conv_bnr_oup(inp, oup, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=padding, groups=oup, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnr_inp(inp, oup, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=padding, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnr_oup1X1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnrDS_oup(inp, oup, kernel_size, padding):
    return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=2, padding=padding, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnrDS_inp(inp, oup, kernel_size, padding):
    return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )


class FeRIv4(nn.Module):
    def __init__(self, in_channel=32):
        super(FeRIv4,self).__init__()
        self.conv = conv_bnr_oup(in_channel, in_channel, 3, 1)
        self.conv1 = conv_bnr_oup1X1(in_channel, in_channel//2)
        self.conv2 = conv_bnr_oup(in_channel, in_channel//2, 3, 1)
        self.conv3 = conv_bnr_oup(in_channel, in_channel, 3, 1)
        
    def forward(self, input):
        x = self.conv(input)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv3(out)
        #out = self.shuffle(out)
        out = out + input
        
        return out
        
class FeRIDSv4(nn.Module):
    def __init__(self, in_channel=32):
        super(FeRIDSv4,self).__init__()
        self.conv = conv_bnrDS_oup(in_channel, in_channel, 3, 1)
        self.conv_down = conv_bnrDS_inp(in_channel, 2*in_channel, 3, 1)
        
        self.conv1 = conv_bnr_oup1X1(in_channel, in_channel//2)
        self.conv2 = conv_bnr_oup(in_channel, in_channel//2, 3, 1)
        self.conv3 = conv_bnr_inp(in_channel, 2*in_channel, 3, 1)
        
    def forward(self, input):
        down = self.conv_down(input)
        x = self.conv(input)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv3(out)
        out = out + down
        
        return out

class BBLiteV4(nn.Module):
    def __init__(self):
        super(BBLiteV4, self).__init__()
        self.conv1 = conv_bn(3, 8, 2, leaky = 0.1)
        self.conv2 = conv_dw(8, 16, 1)
        self.conv3 = conv_dw(16, 32, 2)
        
        self.stage1_1 = FeRIDSv4(32)
        self.stage1_2 = FeRIv4(64)
        self.stage1_3 = FeRIv4(64)
        self.stage1 = FeRIv4(64)

        self.stage2_1 = FeRIDSv4(64)
        self.stage2_2 = FeRIv4(128)
        self.stage2_3 = FeRIv4(128)
        self.stage2 = FeRIv4(128)

        self.stage3_1 = FeRIDSv4(128)
        self.stage3_2 = FeRIv4(256)
        self.stage3_3 = FeRIv4(256)
        self.stage3 = FeRIv4(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1_3(x)
        x = self.stage1(x)
        
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage2_3(x)
        x = self.stage2(x)
        
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
