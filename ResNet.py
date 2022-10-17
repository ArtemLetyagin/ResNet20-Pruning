import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k, padding, stride, skip=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, k, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        
        self.skipper = nn.Sequential()
        if skip:
            self.skipper = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skipper(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 =  ResBlock(16,16,3,1,1)
        self.layer2 =  ResBlock(16,16,3,1,1)
        self.layer3 =  ResBlock(16,16,3,1,1)
        self.layer4 =  ResBlock(16,32,3,1,2,skip=True)
        self.layer5 =  ResBlock(32,32,3,1,1)
        self.layer6 =  ResBlock(32,32,3,1,1)
        self.layer7 =  ResBlock(32,64,3,1,2,skip=True)
        self.layer8 =  ResBlock(64,64,3,1,1)
        self.layer9 =  ResBlock(64,64,3,1,1)
        self.avg = nn.AvgPool2d(8)
        self.l = nn.Linear(64,10)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.avg(out).view(-1,1,64)
        out = self.l(out)
        return out