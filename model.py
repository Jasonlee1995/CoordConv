import torch
import torch.nn as nn


class ConvUniform(nn.Module):
    def __init__(self, init_weights=True):
        super(ConvUniform, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        if init_weights: self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

                    
class ConvQuadrant(nn.Module):
    def __init__(self, init_weights=True):
        super(ConvQuadrant, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.9),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.9), nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        
        if init_weights: self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
                    
                    
class CoordConv(nn.Module):
    def __init__(self, init_weights=True):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=64, stride=64)
        )
        
        if init_weights: self._initialize_weights()

    def forward(self, x):
        x = self.addcoords(x)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
                    
                    
                    
class AddCoords(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, _, H, W = x.size()

        xx_channel = torch.arange(W, device=x.device).repeat(N, 1, H, 1)
        yy_channel = torch.arange(H, device=x.device).repeat(N, 1, W, 1).transpose(2, 3)

        xx_channel = xx_channel.float() / (W - 1)
        yy_channel = yy_channel.float() / (H - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        x_with_coord = torch.cat([x, xx_channel, yy_channel], dim=1)
        return x_with_coord