import torch.nn as nn 
    
    
class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features, reduction_factor=2):
        super(SE_block, self).__init__()
        # squeeze block
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)
        excite_res = self.excite(squeeze_res)
        f_scale = excite_res.view(batch, channel, 1, 1)
        return x * f_scale


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  


class DoubleConv_v2(nn.Module):
    def __init__(self, c_in, c_out, hidden=None, kernel_size=3, padding=1):
        super(DoubleConv_v2, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        hidden = c_out if not hidden else hidden
        
        self.depthwise_conv1 = nn.Conv2d(c_in, c_in, kernel_size=kernel_size, padding=padding, groups=c_in)
        self.pointwise_conv1 = nn.Conv2d(c_in, hidden, kernel_size=1)
        self.norm_relu1 = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        self.depthwise_conv2 = nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, groups=hidden)
        self.pointwise_conv2 = nn.Conv2d(hidden, c_out, kernel_size=1)
        self.norm_relu2 = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        
        if c_in != c_out:
            self.projection = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        out = self.depthwise_conv1(x)
        out = self.pointwise_conv1(out)
        out = self.norm_relu1(out)

        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        out = self.norm_relu2(out)
        
        out = out + self.projection(x)
        
        return out
    