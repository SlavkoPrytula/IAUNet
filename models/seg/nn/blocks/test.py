import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv_v1(nn.Module):
    def __init__(self, c_in, c_out, hidden=None, kernel_size=3, padding=1):
        super(DoubleConv_v1, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        hidden = c_out if not hidden else hidden

        self.layer_1 = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(hidden, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        if c_in != c_out:
            self.projection = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)

        out = out + self.projection(x)
        return out




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
    
    


class DoubleConv_v3(nn.Module):
    def __init__(self, c_in, c_out, hidden=None, kernel_size=3, padding=1):
        super(DoubleConv_v3, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        hidden = c_out if not hidden else hidden
        
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    

class DoubleConv_v3_1(nn.Module):
    def __init__(self, c_in, c_out, hidden=None):
        super(DoubleConv_v3_1, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        hidden = c_out if hidden is None else hidden

        self.depthwise_conv1 = nn.Conv2d(c_in, c_in, kernel_size=7, padding=3, groups=c_in)
        self.pointwise_conv1 = nn.Conv2d(c_in, hidden, kernel_size=1)
        self.norm_relu1 = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.depthwise_conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.pointwise_conv2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.norm_relu2 = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.depthwise_conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.pointwise_conv3 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.norm_relu3 = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.depthwise_conv4 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.pointwise_conv4 = nn.Conv2d(hidden, c_out, kernel_size=1)
        self.norm_relu4 = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        out = self.depthwise_conv1(x)
        out = self.pointwise_conv1(out)
        out = self.norm_relu1(out)

        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        out = self.norm_relu2(out)

        out = self.depthwise_conv3(out)
        out = self.pointwise_conv3(out)
        out = self.norm_relu3(out)

        out = self.depthwise_conv4(out)
        out = self.pointwise_conv4(out)
        out = self.norm_relu4(out)

        return out

    
    

class DoubleConv_v4(nn.Module):
    def __init__(self, c_in, c_out, hidden=None, kernel_size=3, stride=1, padding=1):
        super(DoubleConv_v4, self).__init__()
        hidden = c_out if not hidden else hidden

        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x   


class Residual(nn.Module):
    def __init__(self, block):
        super(Residual, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class DoubleConvModule(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, depth=1):
        super(DoubleConvModule, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        
        self.layers = nn.ModuleList([])
        for i in range(depth-1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        x = self.projection(x)

        for layer in self.layers:
            x = layer(x)

        return x


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], out_channels=64):
        super(PyramidPooling, self).__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
                # nn.BatchNorm2d(out_channels),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        self.out_channels = out_channels * len(pool_sizes) + in_channels

    def forward(self, x):
        input_size = x.size()[2:]
        pooled_features = [F.interpolate(pool(x), size=input_size, mode='bilinear', align_corners=False) 
                           for pool in self.pools]
        pooled_features = torch.cat([x] + pooled_features, dim=1)
        return pooled_features


class PyramidPooling_v2(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPooling_v2, self).__init__()
        self.pool_sizes = pool_sizes
        self.pyramid_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for pool_size in self.pool_sizes:
            pool = nn.AdaptiveAvgPool2d(output_size=pool_size)
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
            self.pyramid_layers.append(pool)
            self.conv_layers.append(conv)

        # Final convolution layer to combine the pyramid outputs with the original features
        self.final_conv = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[2:]
        pyramid_outputs = [x]  # Start with original features for skip connection

        for pool, conv in zip(self.pyramid_layers, self.conv_layers):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=original_size, mode='bilinear', align_corners=True)
            pyramid_outputs.append(upsampled)

        combined = torch.cat(pyramid_outputs, dim=1)
        output = self.final_conv(combined)

        return output
    

class PyramidPooling_v3(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPooling_v3, self).__init__()
        self.pool_sizes = pool_sizes
        self.pyramid_layers = nn.ModuleList()
        self.expand = 4

        for pool_size in self.pool_sizes:
            layers = [
                nn.AdaptiveAvgPool2d(output_size=pool_size),

                # Depthwise separable convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels * self.expand, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=out_channels * self.expand),
                nn.ReLU(inplace=True),

                # Additional dilated convolution for expanded receptive field
                nn.Conv2d(out_channels * self.expand, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            ]
            self.pyramid_layers.append(nn.Sequential(*layers))

        self.combine_conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        original_size = x.size()[2:]
        pyramid_outputs = [x]

        for layer in self.pyramid_layers:
            conv_out = layer(x)
            print(conv_out.shape)
            upsampled = F.interpolate(conv_out, size=original_size, mode='bilinear', align_corners=True)
            pyramid_outputs.append(upsampled)

        combined = torch.cat(pyramid_outputs, dim=1)
        output = self.combine_conv(combined)
        return output



class PyramidPooling_v3_1(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPooling_v3_1, self).__init__()
        self.pool_sizes = pool_sizes
        self.pyramid_layers = nn.ModuleList()

        for pool_size in self.pool_sizes:
            self.pyramid_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            ))

        self.final_conv = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[2:]
        pyramid_outputs = [x]

        for layer in self.pyramid_layers:
            pooled_features = layer(x)
            upsampled_features = F.interpolate(pooled_features, size=original_size, mode='bilinear', align_corners=True)
            pyramid_outputs.append(upsampled_features)

        concatenated_features = torch.cat(pyramid_outputs, dim=1)
        output = self.final_conv(concatenated_features)
        return output
    


class PyramidPooling_v4(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPooling_v4, self).__init__()
        self.pool_sizes = pool_sizes
        self.pyramid_layers = []

        for pool_size in self.pool_sizes:
            # depthwise separable convolution
            block1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool_size),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )

            # dilated convolution + skip connection from block1's output
            block2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )

            self.pyramid_layers.append([block1, block2])

        self.combine_conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        original_size = x.size()[2:]
        pyramid_outputs = [x]

        for block1, block2 in self.pyramid_layers:
            x_pooled = block1(x)
            x_block1_out = x_pooled

            x_block2_out = block2(x_block1_out) + x_block1_out

            upsampled = F.interpolate(x_block2_out, size=original_size, mode='bilinear', align_corners=True)
            pyramid_outputs.append(upsampled)

        combined = torch.cat(pyramid_outputs, dim=1)
        output = self.combine_conv(combined)
        return output




class PyramidPooling_v5(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels, expand=1):
        super(PyramidPooling_v5, self).__init__()
        self.scales = pool_sizes
        self.pyramid_layers = nn.ModuleList()
        self.expand = expand

        for scale in self.scales:
            layers = [
                # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
                # nn.Conv2d(in_channels, out_channels * self.expand, kernel_size=1),

                nn.Conv2d(in_channels, out_channels * self.expand, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels * self.expand),
                nn.ReLU(inplace=True),

                
                # nn.Conv2d(out_channels * self.expand, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            ]
            self.pyramid_layers.append(nn.Sequential(*layers))
            # self.pyramid_layers.append(ASPP(in_channels=in_channels, 
            #                                 out_channels=out_channels * self.expand, 
            #                                 dilation_rates=[1, 6, 12, 18]))

        self.combine_conv = nn.Sequential(
            nn.Conv2d(len(pool_sizes) * out_channels * self.expand, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        original_size = x.size()[2:]
        pyramid_outputs = []

        for i, layer in enumerate(self.pyramid_layers):
            output_size = (max(1, original_size[0] // self.scales[i]), max(1, original_size[1] // self.scales[i]))
            pooled = F.adaptive_avg_pool2d(x, output_size=output_size)
            conv_out = layer(pooled)
            upsampled = F.interpolate(conv_out, size=original_size, mode='bilinear', align_corners=True)
            pyramid_outputs.append(upsampled)

        combined = torch.cat(pyramid_outputs, dim=1)
        output = self.combine_conv(combined)
        return output
    



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        
        for rate in dilation_rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate, groups=in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.concat_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        aspp_features = [block(x) for block in self.aspp_blocks]
        concatenated = torch.cat(aspp_features, dim=1)
        result = self.concat_conv(concatenated)
        
        return result



if __name__ == "__main__":
    in_channels = 64
    pool_sizes = [1, 2, 4, 8, 16]
    out_channels = 128

    # block = PyramidPooling_v5(in_channels=in_channels, pool_sizes=pool_sizes, out_channels=out_channels, expand=2)
    block = DoubleConvModule(in_channels, out_channels, depth=2)
    # block = ASPP(in_channels=in_channels, out_channels=out_channels, dilation_rates=[1, 6, 12, 18])
    x = torch.rand(1, 64, 128, 128)
    out = block(x)
    print(out.shape)

    from thop import profile
    ops, params = profile(block, inputs=(x,), verbose=True)
    print(params)
