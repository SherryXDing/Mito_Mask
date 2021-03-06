import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two 3D convolutional layer: 3D conv + batch norm + ReLu 
    """
    def __init__(self, in_ch, out_ch, pad=0, bias=False):
        """
        Args:
        in_ch: number of input channels
        out_ch: number of output channels
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=pad, bias=bias),
            nn.BatchNorm3d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=pad, bias=bias),
            nn.BatchNorm3d(num_features=out_ch),
            nn.ReLU(inplace=True))

    def forward(self, img):
        out = self.conv_block(img)
        return out


class UpBlock(nn.Module):
    """
    Up block: Up-conv + crop + ConvBlock
    """
    def __init__(self, in_ch, out_ch, pad=0, bias=False):
        """
        Args:
        in_ch: number of input channels
        out_ch: number of output channels
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_ch=in_ch, out_ch=out_ch, pad=pad, bias=bias)

    def forward(self, img, down_layer):
        """
        Args:
        down_layer: layer in down-sampling to crop from
        """
        up = self.upsample(img)
        # Cropping down_layer
        _, _, h, w, d = down_layer.shape
        target_sz = up.shape[2:]
        low_x = (h-target_sz[0]) // 2
        high_x = low_x + target_sz[0]
        low_y = (w-target_sz[1]) // 2
        high_y = low_y + target_sz[1]
        low_z = (d-target_sz[2]) // 2
        high_z = low_z + target_sz[2]
        cropped = down_layer[:, :, low_x:high_x, low_y:high_y, low_z:high_z]
        out = torch.cat([cropped, up], dim=1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    """
    3D UNet
    """
    def __init__(self, in_channels=1, base_filters=16, out_channels=2, depth=3, pad=0, bias=False):
        """
        Args:
        in_channels: number of input channels
        base_filters: numer of base filters
        out_channels: number of output channels
        depth: UNet depth
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.down_modules = nn.ModuleList()
        for i in range(depth):
            self.down_modules.append(
                ConvBlock(in_ch=in_channels, out_ch=2**i*base_filters, pad=pad, bias=bias))
            in_channels = 2**i*base_filters
        self.up_modules = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_modules.append(
                UpBlock(in_ch=in_channels, out_ch=2**i*base_filters, pad=0, bias=bias))
            in_channels = 2**i*base_filters
        self.final_module = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, img):
        down_layers = []
        for i, down in enumerate(self.down_modules):
            img = down(img)
            if i != len(self.down_modules) -1:
                down_layers.append(img)
                img = F.max_pool3d(img, 2)
        for i, up in enumerate(self.up_modules):
            img = up(img, down_layers[-1-i])
        img = self.final_module(img)
        img = F.softmax(img, dim=1)
        return img




class GoogleAlexNet(nn.Module):
    """
    AlexNet style network described in Google's email:
    We used these labels to train a 3d convolutional network that receives as input a field of view of 65x65x65 voxels at [16 nm]^3/voxel resolution. 
    The network uses ‘valid’ convolution padding and ‘max’ pooling operations with a kernel and striding shape of 2x2x2, 
    with convolution and pooling operations interleaved in the following sequence: 
        convolution with 64 features maps and a 3x3x3 kernel shape, max-pooling, 
        convolution with 64 feature maps, max-pooling, 
        convolution with 64 feature maps, max-pooling, 
        convolution with 3x3x3 kernel size and 16 feature maps, 
        convolution with 4x4x4 kernel shape 512 feature maps (i.e., fully connected layer), 
        and finally a logistic layer output with 8 units (first unit was unused in the labeling scheme).
    """
    def __init__(self, in_channels=1, num_classes=8):
        """
        Args:
        in_channels: number of input channels
        num_classes: number of classes
        """
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, img):
        img = self.convs(img)
        img = torch.flatten(img, 1)
        img = self.classifier(img)
        return img


if __name__ == "__main__":
    model = UNet()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
