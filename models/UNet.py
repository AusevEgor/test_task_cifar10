import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        return self.mp(self.conv_block(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=2,
                                         stride=2)
        self.conv_block = ConvBlock(in_channels = in_channels,
                                    out_channels= out_channels)

    def forward(self, x, skip_feats=None):

        con = self.conv_t(x)
        # if skip_feats != None:
        #     con = self.conv_t(x)
        #     new_x = torch.cat((skip_feats, con), 1)
        # else:
        #     new_x = F.upsample(x, scale_factor=2)
        out = self.conv_block(con)
        return out

class Unet(nn.Module):
    def __init__(self, size_model):
        super(Unet, self).__init__()

        # Encoder ______________________________________

        self.en1 = EncoderBlock(3, size_model)                  # 16x16
        self.en2 = EncoderBlock(size_model, size_model *2)      # 8x8
        self.en3 = EncoderBlock(size_model *2, size_model *4)   # 4x4
        self.en4 = EncoderBlock(size_model *4, size_model *8)   # 2x2
        self.en5 = EncoderBlock(size_model *8, size_model *16)  # 1x1

        # Decoder ______________________________________

        self.dec1 = DecoderBlock(size_model *16, size_model *8) # 2x2
        self.dec2 = DecoderBlock(size_model *8, size_model *4)  # 4x4
        self.dec3 = DecoderBlock(size_model *4, size_model *2)  # 8x8
        self.dec4 = DecoderBlock(size_model *2, size_model)     # 16x16
        self.dec5 = DecoderBlock(size_model, 3)                 # 32x32
        self.last = nn.Conv2d(3,3, kernel_size=3, padding=1)

    def forward(self, x):

        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)

        d1 = self.dec1(e5, e4)
        d2 = self.dec2(d1, e3)
        d3 = self.dec3(d2, e2)
        d4 = self.dec4(d3, e1)
        out = self.dec5(d4)

        return F.sigmoid(self.last(out))

    def embedings(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)


        return e5