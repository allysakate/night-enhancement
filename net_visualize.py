import torch
import torch.nn as nn
import torch.nn.functional as F


class VizNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(VizNet, self).__init__()
        self.input_nc = input_nc

        self.conv1_1_lr_bn = nn.Conv2d(input_nc, 32, 3, padding=1)
        self.conv1_2_lr_bn = nn.Conv2d(32, 32, 3, padding=1)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1_lr_bn = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2_lr_bn = nn.Conv2d(64, 64, 3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1_lr_bn = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2_lr_bn = nn.Conv2d(128, 128, 3, padding=1)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1_lr_bn = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2_lr_bn = nn.Conv2d(256, 256, 3, padding=1)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1_lr_bn = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_2_lr_bn = nn.Conv2d(512, 512, 3, padding=1)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_1_lr_bn = nn.Conv2d(512, 256, 3, padding=1)
        self.conv6_2_lr_bn = nn.Conv2d(256, 256, 3, padding=1)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_1_lr_bn = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2_lr_bn = nn.Conv2d(128, 128, 3, padding=1)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_1_lr_bn = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.LReLU8_2_lr_bn = nn.LeakyReLU(0.2, inplace=True)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_1_lr_bn = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9_2_lr = nn.Conv2d(32, 32, 3, padding=1)

        self.conv10 = nn.Conv2d(32, output_nc, 1)

    # TODO: weights_init, xavier
    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         xavier(m.weight.data)
    #         xavier(m.bias.data)

    def forward(self, input):
        x = self.conv1_1_lr_bn(input)
        conv1 = self.conv1_2_lr_bn(x)
        x = self.max_pool1(conv1)

        x = self.conv2_1_lr_bn(x)
        conv2 = self.conv2_2_lr_bn(x)
        x = self.max_pool2(conv2)

        x = self.conv3_1_lr_bn(x)
        conv3 = self.conv3_2_lr_bn(x)
        x = self.max_pool3(conv3)

        x = self.conv4_1_lr_bn(x)
        conv4 = self.conv4_2_lr_bn(x)
        x = self.max_pool4(conv4)

        x = self.conv5_1_lr_bn(x)
        conv5 = self.conv5_2_lr_bn(x)

        conv5 = F.interpolate(
            conv5, scale_factor=2, mode="bilinear", align_corners=False
        )
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.conv6_1_lr_bn(up6)
        conv6 = self.conv6_2_lr_bn(x)

        conv6 = F.interpolate(
            conv6, scale_factor=2, mode="bilinear", align_corners=False
        )
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.conv7_1_lr_bn(up7)
        conv7 = self.conv7_2_lr_bn(x)

        conv7 = F.interpolate(
            conv7, scale_factor=2, mode="bilinear", align_corners=False
        )
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.conv8_1_lr_bn(up8)
        conv8 = self.conv8_2(x)

        conv8 = F.interpolate(
            conv8, scale_factor=2, mode="bilinear", align_corners=False
        )
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.conv9_1_lr_bn(up9)
        conv9 = self.conv9_2_lr(x)

        latent = self.conv10(conv9)

        return latent
