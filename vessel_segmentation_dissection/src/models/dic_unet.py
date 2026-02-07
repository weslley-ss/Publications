import torch
import torch.nn as nn
import torch.nn.functional as F


class dic_unet(nn.Module):
    def __init__(self, parametros):
        super(dic_unet, self).__init__()

        self.n_classes = 1
        # FIRST
        self.first_residual = nn.Sequential(
            nn.Conv2d(**parametros["FR"]),
            nn.BatchNorm2d(parametros["FR"]["out_channels"])
        )

        self.first_conv = nn.Sequential(
            nn.Conv2d(**parametros["FC"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["FC"][0]["out_channels"]),
            nn.Conv2d(**parametros["FC"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["FC"][1]["out_channels"])
        )

        # DOWN PATH
        if parametros["D1"]["kernel_size"] == 2 and parametros["UP2"]["kernel_size"] == 2:
            self.down_pooling1 = nn.MaxPool2d(**parametros["D1"])
        else: 
            self.down_pooling1 = nn.Identity()

        self.block1_residual = nn.Sequential(
            nn.Conv2d(**parametros["B1R"]),
            nn.BatchNorm2d(parametros["B1R"]["out_channels"])
        )
        self.block1_conv = nn.Sequential(
            nn.Conv2d(**parametros["B1C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["B1C"][0]["out_channels"]),
            nn.Conv2d(**parametros["B1C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["B1C"][1]["out_channels"])
        )

        if parametros["D2"]["kernel_size"] == 2 and parametros["UP1"]["kernel_size"] == 2:
            self.down_pooling2 = nn.MaxPool2d(**parametros["D2"])
        else:
            self.down_pooling2 = nn.Identity()
        self.block2_residual = nn.Sequential(
            nn.Conv2d(**parametros["B2R"]),
            nn.BatchNorm2d(parametros["B2R"]["out_channels"])
        )
        self.block2_conv = nn.Sequential(
            nn.Conv2d(**parametros["B2C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["B2C"][0]["out_channels"]),
            nn.Conv2d(**parametros["B2C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["B2C"][1]["out_channels"])
        )

        # UP PATH
        if parametros["D2"]["kernel_size"] == 2 and parametros["UP1"]["kernel_size"] == 2:
            self.up_layer1 = nn.ConvTranspose2d(**parametros["UP1"])
        else: 
            parametros_aux = {
                                "in_channels": parametros["B2C"][1]["out_channels"],
                                "out_channels": parametros["B1C"][1]["out_channels"],
                                "kernel_size": 1,
                                "padding": 0,
                                "dilation": 1
                            }
            self.up_layer1 = nn.Conv2d(**parametros_aux)

        self.conv_bridge1 = nn.Sequential(
            nn.Conv2d(**parametros["CB1"]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["CB1"]["out_channels"])
        )
        self.up_block1_residual = nn.Sequential(
            nn.Conv2d(**parametros["UB1R"]),
            nn.BatchNorm2d(parametros["UB1R"]["out_channels"])
        )
        self.up_block1_conv = nn.Sequential(
            nn.Conv2d(**parametros["UB1C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["UB1C"][0]["out_channels"]),
            nn.Conv2d(**parametros["UB1C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["UB1C"][1]["out_channels"])
        )

        
        # UP PATH
        if parametros["D1"]["kernel_size"] == 2 or parametros["UP2"]["kernel_size"] == 2:
            self.up_layer2 = nn.ConvTranspose2d(**parametros["UP2"])          
        else: 
            parametros_aux = {
                                "in_channels": parametros["UB1C"][1]["out_channels"],
                                "out_channels": parametros["FC"][1]["out_channels"],
                                "kernel_size": 1,
                                "padding": 0,
                                "dilation": 1
                            }
            self.up_layer2 = nn.Conv2d(**parametros_aux)

        self.conv_bridge2 = nn.Sequential(
            nn.Conv2d(**parametros["CB2"]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["CB2"]["out_channels"])
        )
        self.up_block2_residual = nn.Sequential(
            nn.Conv2d(**parametros["UB2R"]),
            nn.BatchNorm2d(parametros["UB2R"]["out_channels"])
        )
        self.up_block2_conv = nn.Sequential(
            nn.Conv2d(**parametros["UB2C"][0]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["UB2C"][0]["out_channels"]),
            nn.Conv2d(**parametros["UB2C"][1]),
            nn.ReLU(),
            nn.BatchNorm2d(parametros["UB2C"][1]["out_channels"])
        )

        self.conv_final = nn.Conv2d(**parametros["F"])

    def forward(self, x):
        # FIRST
        first_residual = self.first_residual(x)
        first_conv = self.first_conv(x)
        out_first = first_conv + first_residual

        # BLOCK 1
        outpooling1 = self.down_pooling1(out_first)
        block1_residual = self.block1_residual(outpooling1)
        block1_conv = self.block1_conv(outpooling1)
        out_block1 = block1_conv + block1_residual

        # BLOCK 2
        outpooling2 = self.down_pooling2(out_block1)
        block2_residual = self.block2_residual(outpooling2)
        block2_conv = self.block2_conv(outpooling2)
        out_block2 = block2_conv + block2_residual

        # UP BLOCK 1
        out_uplayer1 = self.up_layer1(out_block2)
        out_convBridge1 = self.conv_bridge1(out_block1)
        out_Concat1 = torch.cat([out_uplayer1, out_convBridge1], dim=1)
        up_block1_residual = self.up_block1_residual(out_Concat1)
        up_block1_conv = self.up_block1_conv(out_Concat1)
        out_up_block1 = up_block1_conv + up_block1_residual

        # UP BLOCK 2
        out_uplayer2 = self.up_layer2(out_up_block1)
        out_convBridge2 = self.conv_bridge2(out_first)
        out_Concat2 = torch.cat([out_uplayer2, out_convBridge2], dim=1)
        up_block2_residual = self.up_block2_residual(out_Concat2)
        up_block2_conv = self.up_block2_conv(out_Concat2)
        out_up_block2 = up_block2_conv + up_block2_residual

        final = self.conv_final(out_up_block2)
        return final
