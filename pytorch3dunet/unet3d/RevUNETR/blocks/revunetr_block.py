# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn

import revtorch.revtorch as rv
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class RevUnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetRevBlock(
                                spatial_dims=spatial_dims,
                                channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetRevBlock(nn.Module):
    """
    A Reversible block module that can be used for DynUNet
    Args:
        spatial_dims: number of spatial dimensions.
        channels: number of channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.rev_block = makeReversibleUnetBlock(spatial_dims,
                                                 channels,
                                                 kernel_size,
                                                 stride,
                                                 dropout,
                                                 3,
                                                 norm_name,
                                                 act_name)

    def forward(self, inp):
        out = self.rev_block(inp)
        return out


class ResidualInner(nn.Module):
    def __init__(self, spatial_dims,
                            channels,
                            kernel_size,
                            stride,
                            dropout,
                            norm_name,
                            act_name):
        super(ResidualInner, self).__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.norm = get_norm_layer(name=norm_name,
                                    spatial_dims=spatial_dims,
                                    channels=channels)
        self.act = get_act_layer(name=act_name)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
def makeReversibleUnetBlock(spatial_dims,
                            channels,
                            kernel_size,
                            stride,
                            dropout,
                            blockCount=3,
                            norm_name="instance",
                            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})):
    innerChannels = channels // 2
    modules = []
    for i in range(blockCount):
        fBlock = ResidualInner(spatial_dims,
                               innerChannels,
                               kernel_size,
                               stride,
                               dropout,
                               norm_name,
                               act_name)
        gBlock = ResidualInner(spatial_dims,
                               innerChannels,
                               kernel_size,
                               stride,
                               dropout,
                               norm_name,
                               act_name)
        modules.append(rv.ReversibleBlock(fBlock, gBlock))
    return rv.ReversibleSequence(nn.ModuleList(modules))