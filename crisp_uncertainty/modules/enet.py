from typing import Tuple, Union

import vital.modules.segmentation.enet as vital_enet
from torch import Tensor, nn


class Enet(vital_enet.Enet):
    """Implementation of the ENet model.

    References:
        - Paper that introduced the model: http://arxiv.org/abs/1606.02147
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        init_channels: int = 16,
        dropout: float = 0.1,
        encoder_relu: bool = True,
        decoder_relu: bool = True,
        sigma_out: bool = False,
        conf_out: bool = False,
        confidnet: bool = False,
    ):
        """Initializes class instance.

        Args:
            input_shape: Shape of the input images.
            output_shape: Shape of the output segmentation map.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
                NOTE: In the initial block, the dropout probability is divided by 10.
            encoder_relu: When ``True`` ReLU is used as the activation function in the encoder blocks/layers; otherwise,
                PReLU is used.
            decoder_relu: When ``True`` ReLU is used as the activation function in the decoder blocks/layers; otherwise,
                PReLU is used.
            sigma_out: When ``True`` second output is added to predict variance.
            conf_out: When ``True`` second output is added to predict confidence.
            confidnet: When ``True`` second output is added to predict confidnet output.
        """
        super().__init__(input_shape, output_shape, init_channels, dropout, encoder_relu, decoder_relu)
        out_channels = output_shape[0]
        self.sigma_out = sigma_out
        self.conf_out = conf_out
        self.confidnet = confidnet

        assert not (sigma_out and conf_out)

        if self.sigma_out:
            self.regular5_1_sigma = vital_enet._RegularBottleneck(
                init_channels, padding=1, dropout=dropout, relu=decoder_relu
            )

            self.transposed_conv_sigma = nn.Sequential(
                nn.ConvTranspose2d(
                    init_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.Softplus(),
            )

        if self.conf_out:
            self.regular_conf = vital_enet._RegularBottleneck(
                init_channels, padding=1, dropout=dropout, relu=decoder_relu
            )

            self.transposed_conv_conf = nn.ConvTranspose2d(
                init_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            )

        if self.confidnet:
            # Stage 3 - Encoder
            self.regular3_0_conf = vital_enet._RegularBottleneck(
                init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
            )
            self.dilated3_1_conf = vital_enet._RegularBottleneck(
                init_channels * 4, dilation=2, padding=2, dropout=dropout, relu=encoder_relu
            )
            self.asymmetric3_2_conf = vital_enet._RegularBottleneck(
                init_channels * 4, kernel_size=5, padding=2, asymmetric=True, dropout=dropout, relu=encoder_relu
            )
            self.dilated3_3_conf = vital_enet._RegularBottleneck(
                init_channels * 4, dilation=4, padding=4, dropout=dropout, relu=encoder_relu
            )
            self.regular3_4_conf = vital_enet._RegularBottleneck(
                init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
            )
            self.dilated3_5_conf = vital_enet._RegularBottleneck(
                init_channels * 4, dilation=8, padding=8, dropout=dropout, relu=encoder_relu
            )
            self.asymmetric3_6_conf = vital_enet._RegularBottleneck(
                init_channels * 4, kernel_size=5, asymmetric=True, padding=2, dropout=dropout, relu=encoder_relu
            )
            self.dilated3_7_conf = vital_enet._RegularBottleneck(
                init_channels * 4, dilation=16, padding=16, dropout=dropout, relu=encoder_relu
            )

            # Stage 4 - Decoder
            self.upsample4_0_conf = vital_enet._UpsamplingBottleneck(
                init_channels * 4, init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu
            )
            self.regular4_1_conf = vital_enet._RegularBottleneck(
                init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu
            )
            self.regular4_2_conf = vital_enet._RegularBottleneck(
                init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu
            )

            # Stage 5 - Decoder
            self.upsample5_0_conf = vital_enet._UpsamplingBottleneck(
                init_channels * 2, init_channels, padding=1, dropout=dropout, relu=decoder_relu
            )
            self.regular5_1_conf = vital_enet._RegularBottleneck(
                init_channels, padding=1, dropout=dropout, relu=decoder_relu
            )

            self.transposed_conv_conf = nn.ConvTranspose2d(
                init_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            )

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        confid_split = x

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Decoder
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        sigma_split = x
        conf_split = x
        x = self.regular5_1(x)

        if self.sigma_out:
            sigma = self.regular5_1_sigma(sigma_split)
            return self.transposed_conv(x), self.transposed_conv_sigma(sigma)
        elif self.conf_out:
            conf = self.regular_conf(conf_split)
            return self.transposed_conv(x), self.transposed_conv_conf(conf)
        elif self.confidnet:
            """Confidence Branch"""
            # Stage 3 - Encoder
            x_conf = self.regular3_0_conf(confid_split)
            x_conf = self.dilated3_1_conf(x_conf)
            x_conf = self.asymmetric3_2_conf(x_conf)
            x_conf = self.dilated3_3_conf(x_conf)
            x_conf = self.regular3_4_conf(x_conf)
            x_conf = self.dilated3_5_conf(x_conf)
            x_conf = self.asymmetric3_6_conf(x_conf)
            x_conf = self.dilated3_7_conf(x_conf)

            # Stage 4 - Decoder
            x_conf = self.upsample4_0_conf(x_conf, max_indices2_0)
            x_conf = self.regular4_1_conf(x_conf)
            x_conf = self.regular4_2_conf(x_conf)

            # Stage 5 - Decoder
            x_conf = self.upsample5_0_conf(x_conf, max_indices1_0)
            x_conf = self.regular5_1_conf(x_conf)
            return self.transposed_conv(x), self.transposed_conv_conf(x_conf)
        else:
            return self.transposed_conv(x)
