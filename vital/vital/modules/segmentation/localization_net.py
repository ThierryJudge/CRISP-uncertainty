from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from torchvision.ops import roi_align

from vital.utils.image.measure import Measure
from vital.utils.image.transform import resize_image


class LocalizationNet(nn.Module):
    """Generalization of the LU-Net model, initially implemented for the CAMUS dataset.

    This implementation generalizes the idea of the network beyond its U-Net roots. It makes it applicable to any
    segmentation model, given that it complies to the following interface:
        - Implemented as an ``nn.Module`` (therefore callable for making predictions)
        - Accepts the following kwargs in its constructor:
            - ``in_channels``
            - ``out_channels``
        - Provides the following attributes:
            - ``encoder``
            - ``bottleneck``
            - ``decoder``
        - When ``encoder`` is called, returns either a single tensor to be passed to the bottleneck, or a tuple of
          tensors, where the expected input for the bottleneck is the first element of the tuple.

    References:
        - Paper describing the original LU-Net model using a UNet base: http://arxiv.org/abs/2004.02043
    """

    def __init__(
        self,
        segmentation_cls: Type[nn.Module],
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        cropped_shape: Tuple[int, int],
        **segmentation_cls_kwargs,
    ):
        """Initializes class instance.

        Args:
            segmentation_cls: Class of the module to use as a base segmentation model for the LocalizationNet.
            in_shape: Shape of the input data.
            out_shape: Shape of the target data.
            cropped_shape: (H, W), Shape at which to resize RoI crops.
            data_shape: Information about the shape of the expected input and output.
            **segmentation_cls_kwargs: Arguments to initialize an instance of ``segmentation_cls``.
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.cropped_shape = cropped_shape
        self.global_segmentation_module = segmentation_cls(
            in_channels=in_shape[0], out_channels=out_shape[0], **segmentation_cls_kwargs
        )
        self.localized_segmentation_module = segmentation_cls(
            in_channels=in_shape[0], out_channels=out_shape[0], **segmentation_cls_kwargs
        )
        segmentation_module = segmentation_cls(
            in_channels=out_shape[0], out_channels=out_shape[0], **segmentation_cls_kwargs
        )
        self.segmentation_encoder = segmentation_module.encoder
        self.segmentation_bottleneck = segmentation_module.bottleneck

        # Compute forward pass with dummy data to compute the output shape of the feature extractor module
        # used by the RoI bbox module (batch_size of 2 for batchnorm)
        x = torch.rand(2, *out_shape, dtype=torch.float)
        features = self.segmentation_encoder(x)
        if isinstance(features, Tuple):  # In case of multiple tensors returned by the encoder
            features = features[0]  # Extract expected bottleneck input
        features = self.segmentation_bottleneck(features)
        self.roi_bbox_module = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(reduce(mul, features.shape[1:]), 1024)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(1024, 256)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(256, 32)),
                    ("relu3", nn.ReLU()),
                    ("bbox", nn.Linear(32, 4)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            - (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the first, global segmentation.
            - (N, 4), Coordinates of the bbox around the RoI, in (x1, y1, x2, y2) format.
            - (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the second segmentation, localized
              in the cropped RoI.
        """
        # First segmentation module: Segment input image
        # Segmentation module trained to take as input the complete image, and to predict a rough segmentation from
        # which the groundtruth segmentation's RoI can be inferred
        global_y_hat = self.global_segmentation_module(x)

        # Feature extraction and bbox module: Regress the bbox coordinates around the RoI
        # Downsampling half of segmentation module trained, in association with the following fully-connected layers,
        # to predict through regression the coordinates of the bbox around the groundtruth segmentation
        features = self.segmentation_encoder(F.softmax(global_y_hat, dim=1))
        if isinstance(features, Tuple):  # In case of multiple tensors returned by the encoder
            features = features[0]  # Extract expected bottleneck input
        features = self.segmentation_bottleneck(features)
        features = torch.flatten(features, 1)  # Format bottleneck output to match expected bbox module input
        roi_bbox_hat = self.roi_bbox_module(features)

        # Denormalize bbox while allowing gradients to flow through
        boxes = roi_bbox_hat.clone()
        boxes[:, (1, 3)] *= self.in_shape[1]
        boxes[:, (0, 2)] *= self.in_shape[2]

        # Crop and resize ``x`` based on ``roi_bbox_hat`` predicted by the previous modules
        cropped_x = roi_align(x, torch.split(boxes, 1), self.cropped_shape, aligned=True)

        # Second segmentation module: Segment cropped RoI
        # Segmentation module trained to take as input the image cropped around the predicted segmentation's RoI, and
        # to predict a highly accurate segmentation from the localised input
        localized_y_hat = self.localized_segmentation_module(cropped_x)

        return global_y_hat, roi_bbox_hat, localized_y_hat

    def predict(self, x: Tensor) -> Tensor:
        """Performs test-time inference on the input.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Input's segmentation, in one-hot format.
        """
        _, roi_bbox_hat, localized_y_hat = self(x)
        y_hat = self._revert_crop(localized_y_hat.argmax(dim=1, keepdim=True), roi_bbox_hat)  # (N, 1, H, W)
        y_hat = to_onehot(y_hat.squeeze(dim=1), num_classes=self.out_shape[0])  # (N, ``out_channels``, H, W)
        return y_hat

    def _revert_crop(self, localized_segmentation: Tensor, roi_bbox: Tensor) -> Tensor:
        """Fits the localized segmentation back to its original position the image.

        Args:
            localized_segmentation: (N, 1, H, W), Segmentation of the content of the bbox around the RoI.
            roi_bbox: (N, 4), Normalized coordinates of the bbox around the RoI, in (x1, y1, x2, y2) format.

        Returns:
            (N, 1, H, W), Localized segmentation fitted to its original position in the image.
        """
        roi_bbox = Measure.denormalize_bbox(roi_bbox, self.out_shape[1:][::-1], check_bounds=True).int()

        # Fit the localized segmentation at its original location in the image, one item at a time
        segmentation = []
        for item_roi_bbox, item_localized_seg in zip(roi_bbox, localized_segmentation):
            # Get bbox size in order (width, height)
            bbox_size = (item_roi_bbox[2] - item_roi_bbox[0], item_roi_bbox[3] - item_roi_bbox[1])

            # Convert segmentation tensor to array (compatible with PIL) to resize, then convert back to tensor
            pil_formatted_localized_seg = item_localized_seg.detach().byte().cpu().numpy().squeeze()
            item_resized_seg = torch.from_numpy(resize_image(pil_formatted_localized_seg, bbox_size)).unsqueeze(0)

            # Place the resized localised segmentation inside an empty segmentation
            segmentation.append(torch.zeros_like(item_localized_seg))
            segmentation[-1][
                :, item_roi_bbox[1] : item_roi_bbox[3], item_roi_bbox[0] : item_roi_bbox[2]
            ] = item_resized_seg

        return torch.stack(segmentation)
