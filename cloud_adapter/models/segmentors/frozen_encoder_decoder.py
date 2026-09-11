from typing import List
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class FrozenBackboneEncoderDecoder(EncoderDecoder):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x

@MODELS.register_module()
class FrozenHeadEncoderDecoder(EncoderDecoder):
    def train(self, mode=True):
        super().train(mode)

        self.decode_head.eval()
        for param in self.decode_head.parameters():
            param.requires_grad = False


@MODELS.register_module()
class FrequencyRoutedEncoderDecoder(EncoderDecoder):
    """EncoderDecoder that includes the adapter's load-balancing objective."""

    def loss(self, inputs: Tensor, data_samples: List[dict]) -> dict:
        losses = super().loss(inputs, data_samples)
        adapter = getattr(self.backbone, "cloud_adapter", None)
        if adapter is not None:
            balance_loss = adapter.get_balance_loss()
            if balance_loss is not None:
                losses["loss_frequency_balance"] = balance_loss
        return losses
