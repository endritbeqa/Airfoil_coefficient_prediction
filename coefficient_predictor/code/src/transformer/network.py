# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn
from typing import Type
from ml_collections import ConfigDict

from src.transformer.encoder import Encoder
from src.transformer.head import Head


class VisionTransformer(nn.Module):
    config: Type[ConfigDict]
    encoder: Type[nn.Module] = Encoder
    head : Type[nn.Module] = Head

    @nn.compact
    def __call__(self, x, train):
        x = self.encoder(**self.config.vit, name='Encoder')(x, train=train)
        y = self.head(**self.config.head, name = 'Head')(x, train=train)

        return y
