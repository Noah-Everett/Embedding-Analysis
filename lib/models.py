import math
from typing import Optional, Callable

import torch
import torch.nn as nn


def _get_activation(name: Optional[str]) -> Optional[Callable[[], nn.Module]]:
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu" or name == "swish":
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    if name == "identity" or name == "none":
        return None
    raise ValueError(f"Unknown activation: {name}")


class BottleneckMLP(nn.Module):
    """
    A generic encoder -> bottleneck -> decoder MLP.

    - in_dim: input dimensionality
    - out_dim: output dimensionality
    - embed_dim: bottleneck (latent) dimensionality
    - hidden: size of hidden layers in encoder/decoder
    - enc_layers: number of hidden layers in encoder (0 means direct to bottleneck)
    - dec_layers: number of hidden layers in decoder before output layer
    - activation: activation for hidden layers (default 'gelu')
    - out_activation: optional activation for output (e.g., 'tanh' or None)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        hidden: int = 64,
        enc_layers: int = 2,
        dec_layers: int = 2,
        activation: str = "gelu",
        out_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1 or embed_dim < 1:
            raise ValueError("in_dim, out_dim, and embed_dim must be >= 1")
        if enc_layers < 0 or dec_layers < 0:
            raise ValueError("enc_layers and dec_layers must be >= 0")

        act_ctor = _get_activation(activation) or nn.Identity
        out_act_ctor = _get_activation(out_activation)

        enc_layers_list = []
        last = in_dim
        if enc_layers == 0:
            # no hidden layers before embedding
            pass
        else:
            # First hidden layer
            enc_layers_list.append(nn.Linear(last, hidden))
            enc_layers_list.append(act_ctor())
            last = hidden
            for _ in range(enc_layers - 1):
                enc_layers_list.append(nn.Linear(last, hidden))
                enc_layers_list.append(act_ctor())
                last = hidden
        self.enc = nn.Sequential(*enc_layers_list) if enc_layers_list else nn.Identity()

        # Bottleneck
        self.embed = nn.Linear(last if enc_layers > 0 else in_dim, embed_dim)

        # Decoder
        dec_layers_list = [act_ctor()]
        if dec_layers == 0:
            dec_layers_list.append(nn.Linear(embed_dim, out_dim))
        else:
            dec_layers_list.append(nn.Linear(embed_dim, hidden))
            dec_layers_list.append(act_ctor())
            for _ in range(dec_layers - 1):
                dec_layers_list.append(nn.Linear(hidden, hidden))
                dec_layers_list.append(act_ctor())
            dec_layers_list.append(nn.Linear(hidden, out_dim))
        if out_act_ctor is not None:
            dec_layers_list.append(out_act_ctor())
        self.dec = nn.Sequential(*dec_layers_list)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        h = self.enc(x)
        z = self.embed(h)
        out = self.dec(z)
        return out, z
