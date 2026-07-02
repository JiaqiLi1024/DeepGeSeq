"""Enformer model and supporting building blocks.

This module contains a PyTorch implementation of Enformer-style sequence
modeling with relative positional attention and optional checkpointed
transformer execution.

Input conventions:
    - Preferred tensor shape: `(batch, sequence_length, 4)` with one-hot DNA.
    - Long-index sequence tensors are converted internally to one-hot encoding.
"""
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

# constants

SEQUENCE_LENGTH = 128_000
TARGET_LENGTH = 1000

# helpers

def exists(val):
    """Return whether a value is not `None`."""
    return val is not None

def default(val, d):
    """Return `val` when present, otherwise return default `d`."""
    return val if exists(val) else d

def map_values(fn, d):
    """Apply `fn` to each value in dictionary `d`."""
    return {key: fn(values) for key, values in d.items()}

def rearrange(tensor, pattern, **axes):
    """Handle the small set of rearrangements used by this module.

    The original implementation used ``einops``. Keeping these local tensor
    transforms avoids making Enformer import fail in lean DGS installations.
    """
    if pattern == 'b d (n p) -> b d n p':
        p = axes['p']
        b, d, length = tensor.shape
        if length % p != 0:
            raise ValueError("sequence length must be divisible by p.")
        return tensor.reshape(b, d, length // p, p)
    if pattern == 'b n (h d) -> b h n d':
        h = axes['h']
        b, n, hd = tensor.shape
        if hd % h != 0:
            raise ValueError("last dimension must be divisible by h.")
        return tensor.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
    if pattern == 'n (h d) -> h n d':
        h = axes['h']
        n, hd = tensor.shape
        if hd % h != 0:
            raise ValueError("last dimension must be divisible by h.")
        return tensor.reshape(n, h, hd // h).permute(1, 0, 2)
    if pattern == 'b h n d -> b n (h d)':
        b, h, n, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).contiguous().reshape(b, n, h * d)
    if pattern == '... -> () ...':
        return tensor.unsqueeze(0)
    if pattern == '() ... -> ...':
        if tensor.shape[0] != 1:
            raise ValueError("first dimension must be singleton.")
        return tensor.squeeze(0)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


class Rearrange(nn.Module):
    """Minimal module form for the rearrangements used by Enformer."""

    def __init__(self, pattern, **axes):
        super().__init__()
        self.pattern = pattern
        self.axes = axes

    def forward(self, tensor):
        """Apply the configured tensor rearrangement."""
        if self.pattern == 'b n d -> b d n':
            return tensor.permute(0, 2, 1).contiguous()
        if self.pattern == 'b d n -> b n d':
            return tensor.permute(0, 2, 1).contiguous()
        return rearrange(tensor, self.pattern, **self.axes)

def exponential_linspace_int(start, end, num, divisible_by = 1):
    """Create an exponentially spaced integer sequence.

    Args:
        start: Start value.
        end: End value.
        num: Number of generated values.
        divisible_by: Round each value to a multiple of this number.

    Returns:
        List of integer values.
    """
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def str_to_one_hot(sequences):
    """Convert DNA strings to channel-last one-hot tensors."""
    if isinstance(sequences, str):
        sequences = [sequences]
    if not sequences:
        raise ValueError("sequences must contain at least one DNA string.")
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError("All sequences must have the same length.")

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = torch.zeros((len(sequences), seq_len, 4), dtype=torch.float32)
    for row, sequence in enumerate(sequences):
        for col, base in enumerate(sequence.upper()):
            idx = mapping.get(base)
            if idx is not None:
                encoded[row, col, idx] = 1.0
    return encoded

def seq_indices_to_one_hot(indices):
    """Convert integer DNA indices to channel-last one-hot tensors.

    Values 0..3 map to A/C/G/T. Other values are treated as unknown bases and
    encoded as all-zero rows.
    """
    mask = (indices >= 0) & (indices < 4)
    clamped = indices.clamp(min=0, max=3)
    one_hot = F.one_hot(clamped, num_classes=4).to(dtype=torch.float32)
    return one_hot * mask.unsqueeze(-1).to(dtype=torch.float32)

def log(t, eps = 1e-20):
    """Numerically stable logarithm with lower clamp."""
    return torch.log(t.clamp(min = eps))

# losses and metrics

def poisson_loss(pred, target):
    """Compute Poisson loss used by Enformer output heads."""
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    """Compute mean Pearson correlation coefficient over selected dimensions."""
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

# relative positional encoding functions

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    """Generate exponential relative-position features."""
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    """Generate central-mask relative-position features."""
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    """Evaluate Gamma probability density values."""
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    """Generate gamma-distribution relative-position features."""
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    """Build concatenated relative positional embeddings.

    Args:
        seq_len: Sequence length used by attention.
        feature_size: Number of positional feature channels.
        device: Torch device used for tensor creation.

    Returns:
        Positional embedding tensor with signed feature pairs.

    Raises:
        ValueError: If `feature_size` is incompatible with feature components.
    """
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def relative_shift(x):
    """Apply relative-shift transform for relative attention logits."""
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]

# classes

class Residual(nn.Module):
    """Residual wrapper that adds function output to the input tensor."""

    def __init__(self, fn):
        """Initialize residual wrapper.

        Args:
            fn: Module or callable applied before residual addition.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Apply wrapped function and add identity shortcut."""
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    """Approximate GELU activation used in the original Enformer code."""

    def forward(self, x):
        """Apply activation to input tensor."""
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    """Attention-based local pooling for 1D sequence features."""

    def __init__(self, dim, pool_size = 2):
        """Initialize attention pooling module.

        Args:
            dim: Channel dimension of input feature map.
            pool_size: Number of neighboring positions to pool.
        """
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = 2)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        """Pool adjacent positions with learned attention weights."""
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    """Center-crop sequence axis to a target output length."""

    def __init__(self, target_length):
        """Initialize cropping module.

        Args:
            target_length: Final output length. `-1` disables cropping.
        """
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        """Crop sequence dimension around center position.

        Raises:
            ValueError: If requested target length is larger than sequence length.
        """
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        #trim = (target_len - seq_len) // 2
        #return x[:, -trim:trim]
        middle = seq_len // 2
        left = middle - target_len // 2
        right = target_len + left
        return x[:, left:right]

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    """Create a batchnorm-GELU-conv block used by Enformer trunk."""
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

# for replacing the batchnorm resnet blocks with convnext blocks

class LayerNorm(nn.Module):
    """Channel-wise LayerNorm for `(batch, channels, length)` tensors."""

    def __init__(self, dim, eps = 1e-5):
        """Initialize normalization parameters.

        Args:
            dim: Channel dimension size.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        """Normalize channel axis while preserving sequence axis."""
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class ConvNextBlock(nn.Module):
    """Depthwise-convolutional residual block used as ConvNeXt-style variant."""

    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 7,
        ff_mult = 2
    ):
        """Initialize ConvNeXt-like block.

        Args:
            dim: Input channel size.
            dim_out: Output channel size. Defaults to `dim`.
            kernel_size: Depthwise convolution kernel size.
            ff_mult: Expansion factor for pointwise feed-forward stage.
        """
        super().__init__()
        dim_out = default(dim_out, dim)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding = kernel_size // 2, groups = dim),
            LayerNorm(dim),
            nn.Conv1d(dim, dim_out * ff_mult, 1),
            GELU(),
            nn.Conv1d(ff_mult * dim_out, dim_out, 1)
        )

    def forward(self, x):
        """Apply block and residual projection."""
        return self.net(x) + self.res_conv(x)

# attention classes

class Attention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        """Initialize attention module.

        Args:
            dim: Embedding dimension.
            num_rel_pos_features: Relative positional feature dimension.
            heads: Number of attention heads.
            dim_key: Per-head key dimension.
            dim_value: Per-head value dimension.
            dropout: Attention dropout probability.
            pos_dropout: Positional feature dropout probability.
        """
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Run self-attention on sequence embeddings.

        Args:
            x: Input tensor with shape `(batch, seq_len, dim)`.

        Returns:
            Tensor with shape `(batch, seq_len, dim)`.
        """
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class Enformer(nn.Module):
    """Enformer architecture with optional species-specific output heads."""

    def __init__(
        self,
        *,
        dim = 1536,
        depth = 11,
        heads = 8,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = TARGET_LENGTH,
        attn_dim_key = 64,
        dropout_rate = 0.4,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        use_checkpointing = False,
        use_convnext = False
    ):
        """Initialize Enformer backbone and prediction heads.

        Args:
            dim: Main model width.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            output_heads: Mapping from head name to output feature count.
            target_length: Length after center-cropping.
            attn_dim_key: Attention key dimension per head.
            dropout_rate: Dropout used in feed-forward paths.
            attn_dropout: Dropout applied to attention probabilities.
            pos_dropout: Dropout applied to positional features.
            use_checkpointing: Whether to checkpoint transformer blocks.
            use_convnext: Whether to use ConvNeXt-style conv blocks.
        """
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        twice_dim = dim * 2

        conv_block_klass = ConvNextBlock if use_convnext else ConvBlock

        # create stem

        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(conv_block_klass(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, dim, num = 6, divisible_by = 128)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                conv_block_klass(dim_in, dim_out, kernel_size = 5),
                Residual(conv_block_klass(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    Attention(
                        dim,
                        heads = heads,
                        dim_key = attn_dim_key,
                        dim_value = dim // heads,
                        dropout = attn_dropout,
                        pos_dropout = pos_dropout,
                        num_rel_pos_features = dim // heads
                    ),
                    nn.Dropout(dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(
            Rearrange('b d n -> b n d'),
            *transformer
        )

        # target cropping

        self.target_length = target_length
        self.crop_final = TargetLengthCrop(target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            conv_block_klass(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        # create trunk sequential module

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # create final heads for human and mouse

        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(twice_dim, features),
            nn.Softplus()
        ), output_heads))

        # use checkpointing on transformer trunk

        self.use_checkpointing = use_checkpointing

    def set_target_length(self, target_length):
        """Update target crop length without rebuilding the model."""
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        """Return the shared trunk module before output heads."""
        return self._trunk

    @property
    def heads(self):
        """Return output-head module dictionary."""
        return self._heads

    def trunk_checkpointed(self, x):
        """Run trunk with checkpointing over transformer layers.

        Args:
            x: Input tensor with shape `(batch, sequence_length, 4)`.

        Returns:
            Trunk embeddings with shape `(batch, target_length, dim * 2)`.
        """
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.transformer[0](x)

        # todo (move the rearrange out of self.transformers sequential module, and transfer all weights to new module rearrangement, directly checkpoint on self.transformers)
        transformer_blocks = self.transformer[1:]
        x = checkpoint_sequential(nn.Sequential(*transformer_blocks), len(transformer_blocks), x)

        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = 'human'#None
    ):
        """Run Enformer forward pass and optionally compute loss/metrics.

        Args:
            x: Input sequences as one-hot tensor, long-index tensor, or list.
            target: Optional target tensor for direct loss/correlation output.
            return_corr_coef: If True with target, return Pearson correlation.
            return_embeddings: If True, return `(predictions, embeddings)`.
            return_only_embeddings: If True, skip heads and return embeddings.
            head: Selected output head key. If `None`, returns all heads.

        Returns:
            One of the following depending on flags:
                - Embeddings tensor.
                - Prediction tensor or dict of tensors.
                - Scalar loss/correlation when `target` is provided.
                - Tuple `(predictions, embeddings)` when requested.
        """
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, '() ... -> ...')

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f'head {head} not found'
            out = out[head]

        if exists(target):
            assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

            if return_corr_coef:
                return pearson_corr_coef(out, target)

            return poisson_loss(out, target)

        if return_embeddings:
            return out, x

        return out
