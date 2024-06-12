import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict
from einops import rearrange

from models.scene_models.pointtransformer import TransitionDown, TransitionUp, PointTransformerBlock

def get_positional_encoding(max_len: int, time_emb_dim: int) -> torch.Tensor:
    """ Get positional encoding

    Args:
        max_len: maximum length of sequence
        time_emb_dim: dimension of time embedding
    
    Return:
        pe: positional encoding
    """
    pe = torch.zeros(max_len, time_emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, time_emb_dim, 2).float() * (-np.log(10000.0) / time_emb_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d]
    return pe

class PositionalEncoding(nn.Module):
    def __init__(self, time_emb_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', get_positional_encoding(max_len, time_emb_dim)) # [max_len, 1, d]

    def forward(self, x):
        x = x + self.pe[:x.shape[0]] # [nframes, bs, d]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, time_embed_dim, max_len=5000):
        super().__init__()

        self.register_buffer('pe', get_positional_encoding(max_len, time_embed_dim)) # [max_len, 1, d]
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, timesteps):
        return self.time_embed(self.pe[timesteps]) # [bs, 1, d]

class SceneMapEncoderDecoder(nn.Module):

    def __init__(self, point_feat_dim: int, planes: List, blocks: List, num_points: int=8192) -> None:
        super().__init__()

        self.num_points = num_points
        self.c = point_feat_dim + 3
        block = PointTransformerBlock

        self.in_planes, planes = self.c, planes
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64

        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], is_head=True)  # transform p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
    
    @property
    def num_groups(self):
        return self.num_points // 64
    
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
        
    def forward(self, p: torch.Tensor, x: torch.Tensor, ) -> torch.Tensor:
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, 'b n c -> (b n) c')
        x0 = rearrange(x, 'b n c -> (b n) c')
        o0 = torch.IntTensor(offset).to(p0.device)

        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return [
            rearrange(x4, '(b n) c -> b n c', b=len(offset)),
            rearrange(x3, '(b n) c -> b n c', b=len(offset)),
            rearrange(x2, '(b n) c -> b n c', b=len(offset)),
            rearrange(x1, '(b n) c -> b n c', b=len(offset)),
        ]

class SceneMapEncoder(nn.Module):
    def __init__(self, point_feat_dim: int, planes: List, blocks: List, num_points: int=8192) -> None:
        super().__init__()

        self.num_points = num_points
        self.c = point_feat_dim + 3
        block = PointTransformerBlock

        self.in_planes, planes = self.c, planes
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
    
    @property
    def num_groups(self):
        return self.num_points // 64
    
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
        
    def forward(self, p: torch.Tensor, x: torch.Tensor, ) -> torch.Tensor:
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, 'b n c -> (b n) c')
        x0 = rearrange(x, 'b n c -> (b n) c')
        o0 = torch.IntTensor(offset).to(p0.device)

        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        return rearrange(x4, '(b n) c -> b n c', b=len(offset))

# Code from https://github.com/krasserm/perceiver-io/blob/main/perceiver/model/core/modules.py
KVCache = Tuple[torch.Tensor, torch.Tensor]

class RotaryPositionEmbedding:
    # Specified in https://arxiv.org/abs/2104.09864
    # Modified from https://github.com/lucidrains/rotary-embedding-torch
    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        # frq_pos_enc shape is (b, n, c).
        # frq_pos_enc is broadcast to (b, h, n, c).
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        seq_len = t.shape[-2]
        if self.right_align:
            # q and k are right-aligned in Perceiver AR
            pos_enc = self.frq_pos_enc[..., -seq_len:, :]
        else:
            # q and k are left-aligned
            pos_enc = self.frq_pos_enc[..., :seq_len, :]

        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim :]
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())

        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        # Rearranges channel dimension [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class ModuleOutput(OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """...

        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length and D the
                number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence length and C
                are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :param kv_cache: cache with past keys and values.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length and F the
                number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            kv_cache = (k, v)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)

        o_chunks = []

        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(self.max_heads_parallel, dim=1),
            k.split(self.max_heads_parallel, dim=1),
            v.split(self.max_heads_parallel, dim=1),
        ):
            attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            attn_max_neg = -torch.finfo(attn.dtype).max

            if pad_mask is not None:
                attn.masked_fill_(pad_mask, attn_max_neg)

            if self.causal_attention:
                attn.masked_fill_(causal_mask, attn_max_neg)

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
            o_chunks.append(o_chunk)

        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o = self.o_proj(o)

        return ModuleOutput(last_hidden_state=o, kv_cache=kv_cache)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer-norm cross-attention (see `MultiHeadAttention` for attention details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        x_kv_prefix: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """Pre-layer-norm cross-attention of query input `x_q` to key/value input (`x_kv` or `x_kv_prefix`).

        If `x_kv_prefix` is defined, the entire key/value input is a concatenation of `x_kv_prefix` and `x_q` along
        the sequence dimension. In this case, the query attends to itself at the end of the key/value sequence (use
        case: Perceiver AR). If `x_kv_prefix` is not defined, `x_kv` is the entire key/value input.
        """
        x_q = self.q_norm(x_q)

        if x_kv is None:
            x_kv_prefix = self.kv_norm(x_kv_prefix)
            x_kv = torch.cat([x_kv_prefix, x_q], dim=1)
        else:
            x_kv = self.kv_norm(x_kv)

        return self.attention(
            x_q, x_kv, pad_mask=pad_mask, rot_pos_emb_q=rot_pos_emb_q, rot_pos_emb_k=rot_pos_emb_k, kv_cache=kv_cache
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(
            x,
            x,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb,
            kv_cache=kv_cache,
        )


class AbstractAttentionLayer(nn.Sequential):
    def empty_kv_cache(self, x) -> KVCache:
        k_cache = torch.empty(x.shape[0], 0, self.num_qk_channels, dtype=x.dtype, device=x.device)
        v_cache = torch.empty(x.shape[0], 0, self.num_v_channels, dtype=x.dtype, device=x.device)
        return k_cache, v_cache

    def forward(self, *args, kv_cache: Optional[KVCache] = None, **kwargs):
        attn_output = self[0](*args, kv_cache=kv_cache, **kwargs)
        mlp_output = self[1](attn_output.last_hidden_state)
        return ModuleOutput(last_hidden_state=mlp_output.last_hidden_state, kv_cache=attn_output.kv_cache)


class CrossAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attention_residual: bool = True,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = cross_attn.attention.num_qk_channels
        self.num_v_channels = cross_attn.attention.num_v_channels

        super().__init__(
            Residual(cross_attn, residual_dropout) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class SelfAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = self_attn.attention.num_qk_channels
        self.num_v_channels = self_attn.attention.num_v_channels

        super().__init__(
            Residual(self_attn, residual_dropout),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )


class SelfAttentionBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_rotary_layers: int = 1,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                max_heads_parallel=max_heads_parallel,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]

        
        self.num_rotary_layers = num_rotary_layers
        super().__init__(*layers)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):
        if kv_cache is None:
            kv_cache_updated = None
        else:
            if len(kv_cache) == 0:
                # initialize kv_cache for each self-attention layer
                kv_cache = [layer.empty_kv_cache(x) for layer in self]
            kv_cache_updated = []

        for i, layer in enumerate(self):
            rot_pos_emb_use = i < self.num_rotary_layers or self.num_rotary_layers == -1
            rot_pos_emb_i = rot_pos_emb if rot_pos_emb_use else None

            kv_cache_i = None if kv_cache is None else kv_cache[i]
            output = layer(x, pad_mask=pad_mask, rot_pos_emb=rot_pos_emb_i, kv_cache=kv_cache_i)

            x = output.last_hidden_state

            if kv_cache_updated is not None:
                kv_cache_updated.append(output.kv_cache)

        return ModuleOutput(last_hidden_state=x, kv_cache=kv_cache_updated)


class MLP(nn.Sequential):
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )

    def forward(self, x):
        return ModuleOutput(last_hidden_state=super().forward(x))
