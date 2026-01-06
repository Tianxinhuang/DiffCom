"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""


import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .checkpoint import checkpoint
from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

def zero_linear(l):
    nn.init.constant_(l.weight, 0.0)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        #self.lora1 = nn.Linear(width, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        
        self.lwidth = 8
        self.fnum = 1

        self.lora1 = nn.Linear(width//self.fnum, self.lwidth, device=device, dtype=dtype)
        self.lora2 = nn.Linear(self.lwidth, 3*width//self.fnum, device=device, dtype=dtype)
        init_linear(self.lora1, init_scale * 0.1)
        init_linear(self.lora2, init_scale * 0.1)

        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x0 = x
        x = self.c_qkv(x)

        #bsize, csize = x0.shape[0], x0.shape[-1]
        #x0 = x0.reshape([bsize, -1, csize//self.fnum])
        #lora = self.lora2(self.lora1(x0))
        #lora = lora.reshape([bsize, -1, 3*csize])
        #x += lora

        x = checkpoint(self.attention, (x,), (), True)

        #x0 = x
        x = self.c_proj(x)
        #lora = self.lora2(self.lora1(x0))
        #x += lora
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()

        self.lwidth = 4
        self.lora1 = nn.Linear(width, self.lwidth, device=device, dtype=dtype)
        self.lora2 = nn.Linear(self.lwidth, width, device=device, dtype=dtype)
        init_linear(self.lora1, init_scale*0.1)
        init_linear(self.lora2, init_scale*0.1)

        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)
        #print(width)
        #assert False

    def forward(self, x):
        result = self.c_proj(self.gelu(self.c_fc(x)))
        #lora = self.lora2(self.lora1(x))
        #result = result + lora
        return result


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        #print(layers)
        #assert False
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        
        self.trans = nn.Sequential(
            nn.Conv1d(12, 16, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 12, 1, stride=1, padding=0),
            #nn.Tanh()
            ).cuda()

        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

            #self.trans.weight.zero_()
            #self.trans.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        #print(cond_as_token)
        for emb, as_token in cond_as_token:
            #print(emb.shape)
            if not as_token:
                h = h + emb[:, None]
        #assert False
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        #print(len(extra_tokens), len(h), extra_tokens[0].shape, h[0].shape)
        #for tok in extra_tokens:
        #    print(tok.shape)
        #for hi in h:
        #    print(hi.shape)
        #assert False
        #if len(extra_tokens):
        #    h = torch.cat(extra_tokens + [h], dim=1)
        #print(h.shape)
        if len(extra_tokens):
            tokens = None
            for tokeni in extra_tokens:
                if tokens is None:
                    tokens = tokeni
                else:
                    if tokeni.shape[0]<tokens.shape[0]:
                        tokeni = tokeni.repeat([tokens.shape[0], 1, 1])
                    tokens = torch.cat([tokens, tokeni], dim=1)
            tokens2=[]
            for hi in h:
                tokens2.append(hi.unsqueeze(0))
            tokens2 = torch.cat(tokens2, dim=0)
            h = torch.cat([tokens, tokens2], dim=1)
            #print(h.shape)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        #print(h[0,0,:])
        #assert False
        #h = h.permute(0, 2, 1)
        #h = h + self.trans(h)
        #h = h.permute(0, 2, 1)
        #print(h.shape)
        #assert False
        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + int(token_cond), **kwargs)
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device, cache_dir=cache_dir)
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    #def forward(
    #    self,
    #    x: torch.Tensor,
    #    t: torch.Tensor,
    #    images: Optional[Iterable[Optional[ImageType]]] = None,
    #    texts: Optional[Iterable[Optional[str]]] = None,
    #    embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    #):
    #    """
    #    :param x: an [N x C x T] tensor.
    #    :param t: an [N] tensor.
    #    :param images: a batch of images to condition on.
    #    :param texts: a batch of texts to condition on.
    #    :param embeddings: a batch of CLIP embeddings to condition on.
    #    :return: an [N x C' x T] tensor.
    #    """
    #    assert x.shape[-1] == self.n_ctx

    #    t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
    #    clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
    #    assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

    #    if self.training:
    #        mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
    #        clip_out = clip_out * mask[:, None].to(clip_out)

    #    # Rescale the features to have unit variance
    #    clip_out = math.sqrt(clip_out.shape[1]) * clip_out

    #    clip_embed = self.clip_embed(clip_out)

    #    cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]

    #    return self._forward_with_cond(x, cond)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        oembed: Optional[torch.Tensor] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        #print(x.shape, self.n_ctx)
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        if not oembed is None:
            #oembed = torch.exp(oembed) 
            clip_out = oembed/(1e-10+oembed.square().sum(-1,keepdim=True).sqrt())
        else:
            clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
            #print(clip_out.sum(-1))
            #assert False
        #print(clip_out.shape, x.shape)
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)
        #print(clip_embed.square().sum(-1))

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]

        return self._forward_with_cond(x, cond)


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx
        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )

        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        oembed: Optional[torch.Tensor] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert images is not None or embeddings is not None or oembed is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if not oembed is None:
            clip_out = oembed#/(1e-10+oembed.detach().square().sum(-1,keepdim=True).sqrt())
        else:
            if images is not None:
                clip_out = self.clip.embed_images_grid(images)
            else:
                clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        #print(clip_out.shape)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)
        #print(clip_embed.shape)
        #assert False

        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]
        return self._forward_with_cond(x, cond)


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            torch.tensor(channel_scales, dtype=dtype, device=device)
            if channel_scales is not None
            else None,
        )
        self.register_buffer(
            "channel_biases",
            torch.tensor(channel_biases, dtype=dtype, device=device)
            if channel_biases is not None
            else None,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor, oembed = None,):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in model_kwargs:
            zero_emb = torch.zeros(
                [batch_size, self.clip.grid_feature_dim, self.clip.grid_size**2],
                device=next(self.parameters()).device,
            )
            return dict(embeddings=zero_emb, low_res=model_kwargs["low_res"])
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
        oembed: Optional[torch.Tensor] = None,
        hybrid: Optional[torch.Tensor] = True,
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        #print(x.shape, self.n_ctx)
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        #print(images, embeddings)
        #assert False
        #print(oembed, images, embeddings)
        if not oembed is None:
            clip_out = oembed
        else: 
            if images is not None:
                clip_out = self.clip.embed_images_grid(images)
            elif embeddings is not None:
                clip_out = embeddings
            else:
                # Support unconditional generation.
                clip_out = torch.zeros(
                    [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
                    dtype=x.dtype,
                    device=x.device,
                )
        #clip_out = torch.zeros(
        #            [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
        #            dtype=x.dtype,
        #            device=x.device,
        #        )

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)
        #print(clip_embed.shape)
        #assert False

        #if hybrid:
        #    #cond = [(t_embed, self.time_token_cond), (clip_embed[t%clip_embed.shape[0]], True), (low_res_embed, True)]
        #    cond = [(t_embed, self.time_token_cond), (clip_embed[t//(512//clip_embed.shape[0])], True), (low_res_embed, True)]
        #    result = self._forward_with_cond(x, cond)
        #return result

        cond = [(t_embed, self.time_token_cond), (clip_embed[:low_res_embed.shape[0]], True), (low_res_embed, True)]
        result = self._forward_with_cond(x, cond)
        return result
        #if hybrid:
        #    for i in range(low_res_embed.shape[0], clip_embed.shape[0]):
        #        condi = [(t_embed, self.time_token_cond), (clip_embed[i:i+1], True), (low_res_embed, True)]
        #        result += self._forward_with_cond(x, condi)
        #        #print(i)
        #    #assert False
        #return result/clip_embed.shape[0]
        ##return self._forward_with_cond(x, cond)
