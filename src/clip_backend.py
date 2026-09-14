"""CLIP backends for TCRT.

Two backends are provided:

- ``TransformersClipBackend`` wraps ``transformers.CLIPModel``
  (``openai/clip-vit-base-patch16``). Both vision and text towers are frozen;
  only the prompt context vectors (CoOp-style) are trainable, exactly as in the
  paper's controlled protocol ("both encoders are frozen; only prompts and the
  method-specific lightweight modules are optimized").
- ``MockClipBackend`` is a lightweight deterministic backend used by the smoke
  tests / CPU sanity checks. It uses the same interface so that the full
  training loop can run without downloading weights.

The text prototypes are built from the single template "a photo of a {class}".
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientCheckpointNoop:
    pass


class PromptedTextEncoder(nn.Module):
    """Frozen CLIP text tower with learnable context vectors (CoOp-style).

    For a batch of class names we build
        [SOS] [ctx_1 ... ctx_M] [class subword tokens] [EOS]
    as input embeddings. Only ``ctx`` is learnable; all other embeddings and
    the transformer itself are frozen.
    """

    def __init__(self, clip_text_model, token_embedding, tokenizer,
                 n_ctx: int = 16, ctx_init: str = "a photo of a",
                 dropout: float = 0.0):
        super().__init__()
        self.text_model = clip_text_model
        self.token_embedding = token_embedding
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.d = clip_text_model.config.hidden_size
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        for p in self.token_embedding.parameters():
            p.requires_grad_(False)

        ctx = torch.empty(n_ctx, self.d)
        init_ids = tokenizer(ctx_init, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            init_embeds = token_embedding(init_ids.to(next(token_embedding.parameters()).device))
        init_embeds = init_embeds.mean(dim=1)  # (1, d)
        nn.init.normal_(ctx, std=0.02)
        with torch.no_grad():
            ctx += init_embeds
        self.ctx = nn.Parameter(ctx)

        # Fixed SOS/EOS embeddings.
        sos_id = tokenizer.convert_tokens_to_ids("<|startoftext|>")
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        with torch.no_grad():
            self.register_buffer("sos_embed", token_embedding(torch.tensor([sos_id])).squeeze(0))
            self.register_buffer("eos_embed", token_embedding(torch.tensor([eos_id])).squeeze(0))

    def forward(self, class_names: Sequence[str]) -> torch.Tensor:
        device = self.ctx.device
        encoded = self.tokenizer(
            list(class_names), return_tensors="pt", padding=False, truncation=True
        )
        class_ids = encoded["input_ids"].to(device)  # (K, L)
        K, L = class_ids.shape
        with torch.no_grad():
            class_embeds = self.token_embedding(class_ids)  # (K, L, d)
        ctx = self.ctx.unsqueeze(0).expand(K, self.n_ctx, self.d)  # (K, M, d)
        sos = self.sos_embed.unsqueeze(0).unsqueeze(1).expand(K, 1, self.d)
        eos = self.eos_embed.unsqueeze(0).unsqueeze(1).expand(K, 1, self.d)
        hidden = torch.cat([sos, ctx, class_embeds, eos], dim=1)  # (K, M+L+2, d)
        # Position embeddings (the encoder's forward expects inputs_embeds with
        # position encodings already added in this transformers version).
        pos = self.text_model.embeddings.position_embedding.weight[None, :hidden.shape[1]]
        hidden = hidden + pos
        S = hidden.shape[1]
        # 4D attention mask (K, 1, S, S): CLIP text attention is bidirectional.
        attention_mask = torch.ones(K, 1, S, S, device=device)
        hidden = self.text_model.encoder(inputs_embeds=hidden, attention_mask=attention_mask)[0]
        hidden = self.text_model.final_layer_norm(hidden)
        pooled = hidden[:, -1]  # EOS position, matches the CLIP pooler
        return pooled


class TransformersClipBackend(nn.Module):
    """Frozen CLIP ViT-B/16 + trainable prompt context (the paper's backbone)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 n_ctx: int = 16, ctx_init: str = "a photo of a",
                 text_logit_scale: Optional[float] = None,
                 image_encoder_freeze: bool = True):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.d = self.clip.config.projection_dim
        if image_encoder_freeze:
            for p in self.clip.parameters():
                p.requires_grad_(False)

        self.text_encoder = PromptedTextEncoder(
            clip_text_model=self.clip.text_model,
            token_embedding=self.clip.text_model.embeddings.token_embedding,
            tokenizer=self.tokenizer,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
        )
        # The learnable prompt context is the only trainable part of the towers.
        for p in self.text_encoder.text_model.parameters():
            p.requires_grad_(False)

        default_scale = float(self.clip.logit_scale.detach().exp().item())
        self.text_logit_scale = float(text_logit_scale) if text_logit_scale is not None else default_scale

    @property
    def backend_kind(self) -> str:
        return "transformers"

    def image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen vision tower; returns L2-normalized features (N, d)."""
        with torch.no_grad():
            out = self.clip.vision_model(images)
            pooled = out.last_hidden_state[:, 0]  # CLS token
            feats = self.clip.visual_projection(pooled)
        return F.normalize(feats, dim=-1)

    def text_prototypes(self, class_names: Sequence[str]) -> torch.Tensor:
        """Frozen-text prototypes T with the CURRENT learnable prompt.

        Returns L2-normalized (K, d). Gradients flow into the context vectors.
        """
        pooled = self.text_encoder(class_names)
        feats = self.clip.text_projection(pooled)
        return F.normalize(feats, dim=-1)

    def projected_text_prototypes(self, class_names: Sequence[str],
                                  V: torch.Tensor) -> torch.Tensor:
        """C = T V (not a free parameter; see Eq. semantic coordinates)."""
        T = self.text_prototypes(class_names)
        return T @ V


class MockClipBackend(nn.Module):
    """Deterministic stand-in for CLIP used by smoke tests / CPU checks.

    The interface is identical to :class:`TransformersClipBackend`; the vision
    projection is a fixed random orthogonal map and the text prototypes are
    fixed random unit vectors plus a small learnable prompt perturbation.
    """

    def __init__(self, d: int = 512, num_classes: int = 10, seed: int = 0,
                 n_ctx: int = 16, text_logit_scale: float = 100.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.text_logit_scale = text_logit_scale
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.register_buffer("vision_proj", q)
        T = F.normalize(torch.randn(num_classes, d, generator=g), dim=-1)
        self.register_buffer("_T_frozen", T)
        self.ctx = nn.Parameter(torch.zeros(num_classes, n_ctx, d))
        nn.init.normal_(self.ctx, std=0.02)
        self.n_ctx = n_ctx

    @property
    def backend_kind(self) -> str:
        return "mock"

    def image_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = images.reshape(images.shape[0], -1)
            if feats.shape[1] != self.d:
                feats = feats[:, : self.d]
            feats = feats @ self.vision_proj
        return F.normalize(feats, dim=-1)

    def text_prototypes(self, class_names: Sequence[str]) -> torch.Tensor:
        K = len(class_names)
        T = F.normalize(self._T_frozen[:K] + 0.02 * self.ctx.mean(dim=1), dim=-1)
        return T

    def projected_text_prototypes(self, class_names: Sequence[str],
                                  V: torch.Tensor) -> torch.Tensor:
        return self.text_prototypes(class_names) @ V


def build_clip_backend(cfg: dict, device: torch.device) -> nn.Module:
    """Factory used by main.py and the smoke tests."""
    kind = cfg.get("backend", "transformers")
    if kind == "transformers":
        return TransformersClipBackend(
            model_name=cfg.get("clip_model", "openai/clip-vit-base-patch16"),
            n_ctx=cfg.get("n_ctx", 16),
            ctx_init=cfg.get("ctx_init", "a photo of a"),
            text_logit_scale=cfg.get("text_logit_scale"),
        ).to(device)
    if kind == "mock":
        return MockClipBackend(
            d=cfg.get("feat_dim", 512),
            num_classes=cfg.get("num_classes", 10),
            seed=cfg.get("seed", 0),
            n_ctx=cfg.get("n_ctx", 16),
            text_logit_scale=cfg.get("text_logit_scale", 100.0),
        ).to(device)
    raise ValueError(f"unknown backend: {kind}")
