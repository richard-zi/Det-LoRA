"""
RF-DETR Detector Wrapper
=========================

Loads and manages the RF-DETR SOTA object detector for use with Det-LoRA.
Provides freezing, classification head expansion, and loss computation.

RF-DETR (ICLR 2026): DINOv2 backbone + Transformer decoder, SOTA real-time detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from rfdetr.util.misc import NestedTensor

# LoRA target modules in the RF-DETR decoder
LORA_TARGET_MODULES = [
    "cross_attn.value_proj",
    "cross_attn.output_proj",
    "self_attn.out_proj",
]

# Named adapter footprints. "default" reproduces the thesis main suite. The
# localization presets additionally adapt the modules that decide WHERE the
# deformable cross-attention samples (sampling_offsets, attention_weights) and
# how boxes are iteratively refined (decoder bbox_embed MLP; shared reference
# with transformer.decoder.bbox_embed). "linear1"/"linear2" (decoder FFN) exist
# only in decoder layers, so suffix matching cannot leak into the backbone.
LORA_TARGET_PRESETS: Dict[str, List[str]] = {
    "default": LORA_TARGET_MODULES,
    "localization": LORA_TARGET_MODULES
    + [
        "cross_attn.sampling_offsets",
        "cross_attn.attention_weights",
    ],
    "localization_box": LORA_TARGET_MODULES
    + [
        "cross_attn.sampling_offsets",
        "cross_attn.attention_weights",
        "bbox_embed.layers.0",
        "bbox_embed.layers.1",
        "bbox_embed.layers.2",
    ],
    "localization_box_ffn": LORA_TARGET_MODULES
    + [
        "cross_attn.sampling_offsets",
        "cross_attn.attention_weights",
        "bbox_embed.layers.0",
        "bbox_embed.layers.1",
        "bbox_embed.layers.2",
        "linear1",
        "linear2",
    ],
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RFDETRDetector:
    """
    Wrapper around RF-DETR (ICLR 2026) for use with Det-LoRA.

    RF-DETR is the current SOTA real-time object detector:
    - DINOv2 ViT backbone (pre-trained, frozen)
    - Transformer decoder with deformable cross-attention
    - This wrapper exposes the open-source detection variants used in this repo
      (`nano`, `small`, `base`, `medium`, `large`)

    Args:
        variant: Model variant ('nano', 'small', 'base', 'medium', 'large')
        device: Target device (auto-detected if None)
    """

    VARIANTS = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "base": "RFDETRBase",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    }

    VARIANT_INFO = {
        "nano": {
            "params": "30.5M",
            "ap": 48.4,
            "resolution": 384,
            "dec_layers": 2,
            "hidden_dim": 256,
        },
        "small": {
            "params": "32.1M",
            "ap": 53.0,
            "resolution": 512,
            "dec_layers": 3,
            "hidden_dim": 256,
        },
        "base": {
            "params": "32.1M",
            "ap": 52.3,
            "resolution": 560,
            "dec_layers": 3,
            "hidden_dim": 256,
        },
        "medium": {
            "params": "33.7M",
            "ap": 54.7,
            "resolution": 576,
            "dec_layers": 4,
            "hidden_dim": 256,
        },
        "large": {
            "params": "33.9M",
            "ap": 56.5,
            "resolution": 704,
            "dec_layers": 4,
            "hidden_dim": 256,
        },
    }

    def __init__(
        self,
        variant: str = "medium",
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.variant = variant

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(self.VARIANTS.keys())}"
            )

        print(f"[RF-DETR] Loading {variant} variant...")

        # Import and instantiate the right variant
        import rfdetr

        model_cls = getattr(rfdetr, self.VARIANTS[variant])
        self._rfdetr = model_cls()

        # Get the actual nn.Module (LWDETR) and move to target device
        self.model: nn.Module = self._rfdetr.model.model
        self.model.to(self.device)
        self.resolution = self._rfdetr.model.args.resolution

        # RF-DETR class_embed has 91 outputs: 90 COCO classes (0-89) + 1 no-object (90).
        # New classes are appended AFTER the existing outputs.
        # class_id_offset for new classes = class_embed.out_features (91)
        self._original_head_size = self.model.class_embed.out_features  # 91
        self.base_num_classes = self._original_head_size  # offset for new class labels

        # Build the criterion for loss computation
        self.criterion = self._build_criterion()
        self.added_classes: List[str] = []

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[RF-DETR] Loaded: {total_params:,} params")
        print(f"[RF-DETR] Resolution: {self.resolution}x{self.resolution}")
        print(f"[RF-DETR] Base classes: {self.base_num_classes} (COCO)")

    def _build_criterion(self, num_classes: Optional[int] = None) -> nn.Module:
        """Build the SetCriterion for DETR loss computation."""
        from rfdetr.main import populate_args
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        args = populate_args()
        args.segmentation_head = False
        # num_classes for criterion = class_embed.out_features - 1
        # (DETR convention: last class in logits = no-object)
        inner = self._get_inner_model()
        args.num_classes = (num_classes or inner.class_embed.out_features) - 1
        criterion, _ = build_criterion_and_postprocessors(args)
        return criterion

    def to(self, device: torch.device) -> "RFDETRDetector":
        """Move model and criterion to device."""
        self.device = device
        self.model.to(device)
        self.criterion.to(device)
        return self

    def freeze_all(self) -> int:
        """Freeze all model parameters."""
        count = 0
        for param in self.model.parameters():
            param.requires_grad = False
            count += 1
        print(f"[RF-DETR] Frozen all {count} parameter tensors")
        return count

    def get_num_classes(self) -> int:
        """Current total number of object classes (base + added)."""
        return self.base_num_classes + len(self.added_classes)

    def get_class_id(self, class_name: str) -> int:
        """Get the absolute class ID for an incrementally added class."""
        if class_name not in self.added_classes:
            raise ValueError(f"Unknown added class '{class_name}'")
        return self.base_num_classes + self.added_classes.index(class_name)

    def _get_inner_model(self) -> nn.Module:
        """Get the inner LWDETR model, handling PEFT wrapping."""
        model = self.model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model
        return model

    def expand_classification_head(self, class_name: str) -> None:
        """
        Expand the classification head by one class.

        Expands: class_embed, all enc_out_class_embed layers.
        """
        if class_name in self.added_classes:
            raise ValueError(f"Class '{class_name}' already added")

        self.added_classes.append(class_name)
        inner = self._get_inner_model()
        d_model = inner.class_embed.in_features  # 256
        old_num = inner.class_embed.out_features  # 91, 92, ...
        new_num = old_num + 1

        # Expand main class_embed
        inner.class_embed = self._expand_linear(inner.class_embed, new_num)

        # Expand all enc_out_class_embed layers
        for i in range(len(inner.transformer.enc_out_class_embed)):
            inner.transformer.enc_out_class_embed[i] = self._expand_linear(
                inner.transformer.enc_out_class_embed[i], new_num
            )

        # Rebuild criterion with updated head size
        # NOTE: base_num_classes stays at its original value (91) - it's the offset for new class IDs
        # New classes get IDs: 91 (tank), 92 (truck), 93 (aircraft), etc.
        self.criterion = self._rebuild_criterion()

        print(f"[RF-DETR] Expanded head: +'{class_name}' → {new_num} output classes")

    def _expand_linear(self, old_layer: nn.Linear, new_out: int) -> nn.Linear:
        """Expand a Linear layer by one output neuron, preserving old weights."""
        new_layer = nn.Linear(old_layer.in_features, new_out, bias=old_layer.bias is not None)
        with torch.no_grad():
            new_layer.weight[: old_layer.out_features] = old_layer.weight
            if old_layer.bias is not None:
                new_layer.bias[: old_layer.out_features] = old_layer.bias
            # New class: small random init with negative bias (low confidence)
            nn.init.xavier_uniform_(new_layer.weight[old_layer.out_features :])
            if old_layer.bias is not None:
                nn.init.constant_(new_layer.bias[old_layer.out_features :], -4.0)
        return new_layer.to(old_layer.weight.device)

    def _rebuild_criterion(self) -> nn.Module:
        """Rebuild criterion after head expansion."""
        return self._build_criterion().to(self.device)

    def prepare_input(self, pixel_values: torch.Tensor) -> NestedTensor:
        """Wrap a batch tensor into NestedTensor for RF-DETR."""
        mask = torch.zeros(
            pixel_values.shape[0],
            pixel_values.shape[2],
            pixel_values.shape[3],
            dtype=torch.bool,
            device=pixel_values.device,
        )
        return NestedTensor(pixel_values, mask)

    def extract_shared_encoder_context(self, pixel_values: torch.Tensor) -> Dict[str, Any]:
        """
        Compute the shared backbone/encoder state once for a batch.

        This is the expensive part of RF-DETR inference and is identical across
        class-specific LoRA adapters because Det-LoRA only adapts decoder
        attention projections. Joint inference can therefore reuse this state
        and run only the adapter-specific proposal+decoder path per class.
        """
        from rfdetr.models.transformer import gen_encoder_output_proposals

        samples = self.prepare_input(pixel_values)
        inner = self._get_inner_model()
        features, poss = inner.backbone(samples)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []

        for feat, pos_embed in zip(features, poss):
            src, mask = feat.decompose()
            bs, _, h, w = src.shape
            spatial_shapes.append((h, w))

            src_flatten.append(src.flatten(2).transpose(1, 2))
            lvl_pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            valid_ratios.append(inner.transformer.get_valid_ratio(mask))

        memory = torch.cat(src_flatten, dim=1)
        mask_flatten_tensor = torch.cat(mask_flatten, dim=1)
        lvl_pos_embed = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes_tensor = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=memory.device,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios_tensor = torch.stack(valid_ratios, dim=1)

        output_memory, output_proposals = gen_encoder_output_proposals(
            memory,
            mask_flatten_tensor,
            spatial_shapes_tensor,
            unsigmoid=not inner.bbox_reparam,
        )

        return {
            "batch_size": pixel_values.shape[0],
            "output_memory": output_memory,
            "output_proposals": output_proposals,
            "memory_key_padding_mask": mask_flatten_tensor,
            "pos": lvl_pos_embed,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes_tensor,
            "valid_ratios": valid_ratios_tensor,
        }

    def forward_from_shared_encoder_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode predictions from a precomputed shared encoder context.

        The current active LoRA adapter and current head state remain in effect,
        so this method can be called repeatedly after switching adapters
        without recomputing the backbone/encoder for each class.
        """
        inner = self._get_inner_model()
        if not inner.two_stage:
            raise RuntimeError("Shared encoder inference requires RF-DETR two-stage mode")

        transformer = inner.transformer
        output_memory = context["output_memory"]
        output_proposals = context["output_proposals"]
        memory_key_padding_mask = context["memory_key_padding_mask"]
        pos = context["pos"]
        level_start_index = context["level_start_index"]
        spatial_shapes = context["spatial_shapes"]
        valid_ratios = context["valid_ratios"]
        bs = int(context["batch_size"])

        output_memory_gidx = transformer.enc_output_norm[0](
            transformer.enc_output[0](output_memory)
        )
        cls_enc = transformer.enc_out_class_embed[0](output_memory_gidx)

        if inner.bbox_reparam:
            enc_outputs_coord_delta = transformer.enc_out_bbox_embed[0](output_memory_gidx)
            enc_outputs_coord_cxcy = (
                enc_outputs_coord_delta[..., :2] * output_proposals[..., 2:]
                + output_proposals[..., :2]
            )
            enc_outputs_coord_wh = (
                enc_outputs_coord_delta[..., 2:].exp() * output_proposals[..., 2:]
            )
            ref_enc = torch.cat([enc_outputs_coord_cxcy, enc_outputs_coord_wh], dim=-1)
            refpoint_embed_ts = ref_enc
        else:
            refpoint_embed_ts = (
                transformer.enc_out_bbox_embed[0](output_memory_gidx) + output_proposals
            )
            ref_enc = refpoint_embed_ts.sigmoid()

        topk = min(inner.num_queries, cls_enc.shape[-2])
        topk_proposals = torch.topk(cls_enc.max(-1)[0], topk, dim=1)[1]
        gather_boxes = topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        gather_feats = topk_proposals.unsqueeze(-1).repeat(1, 1, transformer.d_model)
        refpoint_embed_ts = torch.gather(refpoint_embed_ts, 1, gather_boxes).detach()
        memory_ts = torch.gather(output_memory_gidx, 1, gather_feats)

        tgt = inner.query_feat.weight[: inner.num_queries].unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = (
            inner.refpoint_embed.weight[: inner.num_queries].unsqueeze(0).repeat(bs, 1, 1)
        )

        ts_len = refpoint_embed_ts.shape[-2]
        refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
        refpoint_embed_subset = refpoint_embed[..., ts_len:, :]
        if inner.bbox_reparam:
            refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
            refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
            refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
            refpoint_embed_ts_subset = torch.cat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
        else:
            refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts
        refpoint_embed = torch.cat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

        hs, references = transformer.decoder(
            tgt,
            output_memory,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios.to(output_memory.dtype),
        )

        if inner.bbox_reparam:
            outputs_coord_delta = inner.bbox_embed(hs)
            outputs_coord_cxcy = (
                outputs_coord_delta[..., :2] * references[..., 2:] + references[..., :2]
            )
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * references[..., 2:]
            outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (inner.bbox_embed(hs) + references).sigmoid()

        outputs_class = inner.class_embed(hs)
        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "decoder_embeddings": hs[-1],
            "proposal_embeddings": memory_ts,
            "enc_outputs": {
                "pred_logits": cls_enc,
                "pred_boxes": ref_enc,
            },
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional loss computation.

        Args:
            pixel_values: Images [B, 3, H, W]
            targets: Optional list of dicts with 'labels' and 'boxes' (cxcywh normalized)

        Returns:
            Dict with 'pred_logits', 'pred_boxes', and optionally 'loss'
        """
        samples = self.prepare_input(pixel_values)
        outputs = self.model(samples)

        result = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        if "enc_outputs" in outputs:
            result["enc_outputs"] = outputs["enc_outputs"]
        if "aux_outputs" in outputs:
            result["aux_outputs"] = outputs["aux_outputs"]

        if targets is not None:
            losses = self.criterion(outputs, targets)
            # Respect the criterion weight_dict when available so the training
            # objective matches RF-DETR's configured loss weighting.
            weight_dict = getattr(self.criterion, "weight_dict", {})
            if weight_dict:
                total_loss = sum(
                    v * weight_dict.get(k, 1.0) for k, v in losses.items() if "loss" in k
                )
            else:
                total_loss = sum(v for k, v in losses.items() if "loss" in k)
            result["loss"] = total_loss
            result["loss_dict"] = losses

        return result

    def summary(self) -> str:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info = self.VARIANT_INFO.get(self.variant, {})
        lines = [
            "=" * 60,
            "RF-DETR Detector Summary",
            "=" * 60,
            f"Variant: {self.variant} (AP={info.get('ap', '?')} COCO)",
            f"Resolution: {self.resolution}x{self.resolution}",
            f"Total params: {total:,}",
            f"Trainable params: {trainable:,}",
            f"Base classes: 90 (COCO)",
            f"Added classes: {len(self.added_classes)} ({', '.join(self.added_classes) or 'none'})",
            "=" * 60,
        ]
        return "\n".join(lines)
