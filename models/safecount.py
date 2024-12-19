import torch
import torch.nn.functional as F
from torch import nn

from models.utils import (
    build_backbone,
    build_regressor,
    crop_roi_feat,
    get_activation,
    get_clones,
)
from utils.init_helper import initialize_from_cfg


class SAFECount(nn.Module):
    def __init__(
        self,
        block,
        backbone,
        pool,
        embed_dim,
        mid_dim,
        head,
        dropout,
        activation,
        exemplar_scales=[],
        initializer=None,
    ):
        super().__init__()
        self.exemplar_scales = [s for s in exemplar_scales if s != 1]
        self.backbone = build_backbone(**backbone)
        self.in_conv = nn.Conv2d(
            self.backbone.out_dim, embed_dim, kernel_size=1, stride=1
        )

        # Cache for storing backbone features at different scales
        self.feat_cache = {}

        self.safecount = SAFECountMultiBlock(
            block=block,
            pool=pool,
            out_stride=backbone.out_stride,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout,
            activation=activation,
        )
        self.count_regressor = build_regressor(in_dim=embed_dim, activation=activation)

        for module in [self.in_conv, self.safecount, self.count_regressor]:
            initialize_from_cfg(module, initializer)

    @torch.no_grad()
    def _compute_scaled_features(self, image, boxes):
        _, _, h, w = image.shape
        feat_scale_list = []
        boxes_scale_list = []

        for scale in self.exemplar_scales:
            # Compute scaled dimensions
            h_rsz = int(h * scale) // 16 * 16
            w_rsz = int(w * scale) // 16 * 16

            # Check cache
            cache_key = f"{h_rsz}_{w_rsz}"
            if cache_key not in self.feat_cache:
                image_scale = F.interpolate(
                    image, size=(h_rsz, w_rsz), mode="bilinear", align_corners=False
                )
                self.feat_cache[cache_key] = self.in_conv(self.backbone(image_scale))

            feat_scale = self.feat_cache[cache_key]

            # Scale boxes
            scale_h, scale_w = h_rsz / h, w_rsz / w
            boxes_scale = boxes.clone()
            boxes_scale[:, [0, 2]] *= scale_h
            boxes_scale[:, [1, 3]] *= scale_w

            feat_scale_list.append(feat_scale)
            boxes_scale_list.append(boxes_scale)

        return feat_scale_list, boxes_scale_list

    def forward(self, input):
        image = input["image"]
        assert image.shape[0] == 1, "Batch size must be 1!"
        boxes = input["boxes"].squeeze(0)

        # Compute backbone features
        feat = self.in_conv(self.backbone(image))

        # Get multi-scale features efficiently
        feat_scale_list, boxes_scale_list = self._compute_scaled_features(image, boxes)

        # Process through SAFE counting blocks
        output = self.safecount(
            feat_orig=feat,
            boxes_orig=boxes,
            feat_scale_list=feat_scale_list,
            boxes_scale_list=boxes_scale_list,
        )

        # Compute density prediction
        density_pred = self.count_regressor(output)
        input["density_pred"] = density_pred

        return input


class SAFECountMultiBlock(nn.Module):
    def __init__(
        self,
        block,
        pool,
        out_stride,
        embed_dim,
        mid_dim,
        head,
        dropout,
        activation,
    ):
        super().__init__()
        self.out_stride = out_stride
        safecount_block = SAFECountBlock(
            pool=pool,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout,
            activation=activation,
        )
        self.blocks = get_clones(safecount_block, block)

    def forward(self, feat_orig, boxes_orig, feat_scale_list, boxes_scale_list):
        output = feat_orig
        feat_list = [feat_orig] + feat_scale_list
        boxes_list = [boxes_orig] + boxes_scale_list
        # block 0 with multi-scale exemplars
        block = self.blocks[0]
        feat_boxes = []
        for feat, boxes in zip(feat_list, boxes_list):
            feat_boxes += crop_roi_feat(feat, boxes, self.out_stride)
        output = block(output, feat_boxes)
        # block 1-n
        for block in self.blocks[1:]:
            # list of box feat b x c x h x w, note that diff boxes have diff shapes
            feat_boxes = crop_roi_feat(output, boxes_orig, self.out_stride)
            output = block(output, feat_boxes)
        return output


class SAFECountBlock(nn.Module):
    def __init__(self, pool, embed_dim, mid_dim, head, dropout, activation):
        super().__init__()
        self.aggt = SimilarityWeightedAggregation(pool, embed_dim, head, dropout)

        # Use a single conv block with grouped convolutions
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                embed_dim, mid_dim, kernel_size=3, stride=1, padding=1, groups=head
            ),
            nn.GroupNorm(head, mid_dim),
            get_activation(activation)(),
            nn.Dropout(dropout),
            nn.Conv2d(
                mid_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=head
            ),
        )

        # Single normalization layer
        self.norm = nn.GroupNorm(head, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src):
        # First attention block
        tgt2 = self.aggt(query=tgt, keys=src, values=src)
        tgt = tgt + self.dropout(tgt2)

        # Efficient feature fusion
        tgt2 = self.conv_block(tgt)
        tgt = self.norm(tgt + self.dropout(tgt2))
        return tgt


class SimilarityWeightedAggregation(nn.Module):
    """
    Implement the multi-head attention with convolution to keep the spatial structure.
    """

    def __init__(self, pool, embed_dim, head, dropout):
        super().__init__()
        assert pool.size[0] % 2 == 1 and pool.size[1] % 2 == 1
        assert pool.type in ["max", "avg"]
        self.pool = pool
        self.embed_dim = embed_dim
        self.head = head
        self.head_dim = embed_dim // head
        assert self.head_dim * head == self.embed_dim

        self.in_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys, values):
        h_p, w_p = self.pool.size
        pad = (w_p // 2, w_p // 2, h_p // 2, h_p // 2)
        _, _, h_q, w_q = query.size()

        # Process query
        query = self.in_conv(query)
        query = query.permute(0, 2, 3, 1).contiguous()
        query = self.norm(query).permute(0, 3, 1, 2).contiguous()
        query = query.view(self.head, self.head_dim, h_q, w_q)

        # Process each key-value pair
        attns_list = []
        for key in keys:
            if self.pool.type == "max":
                key = F.adaptive_max_pool2d(key, self.pool.size)
            else:
                key = F.adaptive_avg_pool2d(key, self.pool.size)

            key = self.in_conv(key)
            key = key.permute(0, 2, 3, 1).contiguous()
            key = self.norm(key).permute(0, 3, 1, 2).contiguous()
            key = key.view(self.head, self.head_dim, h_p, w_p)

            # Compute attention for each head separately
            attn_list = []
            for q, k in zip(query, key):
                # Use conv2d for spatial attention
                attn = F.conv2d(F.pad(q.unsqueeze(0), pad), k.unsqueeze(0))  # [1,1,h,w]
                attn_list.append(attn)
            attn = torch.cat(attn_list, dim=0)  # [head,1,h,w]
            attns_list.append(attn)

        attns = torch.cat(attns_list, dim=1)  # [head,n,h,w]

        # Normalize attention scores
        attns = attns * float(self.embed_dim * h_p * w_p) ** -0.5
        attns = torch.exp(attns)
        attns_sn = attns / (attns.amax(dim=(2, 3), keepdim=True) + 1e-6)
        attns_en = attns / (attns.sum(dim=1, keepdim=True) + 1e-6)
        attns = self.dropout(attns_sn * attns_en)

        # Weighted aggregation
        feats = 0
        for idx, value in enumerate(values):
            if self.pool.type == "max":
                value = F.adaptive_max_pool2d(value, self.pool.size)
            else:
                value = F.adaptive_avg_pool2d(value, self.pool.size)

            value = self.in_conv(value)
            value = value.view(self.head, self.head_dim, h_p, w_p)

            attn = attns[:, idx, :, :].unsqueeze(1)  # [head,1,h,w]

            feat_list = []
            for w, v in zip(attn, value):
                # [1,c,h,w]
                feat = F.conv2d(F.pad(w.unsqueeze(0), pad), v.unsqueeze(1).flip(2, 3))
                feat_list.append(feat)
            feat = torch.cat(feat_list, dim=0)  # [head,c,h,w]
            feats += feat

        feats = feats.contiguous().view(1, self.embed_dim, h_q, w_q)
        return self.out_conv(feats)


def build_network(**kwargs):
    return SAFECount(**kwargs)
