#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AQUARIUS 四元组模型 - 论文精确实现版本

严格按照论文3.Methodology.tex实现:
1. Graph-Attention Quadruple Fusion Module
   - 4个元素投影到128维空间
   - 两层图注意力网络 (4 heads)
   - 可学习边权重矩阵 A ∈ R^{4×4}
   - Mean pooling + Two-layer MLP

2. Rating Prediction with Residual Fusion
   - r_final = r_GNN + α * (r_quad - r_GNN)
   - α = σ(w_α), w_α初始化为0.3

3. Contrastive Consistency Loss
   - 温度 τ = 0.1
   - Aspect-Opinion对比学习

增强模块 (来自Enhanced Ablation):
- Adaptive Gating
- Consistency Loss
- Contrastive Learning
- Residual Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math


class BaseProjection(nn.Module):
    """
    基础投影层 - 论文公式(2)

    将四元组各元素投影到统一的128维空间:
    h_a = GELU(LN(W_a * Emb(at)))
    h_c = GELU(LN(W_c * Emb(ac)))
    h_o = GELU(LN(W_o * Emb(ot)))
    h_s = GELU(LN(w_s * sp))
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        # 论文: 线性层 + LayerNorm + GELU
        self.aspect_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.category_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores,
                user_profile_feat=None, profile_stats=None):
        """
        Args:
            aspect_feat: [batch, input_dim] BERT-Whitening嵌入
            category_feat: [batch, input_dim]
            opinion_feat: [batch, input_dim]
            sentiment_scores: [batch] 标量情感极性
        Returns:
            四个投影后的特征 [batch, hidden_dim]
        """
        h_a = self.aspect_proj(aspect_feat)
        h_c = self.category_proj(category_feat)
        h_o = self.opinion_proj(opinion_feat)
        if sentiment_scores.dim() == 1:
            sentiment_scores = sentiment_scores.unsqueeze(-1)
        h_s = self.sentiment_proj(sentiment_scores)
        return h_a, h_c, h_o, h_s


class GraphAttentionQuadrupleFusion(nn.Module):
    """
    Graph-Attention Quadruple Fusion Module - 论文精确实现

    论文3.5节描述:
    1. 四个投影向量堆叠成节点矩阵 N^(0) = [h_a, h_c, h_o, h_s] ∈ R^{4×d}
    2. 两层图注意力网络，每层: N^(l+1) = LN(N^(l) + MHA(N^(l), N^(l), N^(l)))
    3. 可学习边权重矩阵 A ∈ R^{4×4}，初始化为0.5
    4. 第二层后mean pooling: h_quad = (1/4) * sum(N^(2)_j)
    5. 两层MLP预测评分
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.name = "PaperGAT"

        # 基础投影
        self.projection = BaseProjection(128, hidden_dim)

        # 论文: 两层图注意力网络，每层4个头
        self.mha1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.mha2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 论文: 可学习边权重矩阵 A ∈ R^{4×4}，初始化为0.5
        self.edge_weight = nn.Parameter(torch.ones(4, 4) * 0.5)

        # 论文: 两层MLP with LayerNorm, GELU, dropout
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 论文: 残差权重 w_α 初始化为0.3
        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

    def _attention_bias(self):
        # MultiheadAttention adds float masks to the attention logits.
        # A sigmoid-bounded learnable adjacency keeps the 4-node graph dense
        # while still allowing the model to emphasize specific tuple links.
        return torch.log(torch.sigmoid(self.edge_weight).clamp_min(1e-6))

    def forward_from_projected(self, h_a, h_c, h_o, h_s):
        """
        Returns:
            rating: [batch] 预测评分
            fused_feat: [batch, hidden_dim] 融合特征
            node_features: [batch, 4, hidden_dim] 节点特征
        """
        # 论文: 堆叠成节点矩阵 N^(0) = [h_a, h_c, h_o, h_s] ∈ R^{4×d}
        N = torch.stack([h_a, h_c, h_o, h_s], dim=1)  # [batch, 4, hidden_dim]

        # 论文: 使用可学习边权重矩阵 A ∈ R^{4×4} 调制注意力
        attn_bias = self._attention_bias()

        # 第一层图注意力
        # 论文: N^(l+1) = LN(N^(l) + MHA(N^(l), N^(l), N^(l)))
        attn_out1, _ = self.mha1(N, N, N, attn_mask=attn_bias)
        N = self.norm1(N + attn_out1)

        # 第二层图注意力
        attn_out2, _ = self.mha2(N, N, N, attn_mask=attn_bias)
        N = self.norm2(N + attn_out2)

        # 论文: Mean pooling
        h_quad = N.mean(dim=1)  # [batch, hidden_dim]

        # 论文: 两层MLP预测评分
        rating = self.predictor(h_quad).squeeze(-1)

        return rating, h_quad, N

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        h_a, h_c, h_o, h_s = self.projection(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        return self.forward_from_projected(h_a, h_c, h_o, h_s)


class AdaptiveGatingModule(nn.Module):
    """
    自适应门控模块 - Enhanced Ablation

    根据输入特征动态调整信息流
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h_a, h_c, h_o, h_s):
        """
        计算自适应门控权重
        """
        combined = torch.cat([h_a, h_c, h_o, h_s], dim=-1)
        gate = self.gate(combined)
        return gate


class UserProfileFusion(nn.Module):
    """Fuse LLM-assisted user preference profile features."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.profile_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.profile_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, user_profile_feat, profile_stats):
        profile_h = self.profile_proj(user_profile_feat)
        stats_h = self.stats_proj(profile_stats)
        gate = self.profile_gate(torch.cat([profile_h, stats_h], dim=-1))
        return profile_h * gate + stats_h * (1.0 - gate)


class ConsistencyLoss(nn.Module):
    """
    一致性损失 - Enhanced Ablation

    确保aspect-category-opinion之间的语义一致性
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.aspect_head = nn.Linear(hidden_dim, hidden_dim)
        self.category_head = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_a, h_c, h_o):
        """
        计算三元组一致性损失
        """
        z_a = F.normalize(self.aspect_head(h_a), dim=-1)
        z_c = F.normalize(self.category_head(h_c), dim=-1)
        z_o = F.normalize(self.opinion_head(h_o), dim=-1)

        # 三元组两两一致性
        sim_ac = (z_a * z_c).sum(dim=-1)
        sim_ao = (z_a * z_o).sum(dim=-1)
        sim_co = (z_c * z_o).sum(dim=-1)

        # 最大化一致性
        loss = 1 - (sim_ac + sim_ao + sim_co) / 3
        return loss.mean()


class ContrastiveConsistencyLoss(nn.Module):
    """
    Contrastive Consistency Loss - 论文精确实现

    论文公式(6):
    L_con = (1/B) * sum_i [-log(exp(z_a^i * z_o^i / τ) / sum_j exp(z_a^i * z_o^j / τ))]

    温度 τ = 0.1
    """

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.1,
                 max_batch_size: int = 2048):
        super().__init__()
        self.temperature = temperature
        self.max_batch_size = max_batch_size

        # 论文: 线性投影头，维度128
        self.W_asp = nn.Linear(hidden_dim, hidden_dim)
        self.W_opi = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_a, h_o):
        """
        计算对比学习损失

        正样本: 同一四元组内的aspect-opinion对
        负样本: 不同四元组的aspect-opinion对
        """
        batch_size = h_a.size(0)
        if batch_size > self.max_batch_size:
            indices = torch.randperm(batch_size, device=h_a.device)[:self.max_batch_size]
            h_a = h_a.index_select(0, indices)
            h_o = h_o.index_select(0, indices)
            batch_size = self.max_batch_size

        # 论文: L2归一化投影
        z_a = F.normalize(self.W_asp(h_a), dim=-1)
        z_o = F.normalize(self.W_opi(h_o), dim=-1)

        # 相似度矩阵
        sim_matrix = torch.matmul(z_a, z_o.T) / self.temperature

        # 论文: 正样本在对角线上
        labels = torch.arange(batch_size, device=h_a.device)

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class AQUARIUSModel(nn.Module):
    """
    AQUARIUS完整模型 - 论文精确实现

    包含所有增强模块:
    1. Graph-Attention Quadruple Fusion
    2. Adaptive Gating
    3. Consistency Loss
    4. Contrastive Learning
    5. Residual Fusion
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.3,
                 use_adaptive_gating: bool = True, use_consistency_loss: bool = True,
                 use_contrastive: bool = True, use_user_profile: bool = True,
                 use_residual_fusion: bool = True):
        super().__init__()

        self.use_adaptive_gating = use_adaptive_gating
        self.use_consistency_loss = use_consistency_loss
        self.use_contrastive = use_contrastive
        self.use_user_profile = use_user_profile
        self.use_residual_fusion = use_residual_fusion

        # 核心融合模块
        self.fusion = GraphAttentionQuadrupleFusion(hidden_dim, num_heads, dropout)

        if use_user_profile:
            self.user_profile_fusion = UserProfileFusion(hidden_dim, dropout)
            self.dynamic_residual_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.profile_delta = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        # 增强模块
        if use_adaptive_gating:
            self.adaptive_gate = AdaptiveGatingModule(hidden_dim)

        if use_consistency_loss:
            self.consistency_loss = ConsistencyLoss(hidden_dim)

        if use_contrastive:
            self.contrastive_loss = ContrastiveConsistencyLoss(hidden_dim, temperature=0.1)

        # 论文: 残差融合权重
        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores,
                user_profile_feat=None, profile_stats=None):
        """
        前向传播

        Returns:
            rating: [batch] 预测评分
            fused_feat: [batch, hidden_dim] 融合特征
            losses: dict 各项损失
        """
        # 获取投影特征
        h_a, h_c, h_o, h_s = self.fusion.projection(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 自适应门控
        if self.use_adaptive_gating:
            gate = self.adaptive_gate(h_a, h_c, h_o, h_s)
            h_a = h_a * gate
            h_c = h_c * gate
            h_o = h_o * gate
            h_s = h_s * gate

        # 图注意力融合。这里必须使用门控后的节点，否则 adaptive gating
        # 只会影响辅助损失而不会影响最终评分。
        rating, fused_feat, node_features = self.fusion.forward_from_projected(
            h_a, h_c, h_o, h_s
        )

        if self.use_user_profile and user_profile_feat is not None and profile_stats is not None:
            profile_context = self.user_profile_fusion(user_profile_feat, profile_stats)
            fused_feat = fused_feat + profile_context

        # 计算损失
        losses = {}
        if self.use_consistency_loss:
            losses['consistency'] = self.consistency_loss(h_a, h_c, h_o)

        if self.use_contrastive:
            losses['contrastive'] = self.contrastive_loss(h_a, h_o)

        return rating, fused_feat, losses

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores, user_profile_feat=None,
                          profile_stats=None):
        """
        论文公式(5): 残差融合

        r_final = r_GNN + α * (r_quad - r_GNN)
        α = σ(w_α) ∈ (0,1)

        Args:
            base_rating: [batch] 基础GNN预测
            四元组特征

        Returns:
            final_rating: [batch] 最终预测，限制在[1,5]
            losses: dict 各项损失
        """
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores,
            user_profile_feat, profile_stats
        )

        if not self.use_residual_fusion:
            final_rating = quad_rating
            final_rating = torch.clamp(final_rating, 1.0, 5.0)
            return final_rating, fused_feat, losses

        # 论文: 残差融合
        alpha = torch.sigmoid(self.residual_weight)
        if self.use_user_profile and user_profile_feat is not None and profile_stats is not None:
            profile_context = self.user_profile_fusion(user_profile_feat, profile_stats)
            scalar_inputs = torch.stack([
                base_rating,
                quad_rating,
                sentiment_scores if sentiment_scores.dim() == 1 else sentiment_scores.squeeze(-1)
            ], dim=-1)
            dynamic_gate = self.dynamic_residual_gate(
                torch.cat([fused_feat, profile_context, scalar_inputs], dim=-1)
            ).squeeze(-1)
            delta = self.profile_delta(torch.cat([fused_feat, profile_context], dim=-1)).squeeze(-1)
            final_rating = base_rating + alpha * dynamic_gate * (quad_rating - base_rating + 0.1 * delta)
            losses['profile_gate'] = dynamic_gate.mean()
        else:
            final_rating = base_rating + alpha * (quad_rating - base_rating)

        # 论文: 限制在[1, 5]范围内
        final_rating = torch.clamp(final_rating, 1.0, 5.0)

        return final_rating, fused_feat, losses


# =============================================================================
# 消融实验模型
# =============================================================================

class TripleModel(nn.Module):
    """
    三元组模型: aspect + opinion + sentiment (去掉category)

    论文消融实验配置
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.name = "Triple"

        # 投影层 (无category)
        self.aspect_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 3个节点的图注意力
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        # 忽略category
        h_a = self.aspect_proj(aspect_feat)
        h_o = self.opinion_proj(opinion_feat)
        h_s = self.sentiment_proj(sentiment_scores.unsqueeze(-1))

        # 3个节点的图
        N = torch.stack([h_a, h_o, h_s], dim=1)
        attn_out, _ = self.mha(N, N, N)
        N = self.norm(N + attn_out)

        fused = N.mean(dim=1)
        rating = self.predictor(fused).squeeze(-1)

        return rating, fused, {}

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores, user_profile_feat=None,
                          profile_stats=None):
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        alpha = torch.sigmoid(self.residual_weight)
        final_rating = torch.clamp(base_rating + alpha * (quad_rating - base_rating), 1.0, 5.0)
        return final_rating, fused_feat, losses


class DoubleModel(nn.Module):
    """
    二元组模型: aspect + opinion (去掉category和sentiment)

    论文消融实验配置
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.name = "Double"

        self.aspect_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 简单融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        # 忽略category和sentiment
        h_a = self.aspect_proj(aspect_feat)
        h_o = self.opinion_proj(opinion_feat)

        combined = torch.cat([h_a, h_o], dim=-1)
        fused = self.fusion(combined)
        rating = self.predictor(fused).squeeze(-1)

        return rating, fused, {}

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores, user_profile_feat=None,
                          profile_stats=None):
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        alpha = torch.sigmoid(self.residual_weight)
        final_rating = torch.clamp(base_rating + alpha * (quad_rating - base_rating), 1.0, 5.0)
        return final_rating, fused_feat, losses


class NoTupleModel(nn.Module):
    """
    无元组模型: 仅使用基础GNN

    论文消融实验配置
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.name = "None"
        self.residual_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        batch_size = aspect_feat.size(0)
        dummy = torch.zeros(batch_size, 128, device=aspect_feat.device)
        return torch.zeros(batch_size, device=aspect_feat.device), dummy, {}

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores, user_profile_feat=None,
                          profile_stats=None):
        # 直接返回基础GNN预测
        return base_rating, torch.zeros_like(base_rating).unsqueeze(-1).expand(-1, 128), {}


# =============================================================================
# 统一接口
# =============================================================================

def create_model(tuple_type: str = 'quadruple', hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.3,
                 use_adaptive_gating: bool = True,
                 use_consistency_loss: bool = True,
                 use_contrastive: bool = True,
                 use_user_profile: bool = True,
                 use_residual_fusion: bool = True):
    """
    创建模型实例

    Args:
        tuple_type: 'quadruple', 'triple', 'double', 'none'
        hidden_dim: 隐藏维度
        num_heads: 注意力头数
        dropout: dropout率
        use_adaptive_gating: 是否使用自适应门控
        use_consistency_loss: 是否使用一致性损失
        use_contrastive: 是否使用对比学习
    """
    if tuple_type == 'quadruple':
        return AQUARIUSModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_adaptive_gating=use_adaptive_gating,
            use_consistency_loss=use_consistency_loss,
            use_contrastive=use_contrastive,
            use_user_profile=use_user_profile,
            use_residual_fusion=use_residual_fusion
        )
    elif tuple_type == 'triple':
        return TripleModel(hidden_dim, num_heads, dropout)
    elif tuple_type == 'double':
        return DoubleModel(hidden_dim, dropout)
    elif tuple_type == 'none':
        return NoTupleModel(hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown tuple_type: {tuple_type}")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AQUARIUS 论文精确实现 - 模型测试")
    print("=" * 60)

    batch_size = 8
    hidden_dim = 128

    # 模拟输入
    aspect_feat = torch.randn(batch_size, 128)
    category_feat = torch.randn(batch_size, 128)
    opinion_feat = torch.randn(batch_size, 128)
    sentiment_scores = torch.randn(batch_size)
    base_rating = torch.randn(batch_size) * 2 + 3  # 模拟GNN预测

    # 测试所有配置
    for tuple_type in ['quadruple', 'triple', 'double', 'none']:
        print(f"\n测试: {tuple_type}")
        model = create_model(tuple_type, hidden_dim=hidden_dim)

        final_rating, fused_feat, losses = model.predict_with_base(
            base_rating, aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        print(f"  最终评分: {final_rating.shape}, 范围: [{final_rating.min():.2f}, {final_rating.max():.2f}]")
        print(f"  融合特征: {fused_feat.shape}")
        if losses:
            for k, v in losses.items():
                print(f"  {k} loss: {v.item():.4f}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
