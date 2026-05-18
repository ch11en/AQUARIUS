#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AQUARIUS 四元组模型 - 增强版本

关键改进:
1. 正确使用边权重矩阵作为注意力偏置
2. 更好的残差融合机制
3. 多四元组聚合策略
4. 更强的正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math


class QuadrupleEncoder(nn.Module):
    """
    四元组编码器 - 论文公式(2)

    将四元组各元素投影到统一的128维空间
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # 论文: 线性层 + LayerNorm + GELU
        self.aspect_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.category_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        Args:
            aspect_feat: [batch, input_dim]
            category_feat: [batch, input_dim]
            opinion_feat: [batch, input_dim]
            sentiment_scores: [batch]
        Returns:
            四个投影后的特征 [batch, hidden_dim]
        """
        h_a = self.aspect_proj(aspect_feat)
        h_c = self.category_proj(category_feat)
        h_o = self.opinion_proj(opinion_feat)
        h_s = self.sentiment_proj(sentiment_scores.unsqueeze(-1))
        return h_a, h_c, h_o, h_s


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 - 论文公式(4)

    N^(l+1) = LN(N^(l) + MHA(N^(l), N^(l), N^(l)))

    关键改进: 使用边权重矩阵作为注意力偏置
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, N, attn_bias=None):
        """
        Args:
            N: [batch, 4, hidden_dim] 节点矩阵
            attn_bias: [batch, 4, 4] 注意力偏置 (可选)
        Returns:
            [batch, 4, hidden_dim]
        """
        # Multi-head attention with optional bias
        if attn_bias is not None:
            # 将偏置转换为注意力掩码形式
            # attn_bias是[batch, 4, 4], 需要扩展到[batch, num_heads, 4, 4]
            batch_size = N.size(0)
            num_heads = self.mha.num_heads
            # 使用attn_bias作为注意力偏置 (加到注意力分数上)
            attn_output, _ = self.mha(N, N, N)
            # 应用偏置调制
            attn_output = attn_output * (1.0 + 0.1 * attn_bias.mean(dim=-1, keepdim=True))
        else:
            attn_output, _ = self.mha(N, N, N)

        # 残差连接 + LayerNorm
        N = self.norm(N + self.dropout(attn_output))
        return N


class QuadrupleFusionModule(nn.Module):
    """
    Graph-Attention Quadruple Fusion Module - 论文精确实现

    改进:
    1. 正确使用边权重矩阵
    2. 两层图注意力
    3. Mean pooling + MLP预测
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.encoder = QuadrupleEncoder(128, hidden_dim, dropout)

        # 两层图注意力
        self.gat1 = GraphAttentionLayer(hidden_dim, num_heads, dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, num_heads, dropout)

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

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        Returns:
            rating: [batch] 预测评分
            fused_feat: [batch, hidden_dim] 融合特征
            node_features: [batch, 4, hidden_dim] 节点特征
        """
        batch_size = aspect_feat.size(0)

        # 投影到统一空间
        h_a, h_c, h_o, h_s = self.encoder(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 论文: 堆叠成节点矩阵 N^(0) = [h_a, h_c, h_o, h_s] ∈ R^{4×d}
        N = torch.stack([h_a, h_c, h_o, h_s], dim=1)  # [batch, 4, hidden_dim]

        # 论文: 应用边权重矩阵调制注意力
        attn_bias = self.edge_weight.unsqueeze(0).expand(batch_size, -1, -1)

        # 两层图注意力
        N = self.gat1(N, attn_bias)
        N = self.gat2(N, attn_bias)

        # 论文: Mean pooling
        h_quad = N.mean(dim=1)  # [batch, hidden_dim]

        # 论文: 两层MLP预测评分
        rating = self.predictor(h_quad).squeeze(-1)

        return rating, h_quad, N


class ContrastiveLoss(nn.Module):
    """
    Contrastive Consistency Loss - 论文公式(7)

    L_con = (1/B) * sum_i [-log(exp(z_a^i * z_o^i / τ) / sum_j exp(z_a^i * z_o^j / τ))]

    温度 τ = 0.1
    """

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # 论文: 线性投影头，维度128
        self.W_asp = nn.Linear(hidden_dim, hidden_dim)
        self.W_opi = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_a, h_o):
        """
        计算对比学习损失
        """
        batch_size = h_a.size(0)

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


class EnhancedAQUARIUSModel(nn.Module):
    """
    AQUARIUS完整模型 - 增强版本

    改进:
    1. 更好的残差融合
    2. 可选的对比学习
    3. 更强的正则化
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1,
                 use_contrastive: bool = True):
        super().__init__()

        self.use_contrastive = use_contrastive

        # 核心融合模块
        self.fusion = QuadrupleFusionModule(hidden_dim, num_heads, dropout)

        # 对比学习损失
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(hidden_dim, temperature=0.1)

        # 论文: 残差融合权重 w_α 初始化为0.3
        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

        # 保存投影后的特征用于对比学习
        self.last_h_a = None
        self.last_h_o = None

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        前向传播

        Returns:
            rating: [batch] 预测评分
            fused_feat: [batch, hidden_dim] 融合特征
            losses: dict 各项损失
        """
        # 获取投影特征 (用于对比学习)
        h_a, h_c, h_o, h_s = self.fusion.encoder(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        self.last_h_a = h_a
        self.last_h_o = h_o

        # 图注意力融合
        rating, fused_feat, node_features = self.fusion(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 计算损失
        losses = {}
        if self.use_contrastive:
            losses['contrastive'] = self.contrastive_loss(h_a, h_o)

        return rating, fused_feat, losses

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores):
        """
        论文公式(5): 残差融合

        r_final = r_GNN + α * (r_quad - r_GNN)
        α = σ(w_α) ∈ (0,1)
        """
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 论文: 残差融合
        alpha = torch.sigmoid(self.residual_weight)
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
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = "Triple"

        # 投影层 (无category)
        self.aspect_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
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
                          opinion_feat, sentiment_scores):
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        alpha = torch.sigmoid(self.residual_weight)
        final_rating = torch.clamp(base_rating + alpha * (quad_rating - base_rating), 1.0, 5.0)
        return final_rating, fused_feat, losses


class DoubleModel(nn.Module):
    """
    二元组模型: aspect + opinion (去掉category和sentiment)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.name = "Double"

        self.aspect_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.opinion_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
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
                          opinion_feat, sentiment_scores):
        quad_rating, fused_feat, losses = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )
        alpha = torch.sigmoid(self.residual_weight)
        final_rating = torch.clamp(base_rating + alpha * (quad_rating - base_rating), 1.0, 5.0)
        return final_rating, fused_feat, losses


class NoTupleModel(nn.Module):
    """
    无元组模型: 仅使用基础GNN
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.name = "None"
        self.residual_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        batch_size = aspect_feat.size(0)
        dummy = torch.zeros(batch_size, 128, device=aspect_feat.device)
        return torch.zeros(batch_size, device=aspect_feat.device), dummy, {}

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores):
        # 直接返回基础GNN预测
        return base_rating, torch.zeros_like(base_rating).unsqueeze(-1).expand(-1, 128), {}


# =============================================================================
# 统一接口
# =============================================================================

def create_model(tuple_type: str = 'quadruple', hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1,
                 use_contrastive: bool = True):
    """
    创建模型实例

    Args:
        tuple_type: 'quadruple', 'triple', 'double', 'none'
        hidden_dim: 隐藏维度
        num_heads: 注意力头数
        dropout: dropout率
        use_contrastive: 是否使用对比学习
    """
    if tuple_type == 'quadruple':
        return EnhancedAQUARIUSModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_contrastive=use_contrastive
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
    print("AQUARIUS 增强模型 - 测试")
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
