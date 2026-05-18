#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AQUARIUS 四元组模型 - 多架构实现
支持三种融合架构：
1. HSFN: 层次化语义融合网络 (Hierarchical Semantic Fusion Network)
2. Transformer: Transformer编码器融合
3. GNN: 图神经网络融合

四元组结构: (aspect_term, aspect_category, opinion_term, sentiment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class BaseProjection(nn.Module):
    """基础投影层，将四元组各元素投影到统一空间"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
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

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        Args:
            aspect_feat: [batch, 128]
            category_feat: [batch, 128]
            opinion_feat: [batch, 128]
            sentiment_scores: [batch]
        Returns:
            四个投影后的特征 [batch, hidden_dim]
        """
        aspect_h = self.aspect_proj(aspect_feat)
        category_h = self.category_proj(category_feat)
        opinion_h = self.opinion_proj(opinion_feat)
        sentiment_h = self.sentiment_proj(sentiment_scores.unsqueeze(-1))
        return aspect_h, category_h, opinion_h, sentiment_h


# =============================================================================
# 方案1: 层次化语义融合网络 (HSFN)
# =============================================================================

class HierarchicalSemanticFusionNetwork(nn.Module):
    """
    层次化语义融合网络 (改进版)

    融合流程:
    1. Category-Aspect Bidirectional Attention: 双向交互
    2. Category Adaptive Gate: 自适应门控控制Category信息流
    3. Aspect → Opinion: Aspect引导Opinion
    4. Sentiment Gating: 情感门控调节
    5. Self-Attention Fusion: 最终融合
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "HSFN"

        # 基础投影
        self.projection = BaseProjection(128, hidden_dim)

        # Level 1: Category-Aspect Bidirectional Attention
        # Category引导Aspect
        self.category_to_aspect_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Aspect引导Category
        self.aspect_to_category_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm1b = nn.LayerNorm(hidden_dim)

        # Category Adaptive Gate: 控制Category信息流
        self.category_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Level 2: Aspect-Guided Opinion Enhancement
        self.aspect_guided_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Level 3: Sentiment Gating
        self.sentiment_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Level 4: Self-Attention Fusion
        self.final_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 预测层 - 更深的网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor([0.3]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        Returns:
            rating: [batch] 预测评分
            fused_feat: [batch, hidden_dim] 融合特征
            attention_weights: dict 注意力权重
        """
        batch_size = aspect_feat.size(0)

        # 投影
        aspect_h, category_h, opinion_h, sentiment_h = self.projection(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # Level 1: Category-Aspect Bidirectional Attention
        # Category引导Aspect
        cat_to_asp, cat_attn = self.category_to_aspect_attn(
            query=aspect_h.unsqueeze(1),
            key=category_h.unsqueeze(1),
            value=category_h.unsqueeze(1)
        )
        aspect_from_cat = self.norm1(aspect_h + cat_to_asp.squeeze(1))

        # Aspect引导Category
        asp_to_cat, _ = self.aspect_to_category_attn(
            query=category_h.unsqueeze(1),
            key=aspect_h.unsqueeze(1),
            value=aspect_h.unsqueeze(1)
        )
        category_enhanced = self.norm1b(category_h + asp_to_cat.squeeze(1))

        # Category Adaptive Gate: 自适应融合Category信息
        gate_input = torch.cat([aspect_from_cat, category_enhanced], dim=-1)
        cat_gate = self.category_gate(gate_input)
        aspect_enhanced = aspect_from_cat + cat_gate * category_enhanced

        # Level 2: Aspect引导Opinion
        asp_guided, asp_attn = self.aspect_guided_attn(
            query=opinion_h.unsqueeze(1),
            key=aspect_enhanced.unsqueeze(1),
            value=aspect_enhanced.unsqueeze(1)
        )
        opinion_enhanced = self.norm2(opinion_h + asp_guided.squeeze(1))

        # Level 3: Sentiment Gating
        gate = self.sentiment_gate(sentiment_h)
        gated_opinion = opinion_enhanced * gate

        # Level 4: Self-Attention Fusion
        # 四个元素作为序列
        seq = torch.stack([aspect_enhanced, category_enhanced, opinion_enhanced, gated_opinion], dim=1)
        fused_out, final_attn = self.final_attn(seq, seq, seq)
        fused = self.norm3(seq + fused_out).mean(dim=1)

        # 预测
        rating = self.predictor(fused).squeeze(-1)

        attention_weights = {
            'category_to_aspect': cat_attn,
            'aspect_guided': asp_attn,
            'final': final_attn
        }

        return rating, fused, attention_weights


# =============================================================================
# 方案2: Transformer编码器融合
# =============================================================================

class TransformerFusionNetwork(nn.Module):
    """
    Transformer编码器融合网络

    将四个元素作为序列输入Transformer，通过自注意力学习关系
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Transformer"

        # 基础投影
        self.projection = BaseProjection(128, hidden_dim)

        # 位置编码 (4个元素)
        self.pos_encoding = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        batch_size = aspect_feat.size(0)

        # 投影
        aspect_h, category_h, opinion_h, sentiment_h = self.projection(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 构建序列 [aspect, category, opinion, sentiment]
        seq = torch.stack([aspect_h, category_h, opinion_h, sentiment_h], dim=1)

        # 添加位置编码
        seq = seq + self.pos_encoding

        # Transformer编码
        encoded = self.transformer(seq)

        # 聚合 (取平均)
        fused = encoded.mean(dim=1)

        # 预测
        rating = self.predictor(fused).squeeze(-1)

        return rating, fused, None


# =============================================================================
# 方案3: 图神经网络融合
# =============================================================================

class GraphAttentionFusionNetwork(nn.Module):
    """
    图神经网络融合网络

    将四个元素建模为图节点，通过GAT学习关系
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "GNN"

        # 基础投影
        self.projection = BaseProjection(128, hidden_dim)

        # 图注意力层
        self.gat1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 边权重 (可学习的邻接矩阵)
        # 4个节点: aspect=0, category=1, opinion=2, sentiment=3
        self.adj_weight = nn.Parameter(torch.ones(4, 4) * 0.5)

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
        batch_size = aspect_feat.size(0)

        # 投影
        aspect_h, category_h, opinion_h, sentiment_h = self.projection(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 构建节点特征 [batch, 4, hidden_dim]
        nodes = torch.stack([aspect_h, category_h, opinion_h, sentiment_h], dim=1)

        # 第一层GAT
        attn_out1, _ = self.gat1(nodes, nodes, nodes)
        nodes = self.norm1(nodes + attn_out1)

        # 第二层GAT
        attn_out2, _ = self.gat2(nodes, nodes, nodes)
        nodes = self.norm2(nodes + attn_out2)

        # 聚合 (取平均)
        fused = nodes.mean(dim=1)

        # 预测
        rating = self.predictor(fused).squeeze(-1)

        return rating, fused, None


# =============================================================================
# 消融实验模型
# =============================================================================

class TripleModel(nn.Module):
    """三元组模型: aspect + opinion + sentiment (去掉category)"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.name = "Triple"

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

        # 简化的注意力
        self.attention = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Sentiment门控
        self.sentiment_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor([0.25]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        # 忽略category
        aspect_h = self.aspect_proj(aspect_feat)
        opinion_h = self.opinion_proj(opinion_feat)
        sentiment_h = self.sentiment_proj(sentiment_scores.unsqueeze(-1))

        # 门控
        gate = self.sentiment_gate(sentiment_h)

        # 3个元素的序列
        seq = torch.stack([aspect_h, opinion_h, sentiment_h], dim=1)
        attn_out, _ = self.attention(seq, seq, seq)
        fused = self.norm(seq + attn_out).mean(dim=1)

        # 应用门控
        fused = fused * gate

        rating = self.predictor(fused).squeeze(-1)
        return rating, fused, None


class DoubleModel(nn.Module):
    """二元组模型: aspect + opinion (去掉category和sentiment)"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
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
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.residual_weight = nn.Parameter(torch.tensor([0.2]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        # 忽略category和sentiment
        aspect_h = self.aspect_proj(aspect_feat)
        opinion_h = self.opinion_proj(opinion_feat)

        # 简单拼接
        combined = torch.cat([aspect_h, opinion_h], dim=-1)
        fused = self.fusion(combined)

        rating = self.predictor(fused).squeeze(-1)
        return rating, fused, None


class NoTupleModel(nn.Module):
    """无元组模型: 仅返回0，使用基础GNN预测"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.name = "None"
        self.residual_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        batch_size = aspect_feat.size(0)
        dummy = torch.zeros(batch_size, 128, device=aspect_feat.device)
        return torch.zeros(batch_size, device=aspect_feat.device), dummy, None


# =============================================================================
# 统一接口
# =============================================================================

class QuadrupleModel(nn.Module):
    """
    四元组模型统一接口

    支持多种架构:
    - 'hsfn': 层次化语义融合网络
    - 'transformer': Transformer编码器融合
    - 'gnn': 图神经网络融合
    - 'triple': 三元组模型 (消融)
    - 'double': 二元组模型 (消融)
    - 'none': 无元组模型 (消融)
    """

    ARCHITECTURES = {
        'hsfn': HierarchicalSemanticFusionNetwork,
        'transformer': TransformerFusionNetwork,
        'gnn': GraphAttentionFusionNetwork,
        'triple': TripleModel,
        'double': DoubleModel,
        'none': NoTupleModel,
    }

    def __init__(self, architecture: str = 'hsfn', hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: {list(self.ARCHITECTURES.keys())}")

        self.architecture = architecture

        # 根据架构类型传递不同的参数
        if architecture == 'hsfn':
            self.model = self.ARCHITECTURES[architecture](
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif architecture == 'transformer':
            self.model = self.ARCHITECTURES[architecture](
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        elif architecture == 'gnn':
            self.model = self.ARCHITECTURES[architecture](
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:  # triple, double, none
            self.model = self.ARCHITECTURES[architecture](
                hidden_dim=hidden_dim,
                dropout=dropout
            )

        # 注册rating_values用于残差融合
        self.register_buffer('rating_values', torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        return self.model(aspect_feat, category_feat, opinion_feat, sentiment_scores)

    def predict_with_base(self, base_rating, aspect_feat, category_feat,
                          opinion_feat, sentiment_scores):
        """
        与基础GNN预测进行残差融合

        Args:
            base_rating: [batch] 基础GNN预测的评分
            aspect_feat, category_feat, opinion_feat, sentiment_scores: 四元组特征

        Returns:
            final_rating: [batch] 最终预测评分
        """
        quad_rating, fused_feat, attn_weights = self.model(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 残差融合
        alpha = torch.sigmoid(self.model.residual_weight)
        final_rating = base_rating + alpha * (quad_rating - base_rating)

        # 限制在1-5范围内
        final_rating = torch.clamp(final_rating, 1.0, 5.0)

        return final_rating, fused_feat, attn_weights


# =============================================================================
# 对比学习损失
# =============================================================================

class QuadrupleContrastiveLoss(nn.Module):
    """
    四元组对比学习损失

    确保aspect-opinion-sentiment的一致性
    """

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # 投影头
        self.aspect_head = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_head = nn.Linear(hidden_dim, hidden_dim)
        self.sentiment_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, aspect_h, opinion_h, sentiment_h, sentiment_scores):
        """
        计算对比学习损失

        正样本: 同一四元组内的aspect-opinion对
        负样本: 不同四元组的aspect-opinion对
        """
        batch_size = aspect_h.size(0)

        # 投影
        aspect_z = F.normalize(self.aspect_head(aspect_h), dim=-1)
        opinion_z = F.normalize(self.opinion_head(opinion_h), dim=-1)

        # 相似度矩阵
        sim_matrix = torch.matmul(aspect_z, opinion_z.T) / self.temperature

        # 正样本在对角线上
        labels = torch.arange(batch_size, device=aspect_h.device)

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("四元组模型测试")
    print("=" * 60)

    batch_size = 8
    hidden_dim = 128

    # 模拟输入
    aspect_feat = torch.randn(batch_size, 128)
    category_feat = torch.randn(batch_size, 128)
    opinion_feat = torch.randn(batch_size, 128)
    sentiment_scores = torch.randn(batch_size)

    # 测试所有架构
    for arch in ['hsfn', 'transformer', 'gnn', 'triple', 'double', 'none']:
        print(f"\n测试架构: {arch}")
        model = QuadrupleModel(architecture=arch, hidden_dim=hidden_dim)

        rating, fused, attn = model(aspect_feat, category_feat, opinion_feat, sentiment_scores)

        print(f"  输出评分: {rating.shape}")
        print(f"  融合特征: {fused.shape}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
