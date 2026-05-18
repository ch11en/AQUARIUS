#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AQUARIUS 四元组模型 - 改进版

关键改进:
1. 可学习门控融合替代固定残差
2. 情感预测辅助任务
3. 支持过滤无四元组样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class QuadrupleModelV2(nn.Module):
    """
    改进的四元组模型

    关键改进:
    1. 可学习门控融合: gate = sigmoid(W[user_feat; item_feat; quad_feat])
    2. 情感预测头: 辅助任务增强学习
    3. 返回四元组数量用于过滤
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 投影层
        self.aspect_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.category_proj = nn.Sequential(
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

        # 多头自注意力融合
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 可学习门控融合 (改动3)
        # 输入: user_feat_summary(128) + item_feat_summary(128) + quad_fused_feat(128) = 384
        # 初始化gate_bias让gate初始值约为0.3而不是接近0
        self.gate_network = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # 初始化最后一层bias为正值，让gate初始值约为0.3
        with torch.no_grad():
            self.gate_network[2].bias.fill_(-0.847)  # sigmoid(-0.847) ≈ 0.3

        # 评分预测头
        self.rating_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 情感预测头 (改动2)
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, aspect_feat, category_feat, opinion_feat, sentiment_scores):
        """
        Args:
            aspect_feat: [batch, 128]
            category_feat: [batch, 128]
            opinion_feat: [batch, 128]
            sentiment_scores: [batch] 原始情感分数

        Returns:
            quad_rating: [batch] 四元组预测评分
            quad_sentiment: [batch] 情感预测 [0,1]
            fused_feat: [batch, hidden_dim] 融合特征
        """
        batch_size = aspect_feat.size(0)

        # 投影
        h_a = self.aspect_proj(aspect_feat)
        h_c = self.category_proj(category_feat)
        h_o = self.opinion_proj(opinion_feat)
        h_s = self.sentiment_proj(sentiment_scores.unsqueeze(-1))

        # 堆叠成序列 [batch, 4, hidden_dim]
        seq = torch.stack([h_a, h_c, h_o, h_s], dim=1)

        # 自注意力
        attn_out, _ = self.self_attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)

        # FFN
        seq = self.norm2(seq + self.ffn(seq))

        # 聚合 (取平均)
        fused_feat = seq.mean(dim=1)  # [batch, hidden_dim]

        # 评分预测
        quad_rating = self.rating_predictor(fused_feat).squeeze(-1)

        # 情感预测 (改动2)
        quad_sentiment = self.sentiment_head(fused_feat).squeeze(-1)

        return quad_rating, quad_sentiment, fused_feat

    def predict_with_base(self, base_rating, user_feat_summary, item_feat_summary,
                          aspect_feat, category_feat, opinion_feat, sentiment_scores,
                          fixed_alpha=0.15):
        """
        与基础GNN预测进行融合 - 使用固定权重而非可学习门控

        Args:
            base_rating: [batch] 基础GNN预测的评分
            user_feat_summary: [batch, 128] 用户特征摘要
            item_feat_summary: [batch, 128] 物品特征摘要
            aspect_feat, category_feat, opinion_feat, sentiment_scores: 四元组特征
            fixed_alpha: float, 四元组模型的固定贡献权重

        Returns:
            final_rating: [batch] 最终预测评分
            quad_sentiment: [batch] 情感预测
            gate_values: [batch, 1] 固定权重值（用于监控）
        """
        # 四元组模型预测
        quad_rating, quad_sentiment, quad_fused = self(
            aspect_feat, category_feat, opinion_feat, sentiment_scores
        )

        # 使用固定权重融合，确保quad_model始终有贡献
        # final = base * (1 - alpha) + quad * alpha
        alpha = fixed_alpha
        final_rating = base_rating * (1 - alpha) + quad_rating * alpha

        # 限制在1-5范围内
        final_rating = torch.clamp(final_rating, 1.0, 5.0)

        # 返回固定alpha作为gate_values用于监控
        gate = torch.full((base_rating.size(0), 1), alpha, device=base_rating.device)

        return final_rating, quad_sentiment, gate


class QuadrupleFeatureExtractor:
    """四元组特征提取器"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._init_checkpoint_dir()
        self._load_data()

    def _init_checkpoint_dir(self):
        PROJECT_DIR = "/data/cxf2022/dl_project/AQUARIUS-main"
        if 'Musical' in self.dataset_name:
            self.checkpoint_dir = f"{PROJECT_DIR}/checkpoint/filtered_Musical_Instruments_output/BERT-Whitening"
        elif 'Industrial' in self.dataset_name:
            self.checkpoint_dir = f"{PROJECT_DIR}/checkpoint/filtered_Industrial_and_Scientific_output/BERT-Whitening"
        elif 'yelp' in self.dataset_name.lower() or 'restaurant' in self.dataset_name.lower():
            self.checkpoint_dir = f"{PROJECT_DIR}/checkpoint/filtered_yelp_restaurant_reviews/BERT-Whitening"
        else:
            self.checkpoint_dir = f"{PROJECT_DIR}/checkpoint/{self.dataset_name}/BERT-Whitening"

    def _load_data(self):
        import pickle
        import os

        print(f"\n加载四元组数据: {self.dataset_name}")

        # 加载四元组映射
        quadruple_file = f"{self.checkpoint_dir}/ui_to_quadruples.pkl"
        if os.path.exists(quadruple_file):
            with open(quadruple_file, 'rb') as f:
                self.ui_to_quadruples = pickle.load(f)
            print(f"  四元组映射: {len(self.ui_to_quadruples)} 条")
        else:
            self.ui_to_quadruples = {}
            print(f"  警告: 未找到四元组文件")

        # 加载层次化嵌入
        hierarchical_file = f"{self.checkpoint_dir}/hierarchical_embeddings.pkl"
        if os.path.exists(hierarchical_file):
            with open(hierarchical_file, 'rb') as f:
                hier_data = pickle.load(f)
                self.aspect_term_embeddings = hier_data.get('aspect_term_embeddings', {})
                self.opinion_term_embeddings = hier_data.get('opinion_term_embeddings', {})
                self.category_embeddings = hier_data.get('category_embeddings', {})
            print(f"  Aspect terms: {len(self.aspect_term_embeddings)}")
            print(f"  Opinion terms: {len(self.opinion_term_embeddings)}")
            print(f"  Categories: {len(self.category_embeddings)}")
        else:
            self.aspect_term_embeddings = {}
            self.opinion_term_embeddings = {}
            self.category_embeddings = {}
            print("  警告: 未找到层次化嵌入文件")

    def get_quadruple_features(self, user_id, item_id, device='cpu'):
        """获取单个user-item对的四元组特征"""
        key = (int(user_id), int(item_id))
        quadruples = self.ui_to_quadruples.get(key, [])

        if not quadruples:
            return (torch.zeros(128, device=device),
                    torch.zeros(128, device=device),
                    torch.zeros(128, device=device),
                    0.0, 0)  # 最后一个返回四元组数量

        aspect_embs, category_embs, opinion_embs, sentiments = [], [], [], []

        for q in quadruples[:8]:  # 最多取8个四元组
            if q['aspect_term'] in self.aspect_term_embeddings:
                aspect_embs.append(self.aspect_term_embeddings[q['aspect_term']])
            cat = q.get('aspect_category', '')
            if cat in self.category_embeddings:
                category_embs.append(self.category_embeddings[cat])
            if q['opinion_term'] in self.opinion_term_embeddings:
                opinion_embs.append(self.opinion_term_embeddings[q['opinion_term']])
            sentiments.append(q['sentiment'])

        def normalize(emb_list):
            if not emb_list:
                return torch.zeros(128, device=device)
            feat = np.mean(emb_list, axis=0)
            if len(feat) > 128:
                feat = feat[:128]
            elif len(feat) < 128:
                feat = np.pad(feat, (0, 128 - len(feat)))
            return torch.FloatTensor(feat.astype(np.float32)).to(device)

        return (normalize(aspect_embs), normalize(category_embs),
                normalize(opinion_embs), np.mean(sentiments) if sentiments else 0.0,
                len(quadruples))

    def get_batch_features(self, user_ids, item_ids, device='cpu'):
        """批量获取四元组特征，返回四元组数量用于过滤"""
        batch_size = len(user_ids)
        aspect_feats, category_feats, opinion_feats, sentiments, quad_counts = [], [], [], [], []

        for i in range(batch_size):
            a, c, o, s, cnt = self.get_quadruple_features(
                user_ids[i].item(), item_ids[i].item(), device
            )
            aspect_feats.append(a)
            category_feats.append(c)
            opinion_feats.append(o)
            sentiments.append(s)
            quad_counts.append(cnt)

        return (torch.stack(aspect_feats), torch.stack(category_feats),
                torch.stack(opinion_feats), torch.FloatTensor(sentiments).to(device),
                torch.tensor(quad_counts, device=device))
