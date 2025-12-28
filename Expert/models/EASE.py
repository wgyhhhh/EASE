import os
import torch
import tqdm
import time
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
import copy
from transformers import BertConfig, BertModel


class EASE(torch.nn.Module):
    def __init__(self, config, expert_name, emb_dim, co_attention_dim, mlp_dims, mlp_dropout):
        super(EASE, self).__init__()

        # Get configuration values
        self.emb_dim = emb_dim
        self.co_attention_dim = co_attention_dim
        self.mlp_dims = mlp_dims
        self.mlp_dropout = mlp_dropout
        self.bert_content = BertModel.from_pretrained(config.bert_path).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config.bert_path).requires_grad_(False)
        self.expert_type = expert_name
        self.eval_expert = expert_name

        # Only fine-tune the last layer of BERT
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.aggregator = MaskAttention(emb_dim)
        self.mlp = MLP(emb_dim, mlp_dims, mlp_dropout)

        # Feature 2 components (sentiment)
        self.hard_ftr_2_attention = MaskAttention(emb_dim)
        self.hard_mlp_ftr_2 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                            nn.ReLU(),
                                            nn.Linear(mlp_dims[-1], 1),
                                            nn.Sigmoid()
                                            )
        self.score_mapper_ftr_2 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                                nn.BatchNorm1d(mlp_dims[-1]),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(mlp_dims[-1], 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                                )

        # Feature 3 components (reasoning)
        self.hard_ftr_3_attention = MaskAttention(emb_dim)
        self.hard_mlp_ftr_3 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                            nn.ReLU(),
                                            nn.Linear(mlp_dims[-1], 1),
                                            nn.Sigmoid()
                                            )
        self.score_mapper_ftr_3 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                                nn.BatchNorm1d(mlp_dims[-1]),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(mlp_dims[-1], 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                                )

        # Feature 4 components (evidence)
        self.hard_ftr_4_attention = MaskAttention(emb_dim)
        self.hard_mlp_ftr_4 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                            nn.ReLU(),
                                            nn.Linear(mlp_dims[-1], 1),
                                            nn.Sigmoid()
                                            )
        self.score_mapper_ftr_4 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                                nn.BatchNorm1d(mlp_dims[-1]),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(mlp_dims[-1], 64),
                                                nn.BatchNorm1d(64),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(64, 1),
                                                nn.Sigmoid()
                                                )

        # Simple feature predictors
        self.simple_ftr_2_attention = MaskAttention(emb_dim)
        self.simple_mlp_ftr_2 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                              nn.ReLU(),
                                              nn.Linear(mlp_dims[-1], 3))
        self.simple_ftr_3_attention = MaskAttention(emb_dim)
        self.simple_mlp_ftr_3 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                              nn.ReLU(),
                                              nn.Linear(mlp_dims[-1], 3))
        self.simple_ftr_4_attention = MaskAttention(emb_dim)
        self.simple_mlp_ftr_4 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]),
                                              nn.Sigmoid(),
                                              nn.Linear(mlp_dims[-1], 3))

        # Attention mechanisms
        self.content_attention = MaskAttention(emb_dim)
        self.FTR4_attention = MaskAttention(emb_dim)
        self.image_attention = MaskAttention(emb_dim)

        # Co-attention networks
        self.co_attention_2 = ParallelCoAttentionNetwork(emb_dim, co_attention_dim, mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(emb_dim, co_attention_dim, mask_in=True)
        self.co_attention_4 = ParallelCoAttentionNetwork(emb_dim, co_attention_dim, mask_in=True)

        # Cross-attention modules for content-FTR interaction
        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_content_4 = SelfAttentionFeatureExtract(1, emb_dim)

        # Cross-attention modules for FTR-content interaction
        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_ftr_4 = SelfAttentionFeatureExtract(1, emb_dim)

        self.eval_mode = getattr(config, 'eval_mode', False)

    def forward(self, **kwargs):
        # Extract input data
        content, content_masks = kwargs['content'], kwargs['content_masks']
        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']
        FTR_3, FTR_3_masks = kwargs['FTR_3'], kwargs['FTR_3_masks']
        FTR_4, FTR_4_masks = kwargs['FTR_4'], kwargs['FTR_4_masks']

        # Extract features using BERT
        content_feature = self.bert_content(content, attention_mask=content_masks)[0]
        content_feature_1, content_feature_2 = content_feature, content_feature

        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask=FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask=FTR_3_masks)[0]
        FTR_4_feature = self.bert_FTR(FTR_4, attention_mask=FTR_4_masks)[0]

        # Cross attention between content and FTR features
        mutual_content_FTR_2, _ = self.cross_attention_content_2( \
            content_feature_2, FTR_2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)

        mutual_content_FTR_3, _ = self.cross_attention_content_3( \
            content_feature_2, FTR_3_feature, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_content_FTR_4, _ = self.cross_attention_content_4( \
            content_feature_2, FTR_4_feature, content_masks)
        expert_4 = torch.mean(mutual_content_FTR_4, dim=1)

        # Simple feature predictions
        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature)[0]).squeeze(1)
        simple_ftr_4_pred = self.simple_mlp_ftr_4(self.simple_ftr_4_attention(FTR_4_feature)[0]).squeeze(1)

        # Content attention
        attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)

        # Expert-specific processing
        if self.expert_type == 'sentiment':
            # print("Sentiment expert processing")
            all_feature = torch.cat(
                (attn_content.unsqueeze(1), expert_2.unsqueeze(1)),
                dim=1
            )
            final_feature, _ = self.aggregator(all_feature)
            label_pred = self.mlp(final_feature)

            res = {
                'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
                'final_feature': final_feature,
                'content_feature': attn_content,
                'simple_ftr_2_pred': simple_ftr_2_pred,
            }

        elif self.expert_type == 'reasoning':
            # print("Reasoning expert processing")
            all_feature = torch.cat(
                (attn_content.unsqueeze(1), expert_3.unsqueeze(1)),
                dim=1
            )
            final_feature, _ = self.aggregator(all_feature)
            label_pred = self.mlp(final_feature)

            res = {
                'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
                'final_feature': final_feature,
                'content_feature': attn_content,
                'simple_ftr_3_pred': simple_ftr_3_pred,
            }

        elif self.expert_type == 'evidence':
            # print("Evidence expert processing")
            all_feature = torch.cat(
                (attn_content.unsqueeze(1), expert_4.unsqueeze(1)),
                dim=1
            )
            final_feature, _ = self.aggregator(all_feature)
            label_pred = self.mlp(final_feature)


            res = {
                'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
                'final_feature': final_feature,
                'content_feature': attn_content,
                'simple_ftr_4_pred': simple_ftr_4_pred,
            }

        return res