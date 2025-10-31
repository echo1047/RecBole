# -*- coding: utf-8 -*-
"""
MambaSubgoal (RecBole Adapter)
################################################

RecBole adapter for MambaSubgoal hierarchical RL model.

This adapter wraps the HybridPolicy (StateTracker + MambaSubgoal + TransformerSlate)
to make it compatible with RecBole's SequentialRecommender interface.

Architecture:
    User History → StateTracker → State Sequence [B, L, 256]
                                ↓
                    (every c steps) MambaSubgoal → Subgoal [B, subgoal_dim]
                                ↓
                    TransformerSlate(state, subgoal, candidates) → Slate [B, K]

Reference:
    - MambaSubgoal implementation in src/mamba_subgoal/models/
    - RecBole SequentialRecommender API
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Optional

# Add src to path to import RL models
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

# Import RL model components
from mamba_subgoal.models.state_tracker import StateTracker
from mamba_subgoal.models.mamba_subgoal import MambaSubgoal as MambaSubgoalModule
from mamba_subgoal.models.transformer_slate import TransformerSlate
from mamba_subgoal.models.hybrid_policy import HybridPolicy


class MambaSubgoal(SequentialRecommender):
    """
    RecBole adapter for MambaSubgoal hierarchical RL model.

    This class wraps the three-layer architecture (StateTracker, MambaSubgoal, TransformerSlate)
    and exposes RecBole's standard interface for training and evaluation.

    Key differences from standard RecBole models:
    1. Hierarchical policy with subgoal caching
    2. Slate generation (list-level recommendation)
    3. Compatible with offline RL algorithms (BC, CQL, IQL)
    """

    def __init__(self, config, dataset):
        super(MambaSubgoal, self).__init__(config, dataset)

        # ============================================
        # Load configuration parameters
        # ============================================

        # State Tracker parameters
        self.state_embedding_dim = config.get('state_embedding_dim', 128)
        self.state_hidden_dim = config.get('state_hidden_dim', 256)
        self.state_num_layers = config.get('state_num_layers', 2)
        self.state_dropout = config.get('state_dropout', 0.1)

        # High-level policy (Mamba) parameters
        self.high_level_hidden_dim = config.get('high_level_hidden_dim', 256)
        self.high_level_num_layers = config.get('high_level_num_layers', 2)
        self.subgoal_dim = config.get('subgoal_dim', 128)

        # Low-level policy (Transformer Slate) parameters
        self.low_level_hidden_dim = config.get('low_level_hidden_dim', 256)
        self.low_level_num_heads = config.get('low_level_num_heads', 8)
        self.low_level_num_layers = config.get('low_level_num_layers', 2)
        self.low_level_dropout = config.get('low_level_dropout', 0.1)

        # Slate generation parameters
        self.slate_size = config.get('slate_size', 10)
        self.candidate_size = config.get('candidate_size', 100)

        # Loss configuration
        self.loss_type = config.get('loss_type', 'CE')

        # ============================================
        # Initialize three-layer architecture
        # ============================================

        # Layer 1: State Tracker (encode user history)
        self.state_tracker = StateTracker(
            n_items=self.n_items,
            embedding_dim=self.state_embedding_dim,
            hidden_dim=self.state_hidden_dim,
            num_layers=self.state_num_layers,
            dropout=self.state_dropout,
            max_seq_length=self.max_seq_length,
            encoder_type='transformer',  # Can also be 'gru' or 'lstm'
        )

        # Layer 2: High-level policy (Mamba subgoal generator)
        self.mamba_subgoal = MambaSubgoalModule(
            state_dim=self.state_hidden_dim,
            hidden_dim=self.high_level_hidden_dim,
            subgoal_dim=self.subgoal_dim,
            num_layers=self.high_level_num_layers,
        )

        # Layer 3: Low-level policy (Transformer slate decoder)
        self.transformer_slate = TransformerSlate(
            n_items=self.n_items,
            item_embedding_dim=self.state_embedding_dim,
            hidden_dim=self.low_level_hidden_dim,
            state_dim=self.state_hidden_dim,
            subgoal_dim=self.subgoal_dim,
            num_heads=self.low_level_num_heads,
            num_layers=self.low_level_num_layers,
            dropout=self.low_level_dropout,
            slate_size=self.slate_size,
        )

        # Hybrid Policy (coordinator)
        self.policy = HybridPolicy(
            state_tracker=self.state_tracker,
            mamba_subgoal=self.mamba_subgoal,
            transformer_slate=self.transformer_slate,
            update_freq=config.get('subgoal_update_freq', 10),
            adaptive_c=config.get('adaptive_c', False),
        )

        # Loss function
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' not supported")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass for sequence encoding.

        Args:
            item_seq: [B, L] User interaction history
            item_seq_len: [B] Actual sequence length (excluding padding)

        Returns:
            seq_output: [B, hidden_dim] Sequence representation
        """
        # Create mask from sequence length
        mask = (item_seq != 0).long()  # [B, L]

        # Encode state sequence using StateTracker
        # StateTracker returns [B, L, state_hidden_dim]
        state_sequence = self.state_tracker(item_seq, mask)

        # Use the last valid state as sequence representation
        # Gather the state at position item_seq_len-1 for each sequence
        batch_size = item_seq.size(0)
        gather_index = (item_seq_len - 1).long().unsqueeze(-1).unsqueeze(-1)
        gather_index = gather_index.expand(-1, -1, state_sequence.size(-1))
        seq_output = state_sequence.gather(1, gather_index).squeeze(1)  # [B, state_hidden_dim]

        return seq_output

    def calculate_loss(self, interaction):
        """
        Calculate training loss.

        Args:
            interaction: RecBole interaction dictionary

        Returns:
            loss: Training loss value
        """
        item_seq = interaction[self.ITEM_SEQ]  # [B, L]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        pos_items = interaction[self.POS_ITEM_ID]  # [B]

        # Get sequence representation
        seq_output = self.forward(item_seq, item_seq_len)  # [B, state_hidden_dim]

        if self.loss_type == 'BPR':
            # BPR loss: compare positive vs negative items
            neg_items = interaction[self.NEG_ITEM_ID]  # [B]

            # Get item embeddings
            pos_items_emb = self.transformer_slate.item_embedding(pos_items)  # [B, item_emb_dim]
            neg_items_emb = self.transformer_slate.item_embedding(neg_items)  # [B, item_emb_dim]

            # Project sequence output to item embedding space
            seq_output_proj = self.transformer_slate.item_proj(seq_output)  # [B, hidden_dim]

            # Calculate scores
            pos_score = torch.sum(seq_output_proj * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_proj * neg_items_emb, dim=-1)  # [B]

            loss = self.loss_fct(pos_score, neg_score)

        else:  # CE loss
            # Cross-entropy loss: predict next item from all candidates
            test_item_emb = self.transformer_slate.item_embedding.weight  # [n_items, item_emb_dim]

            # Project sequence output
            seq_output_proj = self.transformer_slate.item_proj(seq_output)  # [B, hidden_dim]

            # Calculate logits for all items
            logits = torch.matmul(seq_output_proj, test_item_emb.transpose(0, 1))  # [B, n_items]

            loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        """
        Predict scores for specific items.

        Args:
            interaction: RecBole interaction dictionary with test items

        Returns:
            scores: [B] Prediction scores
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # Get sequence representation
        seq_output = self.forward(item_seq, item_seq_len)

        # Get test item embeddings
        test_item_emb = self.transformer_slate.item_embedding(test_item)

        # Project sequence output
        seq_output_proj = self.transformer_slate.item_proj(seq_output)

        # Calculate scores
        scores = torch.mul(seq_output_proj, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (full ranking).

        Args:
            interaction: RecBole interaction dictionary

        Returns:
            scores: [B, n_items] Prediction scores for all items
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # Get sequence representation
        seq_output = self.forward(item_seq, item_seq_len)

        # Get all item embeddings
        test_items_emb = self.transformer_slate.item_embedding.weight  # [n_items, item_emb_dim]

        # Project sequence output
        seq_output_proj = self.transformer_slate.item_proj(seq_output)

        # Calculate scores for all items
        scores = torch.matmul(seq_output_proj, test_items_emb.transpose(0, 1))  # [B, n_items]

        return scores

    def generate_slate(self, item_seq, item_seq_len, candidates=None):
        """
        Generate recommendation slate using the full hierarchical policy.

        This method uses the complete pipeline:
        1. StateTracker encodes history
        2. MambaSubgoal generates subgoal (with caching)
        3. TransformerSlate autoregressively generates slate

        Args:
            item_seq: [B, L] User interaction history
            item_seq_len: [B] Actual sequence length
            candidates: [B, C] Candidate items (optional, defaults to top items)

        Returns:
            slate: [B, K] Generated recommendation slate
        """
        batch_size = item_seq.size(0)
        mask = (item_seq != 0).long()

        # If no candidates provided, use top-K most popular items as candidates
        if candidates is None:
            # Simple heuristic: use items 1 to candidate_size
            candidates = torch.arange(
                1, self.candidate_size + 1,
                device=item_seq.device
            ).unsqueeze(0).expand(batch_size, -1)  # [B, C]

        # Use HybridPolicy to generate slate
        slate, _, _ = self.policy.forward(
            item_ids=item_seq,
            candidates=candidates,
            mask=mask,
            training=False,
            temperature=1.0,
            return_logits=False,
        )

        return slate  # [B, K]
