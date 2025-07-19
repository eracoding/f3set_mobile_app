#!/usr/bin/env python3
"""
Optimized F3Set Model - Cleaned for specific configuration:
- feature_arch: rny002_tsm
- temporal_arch: gru  
- use_ctx: true
- clip_len: 96
- num_classes: 29
- window: 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from contextlib import nullcontext
from tqdm import tqdm
import math

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.modules import GRU, GRUPrediction
from util.eval import non_maximum_suppression

HIDDEN_DIM = 768

class F3Set(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes=29, clip_len=96, step=2, window=5, device='cuda'):
            super().__init__()
            self._device = device
            self._num_classes = num_classes
            self._window = window
            self._require_clip_len = clip_len

            # RegNetY-002 with TSM - fixed architecture
            glb_feat = timm.create_model('regnety_002', pretrained=True)
            glb_feat_dim = glb_feat.head.fc.in_features
            glb_feat.head.fc = nn.Identity()

            # Add Temporal Shift Module
            make_temporal_shift(glb_feat, clip_len, is_gsm=False, step=step)

            self._glb_feat = glb_feat
            self._feat_dim = glb_feat_dim

            # GRU temporal head - single layer
            d_model = min(HIDDEN_DIM, self._feat_dim)
            self._head = GRU(self._feat_dim, d_model, num_layers=1)

            # Prediction heads
            self._coarse_pred = nn.Linear(d_model, 2)  # binary: background/foreground
            self._fine_pred = nn.Linear(d_model, num_classes)  # multi-label actions

            # Contextual refinement module
            self._ctx = GRUPrediction(num_classes + 1, num_classes + 1, d_model, num_layers=1)
            
        def forward(self, frame, coarse_label=None, fine_label=None, hand=None, max_seq_len=20):
            batch_size, true_clip_len, channels, height, width = frame.shape

            # Pad frame if needed for TSM
            clip_len = true_clip_len
            if true_clip_len < self._require_clip_len:
                padding = (0, 0, 0, 0, 0, 0, 0, self._require_clip_len - true_clip_len)
                frame = F.pad(frame, padding)
                clip_len = self._require_clip_len

            # Feature extraction with TSM-enabled RegNetY
            # Reshape for 2D CNN: (batch_size * clip_len, channels, height, width)
            frame_2d = frame.view(-1, channels, height, width)
            features_2d = self._glb_feat(frame_2d)
            
            # Reshape back to temporal: (batch_size, clip_len, feature_dim)
            im_feat = features_2d.reshape(batch_size, clip_len, -1)

            # Temporal modeling with GRU
            enc_feat = self._head(im_feat)

            # Prediction heads
            coarse_pred = self._coarse_pred(enc_feat)  # (batch_size, clip_len, 2)
            fine_pred = self._fine_pred(enc_feat)      # (batch_size, clip_len, num_classes)

            # Apply activations for contextual module
            coarse_pred_score = torch.softmax(coarse_pred, dim=2)
            fine_pred_score = torch.sigmoid(fine_pred)

            # Get shot boundaries via NMS
            if coarse_label is None:
                coarse_label = non_maximum_suppression(coarse_pred_score, self._window)
                coarse_label = torch.argmax(coarse_label, dim=2)
            else:
                coarse_label = coarse_pred_score * coarse_label.unsqueeze(-1)
                coarse_label = torch.argmax(coarse_label, dim=2)

            if fine_label is None:
                fine_label = fine_pred_score

            # Prepare sequences for contextual refinement
            seq_pred = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, 
                                 dtype=fine_label.dtype, device=self._device)
            seq_label = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, 
                                  dtype=fine_label.dtype, device=self._device)
            seq_mask = torch.ones((batch_size, max_seq_len), dtype=torch.bool, device=self._device)

            # Extract detected shots for each batch item
            for i in range(batch_size):
                # Get frames where shots are detected
                shot_frames = coarse_label[i].bool()
                selected_label = fine_label[i, shot_frames]
                selected_pred = fine_pred_score[i, shot_frames]
                
                num_shots = selected_label.shape[0]
                if num_shots > 0:
                    # Store shot sequences (action classes)
                    seq_label[i, :num_shots, 1:] = selected_label
                    seq_pred[i, :num_shots, 1:] = selected_pred
                    
                    # Store hand information for each shot
                    for j in range(num_shots):
                        # Use hand encoding for shot context
                        hand_idx = int(torch.round(selected_pred[j, 0]).clamp(0, 1))
                        seq_pred[i, j, 0] = hand[i, hand_idx]
                        hand_idx_label = int(torch.round(selected_label[j, 0]).clamp(0, 1))
                        seq_label[i, j, 0] = hand[i, hand_idx_label]
                    
                    # Update mask for valid sequences
                    seq_mask[i, :num_shots] = False

            # Contextual refinement
            seq_pred_refined = self._ctx(seq_pred)
            
            return coarse_pred, fine_pred, seq_pred_refined, seq_label, seq_mask

    def __init__(self, num_classes=29, clip_len=96, step=2, window=5, device='cuda'):
        self._device = device
        self._window = window
        self._num_classes = num_classes
        
        # Create optimized model implementation
        self._model = F3Set.Impl(
            num_classes=num_classes,
            clip_len=clip_len, 
            step=step,
            window=window,
            device=device
        )
        
        self._model.to(device)

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, 
              acc_grad_iter=1, fg_weight=5, epoch=0):
        """Training/validation epoch"""
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # Weighted loss for foreground class
        ce_weight = torch.FloatTensor([1, fg_weight]).to(self._device)

        epoch_loss = 0.
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                # Load batch data
                frame, _ = loader.dataset.load_frame_gpu(batch, self._device)
                coarse_label = batch['coarse_label'].to(self._device)
                fine_label = batch['fine_label'].float().to(self._device)
                hand = batch['hand'].float().to(self._device)

                # Forward pass
                with torch.cuda.amp.autocast():
                    coarse_pred, fine_pred, seq_pred, seq_label, seq_mask = self._model(
                        frame, coarse_label, fine_label, hand=hand
                    )

                    # Compute losses
                    loss = 0.
                    
                    # Coarse-grained binary classification loss
                    coarse_loss = F.cross_entropy(
                        coarse_pred.reshape(-1, 2), 
                        coarse_label.flatten(), 
                        weight=ce_weight
                    )
                    if not math.isnan(coarse_loss.item()):
                        loss += coarse_loss

                    # Fine-grained multi-label loss (masked by coarse detections)
                    fine_bce_loss = F.binary_cross_entropy_with_logits(
                        fine_pred, fine_label, reduction='none'
                    )
                    fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                    masked_fine_loss = fine_bce_loss * fine_mask
                    fine_loss = masked_fine_loss.sum() / fine_mask.sum()
                    if not math.isnan(fine_loss.item()):
                        loss += fine_loss
                    
                    # Contextual refinement loss
                    if (~seq_mask).any():  # If there are valid sequences
                        ctx_loss = F.binary_cross_entropy_with_logits(
                            seq_pred[~seq_mask], seq_label[~seq_mask]
                        )
                        if not math.isnan(ctx_loss.item()):
                            loss += ctx_loss

                # Backward pass
                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter, 
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
                
        return epoch_loss / len(loader)

    def predict(self, frame, hand, use_amp=True):
        """Inference method"""
        # Ensure tensor format
        if not isinstance(frame, torch.Tensor):
            frame = torch.FloatTensor(frame)
        if len(frame.shape) == 4:  # (L, C, H, W) -> (1, L, C, H, W)
            frame = frame.unsqueeze(0)
        
        frame = frame.to(self._device)
        hand = hand.to(self._device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                coarse_pred, fine_pred, seq_pred, _, _ = self._model(frame, hand=hand)
            
            # Apply NMS to coarse predictions
            coarse_pred_softmax = torch.softmax(coarse_pred, dim=2)
            coarse_pred_nms = non_maximum_suppression(coarse_pred_softmax, self._window)
            coarse_pred_cls = torch.argmax(coarse_pred_nms, dim=2)

            # Refine fine predictions with contextual module
            fine_pred_refine = fine_pred.clone()
            for i in range(coarse_pred_cls.size(0)):
                shot_id = 0
                for j in range(coarse_pred_cls.size(1)):
                    if coarse_pred_cls[i, j] == 1:
                        fine_pred_refine[i, j] = seq_pred[i, shot_id, 1:]
                        shot_id += 1

            # Apply sigmoid to get final probabilities
            fine_pred_final = torch.sigmoid(fine_pred_refine)
            
            return (
                coarse_pred_cls.cpu().numpy(), 
                coarse_pred_softmax.cpu().numpy(), 
                fine_pred_final.cpu().numpy()
            )

    def get_optimizer(self, opt_args):
        """Get optimizer and scaler for training"""
        return (
            torch.optim.AdamW(self._model.parameters(), **opt_args),
            torch.cuda.amp.GradScaler() if self._device == 'cuda' else None
        )

    def state_dict(self):
        """Get model state dict"""
        return {'model_state_dict': self._model.state_dict()}

    def load(self, state_dict):
        """Load model state dict"""
        self._model.load_state_dict(state_dict['model_state_dict'])


# Factory function for easy instantiation
def create_f3set_model(config, device='cuda'):
    """Create F3Set model from config dict"""
    return F3Set(
        num_classes=config['num_classes'],
        clip_len=config['clip_len'],
        step=config['stride'],
        window=config['window'],
        device=device
    )


# For backward compatibility with existing code
def F3SetOptimized(*args, **kwargs):
    """Alias for the optimized model"""
    return F3Set(*args, **kwargs)