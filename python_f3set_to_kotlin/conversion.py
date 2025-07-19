#!/usr/bin/env python3
"""
F3Set Mobile Conversion & Diagnostic Script - FIXED VERSION
Converts trained F3Set model to mobile format and validates the conversion
by comparing outputs between original and mobile models.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
import traceback
from torch.utils.data import DataLoader
import timm
from typing import Tuple, Optional

# Add current directory to path for imports
sys.path.append('.')

from train_f3set_f3ed import F3Set
from dataset.frame_process import ActionSeqVideoDataset
from util.io import load_json
from util.dataset import load_classes
from util.eval import non_maximum_suppression_np
from model.shift import make_temporal_shift
from model.modules import GRU, GRUPrediction


class F3SetMobileWrapper(nn.Module):
    """
    Mobile-optimized wrapper for F3Set model
    Designed for TorchScript conversion and mobile deployment
    """
    
    def __init__(self, 
                 num_classes: int = 29,
                 clip_len: int = 96, 
                 step: int = 2,
                 window: int = 5,
                 hidden_dim: int = 768,
                 device: str = 'cpu'):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        self.clip_len = clip_len
        self.window = window
        self.require_clip_len = clip_len
        self.device = device
        
        # Feature extractor: RegNetY-002 with TSM
        self.backbone = timm.create_model('regnety_002', pretrained=True)
        feat_dim = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()
        
        # Add Temporal Shift Module
        make_temporal_shift(self.backbone, clip_len, is_gsm=False, step=step)
        
        # Temporal head: Single-layer GRU
        d_model = min(hidden_dim, feat_dim)
        self.temporal_head = GRU(feat_dim, d_model, num_layers=1)
        
        # Prediction heads
        self.coarse_predictor = nn.Linear(d_model, 2)      # binary classification
        self.fine_predictor = nn.Linear(d_model, num_classes)  # multi-label
        
        # Fixed parameters for mobile
        self.max_seq_len = 20
    
    def extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract visual features using RegNetY + TSM"""
        batch_size, true_clip_len, channels, height, width = frame.shape
        
        # Pad frames if needed for TSM
        clip_len = true_clip_len
        if true_clip_len < self.require_clip_len:
            padding = (0, 0, 0, 0, 0, 0, 0, self.require_clip_len - true_clip_len)
            frame = F.pad(frame, padding)
            clip_len = self.require_clip_len
        
        # Reshape for 2D CNN: (batch_size * clip_len, channels, height, width)
        frame_2d = frame.view(-1, channels, height, width)
        features_2d = self.backbone(frame_2d)
        
        # Reshape back: (batch_size, clip_len, feature_dim)
        features = features_2d.view(batch_size, clip_len, -1)
        
        return features
    
    def apply_nms(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression - exact match to original implementation"""
        batch_size, seq_len, num_classes = scores.shape
        result = torch.zeros_like(scores, dtype=scores.dtype)
        
        # Process each batch independently
        for idx in range(batch_size):
            for i in range(seq_len):
                # Determine the window for this frame
                start = max(i - self.window // 2, 0)
                end = min(i + self.window // 2 + 1, seq_len)
                
                # Slice the window - looking at background scores (index 0)
                window = scores[idx, start:end, 0]
                
                # Get the MINIMUM score in the window (original uses torch.min)
                min_score = torch.min(window)
                
                # Check if current background score equals the minimum in window
                if scores[idx, i, 0] == min_score:
                    result[idx, i] = scores[idx, i]
        
        return result
    
    def forward(self, frame: torch.Tensor, hand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for mobile inference
        
        Args:
            frame: (batch_size, clip_len, 3, height, width)
            hand: (batch_size, 2) - hand encoding [far_hand_is_left, near_hand_is_left]
            
        Returns:
            coarse_cls: (batch_size, clip_len) - binary decisions after NMS
            coarse_scores: (batch_size, clip_len, 2) - background/foreground scores  
            fine_scores: (batch_size, clip_len, num_classes) - action probabilities
        """
        
        # 1. Extract visual features
        visual_features = self.extract_features(frame)
        
        # 2. Apply temporal modeling
        temporal_features = self.temporal_head(visual_features)
        
        # 3. Get predictions
        coarse_logits = self.coarse_predictor(temporal_features)
        fine_logits = self.fine_predictor(temporal_features)
        
        # 4. Apply activations
        coarse_scores = F.softmax(coarse_logits, dim=2)
        fine_scores = torch.sigmoid(fine_logits)
        
        # 5. Apply NMS to coarse predictions
        coarse_nms = self.apply_nms(coarse_scores)
        coarse_decisions = torch.argmax(coarse_nms, dim=2)
        
        # 6. For mobile, skip complex contextual refinement to avoid TorchScript issues
        return coarse_decisions, coarse_nms, fine_scores


class F3SetMobileConverter:
    """Helper class to convert trained F3Set model to mobile wrapper"""
    
    @staticmethod
    def convert_from_f3set(f3set_model: 'F3Set', 
                          num_classes: int = 29,
                          clip_len: int = 96,
                          step: int = 2, 
                          window: int = 5,
                          device: str = 'cpu') -> F3SetMobileWrapper:
        """
        Convert a trained F3Set model to mobile wrapper
        
        Args:
            f3set_model: Trained F3Set model
            num_classes, clip_len, step, window: Model configuration
            device: Target device for mobile model
            
        Returns:
            F3SetMobileWrapper with transferred weights
        """
        
        # Create mobile wrapper
        mobile_wrapper = F3SetMobileWrapper(
            num_classes=num_classes,
            clip_len=clip_len, 
            step=step,
            window=window,
            device=device
        )
        
        # Move to target device
        mobile_wrapper = mobile_wrapper.to(device)
        
        # Get the internal model
        if hasattr(f3set_model._model, 'module'):
            # Handle DataParallel
            source_model = f3set_model._model.module
        else:
            source_model = f3set_model._model
        
        # Transfer weights
        print("Transferring weights...")
        
        try:
            # Feature extractor (backbone)
            mobile_wrapper.backbone.load_state_dict(source_model._glb_feat.state_dict())
            print("✅ Backbone weights transferred")
            
            # Temporal head
            mobile_wrapper.temporal_head.load_state_dict(source_model._head.state_dict())
            print("✅ Temporal head weights transferred")
            
            # Prediction heads
            mobile_wrapper.coarse_predictor.load_state_dict(source_model._coarse_pred.state_dict())
            mobile_wrapper.fine_predictor.load_state_dict(source_model._fine_pred.state_dict())
            print("✅ Prediction heads weights transferred")
            
        except Exception as e:
            print(f"❌ Weight transfer failed: {e}")
            raise
        
        print("✅ Weight transfer complete!")
        
        return mobile_wrapper


class F3SetMobileDiagnostic:
    """Wrapper for mobile model to match original model interface"""
    
    def __init__(self, mobile_model_path: str, device: str = 'cpu'):
        self.device = device
        if mobile_model_path.endswith('.ptl'):
            try:
                from torch.jit.mobile import _load_for_lite_interpreter
                self.model = _load_for_lite_interpreter(mobile_model_path)
            except ImportError:
                print("⚠️ Lite interpreter not available, trying regular JIT load")
                self.model = torch.jit.load(mobile_model_path, map_location=device)
        else:
            self.model = torch.jit.load(mobile_model_path, map_location=device)
    
    def predict(self, frame, hand, use_amp=True):
        """Match the original model's predict interface"""
        with torch.no_grad():
            # Ensure correct input shapes for mobile model
            if frame.dim() == 4:  # (L, C, H, W)
                frame = frame.unsqueeze(0)  # Add batch dimension -> (1, L, C, H, W)
            if hand.dim() == 1:  # (2,)
                hand = hand.unsqueeze(0)    # Add batch dimension -> (1, 2)
            
            # Ensure correct data types and device
            frame = frame.float().to(self.device)
            hand = hand.float().to(self.device)
            
            # Run mobile model
            outputs = self.model(frame, hand)
            
            # Extract outputs - mobile model returns (coarse_cls, coarse_scores, fine_scores)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                coarse_cls = outputs[0].cpu().numpy()
                coarse_scores = outputs[1].cpu().numpy() 
                fine_scores = outputs[2].cpu().numpy()
            else:
                raise ValueError(f"Unexpected mobile model output format: {type(outputs)}")
            
            return coarse_cls, coarse_scores, fine_scores


def load_test_clip(dataset, video_name=None, clip_index=0):
    """Load a specific test clip for comparison"""
    clips = []
    target_video = video_name
    
    # Collect clips
    for i, clip in enumerate(DataLoader(dataset, batch_size=1, num_workers=0)):
        if target_video is None or clip['video'][0] == target_video:
            clips.append({
                'index': i,
                'video': clip['video'][0],
                'start': clip['start'][0].item(),
                'frame': clip['frame'][0],  # Remove batch dimension
                'hand': clip['hand'][0]     # Remove batch dimension
            })
            
            if len(clips) > clip_index:
                break
    
    if not clips:
        raise ValueError(f"No clips found for video: {target_video}")
    
    if clip_index >= len(clips):
        print(f"Warning: Requested clip {clip_index} but only {len(clips)} available")
        clip_index = 0
    
    selected_clip = clips[clip_index]
    
    print(f"Selected clip info:")
    print(f"  Video: {selected_clip['video']}")
    print(f"  Start frame: {selected_clip['start']}")
    print(f"  Frame tensor shape: {selected_clip['frame'].shape}")
    print(f"  Hand tensor shape: {selected_clip['hand'].shape}")
    print(f"  Hand tensor values: {selected_clip['hand']}")
    
    return selected_clip


def analyze_input_preprocessing(frame_tensor, hand_tensor):
    """Analyze input preprocessing to ensure consistency"""
    print("\n" + "="*60)
    print("INPUT PREPROCESSING ANALYSIS")
    print("="*60)
    
    # Frame tensor analysis
    frame_data = frame_tensor.cpu().numpy()
    print(f"Frame tensor: shape={frame_tensor.shape}, dtype={frame_tensor.dtype}, device={frame_tensor.device}")
    print(f"Frame stats: min={frame_data.min():.6f}, max={frame_data.max():.6f}, "
          f"mean={frame_data.mean():.6f}, std={frame_data.std():.6f}")
    
    # Check for ImageNet normalization - adjusted range for actual ImageNet stats
    # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # After normalization: (x - mean) / std, typical range is roughly [-2.2, 2.7]
    expected_range = (-2.5, 3.0)
    if frame_data.min() < expected_range[0] or frame_data.max() > expected_range[1]:
        print("⚠️  Frame values outside typical ImageNet normalized range!")
        print(f"   Expected range: {expected_range}, Actual range: ({frame_data.min():.3f}, {frame_data.max():.3f})")
    else:
        print("✅ Frame values within expected ImageNet normalized range")
    
    # Hand tensor analysis
    hand_data = hand_tensor.cpu().numpy().flatten()
    print(f"Hand tensor: shape={hand_tensor.shape}, dtype={hand_tensor.dtype}, device={hand_tensor.device}")
    print(f"Hand values: {hand_data}")
    
    # Validate hand encoding
    if len(hand_data) != 2:
        print("⚠️  Hand tensor should have exactly 2 values!")
    if not all(val in [0.0, 1.0] for val in hand_data):
        print("⚠️  Hand values should be binary (0.0 or 1.0)!")
    else:
        print("✅ Hand tensor format is correct")


def compare_model_outputs(orig_outputs, mobile_outputs, frame_offset=0, tolerance=1e-4):
    """Detailed comparison of model outputs with debugging"""
    print("\n" + "="*60)
    print("MODEL OUTPUT COMPARISON")
    print("="*60)
    
    orig_coarse_cls, orig_coarse_scores, orig_fine_scores = orig_outputs
    mobile_coarse_cls, mobile_coarse_scores, mobile_fine_scores = mobile_outputs
    
    # Shape comparison
    print("SHAPE COMPARISON:")
    print(f"Original - Coarse cls: {orig_coarse_cls.shape}, Coarse scores: {orig_coarse_scores.shape}, Fine scores: {orig_fine_scores.shape}")
    print(f"Mobile   - Coarse cls: {mobile_coarse_cls.shape}, Coarse scores: {mobile_coarse_scores.shape}, Fine scores: {mobile_fine_scores.shape}")
    
    # Validate shapes
    if orig_coarse_scores.shape != mobile_coarse_scores.shape:
        print("⚠️  Coarse scores shape mismatch!")
        return None
    if orig_fine_scores.shape != mobile_fine_scores.shape:
        print("⚠️  Fine scores shape mismatch!")
        return None
    
    # Check for specific frames
    print(f"\nDETAILED FRAME ANALYSIS:")
    check_frames = [70, 71, 72] + list(range(0, min(96, orig_coarse_scores.shape[1]), 10))
    check_frames = sorted(set(check_frames))
    
    problem_frames = []
    zero_frames = []
    
    for frame_idx in check_frames:
        if frame_idx < orig_coarse_scores.shape[1]:
            orig_bg, orig_fg = orig_coarse_scores[0, frame_idx, 0], orig_coarse_scores[0, frame_idx, 1]
            mobile_bg, mobile_fg = mobile_coarse_scores[0, frame_idx, 0], mobile_coarse_scores[0, frame_idx, 1]
            
            bg_diff = abs(orig_bg - mobile_bg)
            fg_diff = abs(orig_fg - mobile_fg)
            
            status = ""
            if mobile_bg == 0.0 and mobile_fg == 0.0:
                status = "❌ ZERO_OUTPUT"
                zero_frames.append(frame_idx)
            elif bg_diff > 0.1 or fg_diff > 0.1:
                status = "🔴 LARGE_DIFF"
                problem_frames.append(frame_idx)
            elif bg_diff > 0.01 or fg_diff > 0.01:
                status = "⚠️  MEDIUM_DIFF"
            
            print(f"  Frame {frame_idx:2d}: Orig=[{orig_bg:.6f}, {orig_fg:.6f}] Mobile=[{mobile_bg:.6f}, {mobile_fg:.6f}] Diff=[{bg_diff:.6f}, {fg_diff:.6f}] {status}")
    
    # Numerical comparison
    coarse_diff = np.abs(orig_coarse_scores - mobile_coarse_scores)
    fine_diff = np.abs(orig_fine_scores - mobile_fine_scores)
    
    print(f"\nNUMERICAL DIFFERENCES:")
    print(f"Coarse scores - Max: {coarse_diff.max():.6f}, Mean: {coarse_diff.mean():.6f}")
    print(f"Fine scores   - Max: {fine_diff.max():.6f}, Mean: {fine_diff.mean():.6f}")
    
    # Detection comparison
    detection_mismatches = []
    batch_size, num_frames = orig_coarse_scores.shape[:2]
    
    for b in range(batch_size):
        for f in range(num_frames):
            orig_detection = 1 if orig_coarse_scores[b, f, 1] > orig_coarse_scores[b, f, 0] else 0
            mobile_detection = 1 if mobile_coarse_scores[b, f, 1] > mobile_coarse_scores[b, f, 0] else 0
            
            if orig_detection != mobile_detection:
                detection_mismatches.append({
                    'frame': frame_offset + f,
                    'orig': orig_detection,
                    'mobile': mobile_detection
                })
    
    print(f"Detection mismatches: {len(detection_mismatches)}/{num_frames} frames")
    
    return {
        'coarse_diff_max': float(coarse_diff.max()),
        'coarse_diff_mean': float(coarse_diff.mean()),
        'fine_diff_max': float(fine_diff.max()),
        'fine_diff_mean': float(fine_diff.mean()),
        'detection_mismatches': len(detection_mismatches),
        'total_frames': num_frames,
        'tolerance_exceeded': coarse_diff.max() > tolerance or fine_diff.max() > tolerance,
        'zero_frames_count': len(zero_frames),
        'problem_frames': problem_frames
    }


def generate_diagnosis(comparison_results):
    """Generate diagnosis of the discrepancies"""
    diagnosis = []
    
    if comparison_results['zero_frames_count'] > 0:
        diagnosis.append(f"🔴 CRITICAL: {comparison_results['zero_frames_count']} frames with zero output")
        diagnosis.append("   Likely causes: Model not properly converted, device mismatch, or inference failure")
    
    if comparison_results['coarse_diff_max'] > 0.1:
        diagnosis.append("🔴 CRITICAL: Large coarse score differences (>0.1)")
        diagnosis.append("   Possible causes: Weight conversion error, different activations")
    elif comparison_results['coarse_diff_max'] > 0.01:
        diagnosis.append("⚠️  WARNING: Moderate coarse score differences (>0.01)")
        diagnosis.append("   Possible causes: Numerical precision, optimization differences")
    elif comparison_results['coarse_diff_max'] > 0.001:
        diagnosis.append("ℹ️  INFO: Small coarse score differences (>0.001)")
    else:
        diagnosis.append("✅ EXCELLENT: Coarse scores are numerically very close")
    
    mismatch_ratio = comparison_results['detection_mismatches'] / comparison_results['total_frames']
    if mismatch_ratio > 0.1:
        diagnosis.append(f"🔴 CRITICAL: {mismatch_ratio*100:.1f}% detection mismatches")
    elif mismatch_ratio > 0.05:
        diagnosis.append(f"⚠️  WARNING: {mismatch_ratio*100:.1f}% detection mismatches")
    elif mismatch_ratio > 0:
        diagnosis.append(f"ℹ️  INFO: {mismatch_ratio*100:.1f}% detection mismatches")
    else:
        diagnosis.append("✅ PERFECT: No detection mismatches")
    
    return diagnosis


def convert_and_validate_model(original_model, config, test_clip, classes, output_path):
    """Complete conversion and validation pipeline with proper device handling"""
    print("\n" + "="*80)
    print("F3SET MOBILE CONVERSION & VALIDATION")
    print("="*80)
    
    # Determine device from original model
    original_device = next(original_model._model.parameters()).device
    print(f"Original model device: {original_device}")
    
    # For mobile, we'll use CPU
    target_device = 'cpu'
    print(f"Target mobile device: {target_device}")
    
    # Move original model to CPU for fair comparison
    print("Moving original model to CPU...")
    original_model._model = original_model._model.to(target_device)
    original_model._device = target_device
    
    # Ensure ALL model components are on CPU (some might be missed)
    def move_to_cpu_recursive(module):
        for child in module.children():
            move_to_cpu_recursive(child)
        for param in module.parameters(recurse=False):
            param.data = param.data.to(target_device)
        for buffer in module.buffers(recurse=False):
            buffer.data = buffer.data.to(target_device)
    
    move_to_cpu_recursive(original_model._model)
    
    # CRITICAL: Update the model's internal device attribute
    # The predict method might use this to move tensors
    if hasattr(original_model._model, '_device'):
        original_model._model._device = target_device
    if hasattr(original_model._model, 'module') and hasattr(original_model._model.module, '_device'):
        original_model._model.module._device = target_device
    
    # Force model to eval mode on CPU
    original_model._model.eval()
    
    # Verify all tensors are on CPU
    cuda_tensors = []
    for name, param in original_model._model.named_parameters():
        if param.device.type == 'cuda':
            cuda_tensors.append(f"param: {name}")
    for name, buffer in original_model._model.named_buffers():
        if buffer.device.type == 'cuda':
            cuda_tensors.append(f"buffer: {name}")
    
    if cuda_tensors:
        print(f"⚠️  Found {len(cuda_tensors)} tensors still on CUDA:")
        for tensor_info in cuda_tensors[:5]:  # Show first 5
            print(f"    {tensor_info}")
        if len(cuda_tensors) > 5:
            print(f"    ... and {len(cuda_tensors) - 5} more")
    else:
        print("✅ All model tensors confirmed on CPU")
    
    print(f"Model _device attribute: {getattr(original_model, '_device', 'not found')}")
    print(f"Internal model _device: {getattr(original_model._model, '_device', 'not found')}")
    
    # Prepare test inputs
    frame_tensor = test_clip['frame']
    hand_tensor = test_clip['hand']
    
    # Ensure proper shapes and types
    if frame_tensor.dim() == 4:  # (L, C, H, W) -> (1, L, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)
    if hand_tensor.dim() == 1:  # (2,) -> (1, 2)
        hand_tensor = hand_tensor.unsqueeze(0)
    
    # Move to target device and convert to float
    frame_tensor = frame_tensor.float().to(target_device)
    hand_tensor = hand_tensor.float().to(target_device)
    
    analyze_input_preprocessing(frame_tensor, hand_tensor)
    
    # Test original model with proper input format - try different combinations
    print("\n1. Testing original model...")
    orig_frame = frame_tensor.squeeze(0).to(target_device)  # (L, C, H, W) and ensure on CPU
    
    # Try different hand tensor formats to find what works
    hand_formats = [
        (hand_tensor.squeeze(0).to(target_device), "1D hand tensor (2,)"),
        (hand_tensor.to(target_device), "2D hand tensor (1, 2)"),
        (hand_tensor.squeeze(0).unsqueeze(0).to(target_device), "2D hand tensor reshaped"),
    ]
    
    orig_outputs = None
    orig_time = 0
    working_hand = None
    
    for test_hand, desc in hand_formats:
        try:
            print(f"  Trying {desc}: frame={orig_frame.shape}, hand={test_hand.shape}")
            print(f"    Frame device: {orig_frame.device}, Hand device: {test_hand.device}")
            
            # Double-check devices before predict call
            assert orig_frame.device.type == 'cpu', f"Frame unexpectedly on {orig_frame.device}"
            assert test_hand.device.type == 'cpu', f"Hand unexpectedly on {test_hand.device}"
            
            # Clone tensors to ensure they stay on CPU
            frame_cpu = orig_frame.clone().detach().to('cpu')
            hand_cpu = test_hand.clone().detach().to('cpu')
            
            print(f"    Cloned tensors - Frame: {frame_cpu.device}, Hand: {hand_cpu.device}")
            
            start_time = time.time()
            orig_outputs = original_model.predict(frame_cpu, hand_cpu)
            orig_time = (time.time() - start_time) * 1000
            print(f"  ✅ Success with {desc}")
            working_hand = test_hand
            break
        except Exception as e:
            print(f"  ❌ Failed with {desc}: {str(e)[:100]}...")
            # Let's see the full error for device issues
            if "device" in str(e).lower():
                print(f"    Full device error: {str(e)}")
            continue
    
    if orig_outputs is None:
        print("❌ Original model failed with all hand tensor formats")
        return None
        
    print(f"✅ Original model inference: {orig_time:.2f}ms")
    print(f"Output shapes: coarse_cls={orig_outputs[0].shape}, coarse_scores={orig_outputs[1].shape}, fine_scores={orig_outputs[2].shape}")
    
    # Check frame 71
    if orig_outputs[1].shape[1] > 71:
        scores_71 = orig_outputs[1][0, 71]
        print(f"Frame 71 original scores: [{scores_71[0]:.6f}, {scores_71[1]:.6f}]")
    # Convert to mobile wrapper
    print("\n2. Converting to mobile wrapper...")
    mobile_wrapper = F3SetMobileConverter.convert_from_f3set(
        original_model,
        num_classes=len(classes),
        clip_len=config['clip_len'],
        step=config['stride'], 
        window=config['window'],
        device=target_device
    )
    
    # Test mobile wrapper
    print("\n3. Testing mobile wrapper...")
    mobile_wrapper.eval()
    with torch.no_grad():
        start_time = time.time()
        # Use the 5D frame tensor and 2D hand tensor for mobile wrapper
        mobile_outputs = mobile_wrapper(frame_tensor, hand_tensor)
        mobile_time = (time.time() - start_time) * 1000
        
        # Convert to numpy for comparison
        mobile_outputs_np = tuple(out.cpu().numpy() for out in mobile_outputs)
    
    print(f"✅ Mobile wrapper inference: {mobile_time:.2f}ms")
    print(f"Output shapes: coarse_cls={mobile_outputs_np[0].shape}, coarse_scores={mobile_outputs_np[1].shape}, fine_scores={mobile_outputs_np[2].shape}")
    
    # Check frame 71 in mobile output
    if mobile_outputs_np[1].shape[1] > 71:
        scores_71 = mobile_outputs_np[1][0, 71]
        print(f"Frame 71 mobile scores: [{scores_71[0]:.6f}, {scores_71[1]:.6f}]")
    
    # Compare outputs
    print("\n4. Comparing outputs...")
    comparison_results = compare_model_outputs(orig_outputs, mobile_outputs_np, test_clip['start'])
    
    if comparison_results is None:
        print("❌ Comparison failed due to shape mismatch")
        return None
    
    # Generate diagnosis
    diagnosis = generate_diagnosis(comparison_results)
    
    print("\n5. Diagnosis:")
    for item in diagnosis:
        print(f"  {item}")
    
    # Export if validation successful
    if not comparison_results['tolerance_exceeded'] and comparison_results['zero_frames_count'] == 0:
        print("\n6. Exporting mobile model...")
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(mobile_wrapper, (frame_tensor, hand_tensor))
            
            # Save regular TorchScript
            torch.jit.save(traced_model, output_path)
            print(f"✅ TorchScript model saved: {output_path}")
            
            # Save lite interpreter version
            lite_path = output_path.replace('.pt', '.ptl')
            traced_model._save_for_lite_interpreter(lite_path)
            print(f"✅ Lite model saved: {lite_path}")
            
            # Test the traced model
            print("\n7. Testing traced model...")
            mobile_diagnostic = F3SetMobileDiagnostic(output_path, device=target_device)
            
            # Use the same format that worked for original model
            test_frame_for_diag = orig_frame
            test_hand_for_diag = working_hand if working_hand.dim() == 1 else working_hand.squeeze(0)
            
            traced_outputs = mobile_diagnostic.predict(test_frame_for_diag, test_hand_for_diag)
            
            # Compare traced vs wrapper
            traced_comparison = compare_model_outputs(mobile_outputs_np, traced_outputs, test_clip['start'], tolerance=1e-6)
            
            if traced_comparison and not traced_comparison['tolerance_exceeded']:
                print("✅ Traced model matches wrapper output")
            else:
                print("⚠️  Traced model differs from wrapper")
            
            # Save metadata
            metadata_path = output_path.replace('.pt', '_metadata.json')
            metadata = {
                'model_info': {
                    'dataset': config['dataset'],
                    'num_classes': len(classes),
                    'classes': {name: idx for idx, name in enumerate(['background'] + list(classes.keys()))},
                    'feature_arch': config['feature_arch'],
                    'temporal_arch': config['temporal_arch'],
                    'epoch': 0,  # Will be filled by main
                    'use_ctx': False  # Mobile version doesn't use context
                },
                'input_config': {
                    'clip_len': config['clip_len'],
                    'crop_dim': config['crop_dim'],
                    'stride': config['stride'],
                    'window': config['window'],
                    'max_seq_len': 20
                },
                'mobile_config': {
                    'optimized': True,
                    'quantized': False,
                    'batch_size': 1
                },
                'version': '2.0'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved: {metadata_path}")
            
            return {
                'mobile_wrapper': mobile_wrapper,
                'traced_model': traced_model,
                'comparison_results': comparison_results,
                'diagnosis': diagnosis,
                'inference_times': {'original': orig_time, 'mobile': mobile_time},
                'output_files': {'torchscript': output_path, 'lite': lite_path, 'metadata': metadata_path}
            }
            
        except Exception as e:
            print(f"❌ Model export failed: {e}")
            traceback.print_exc()
            return None
    else:
        print("❌ Validation failed - not exporting model")
        print(f"   Tolerance exceeded: {comparison_results['tolerance_exceeded']}")
        print(f"   Zero frames: {comparison_results['zero_frames_count']}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert and validate F3Set mobile model")
    parser.add_argument('model_dir', help='Path to original model directory')
    parser.add_argument('frame_dir', help='Path to frame directory')
    parser.add_argument('-o', '--output', default='f3set_mobile.pt', 
                        help='Output path for mobile model')
    parser.add_argument('-s', '--split', choices=['train', 'val', 'test', 'test_one'],
                        default='test_one', help='Dataset split')
    parser.add_argument('-d', '--dataset', help='Dataset name')
    parser.add_argument('--video_name', help='Specific video name to test')
    parser.add_argument('--clip_index', type=int, default=0, 
                        help='Clip index to test (default: 0)')
    parser.add_argument('--epoch', type=int, help='Specific epoch to use')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for detecting significant differences')
    parser.add_argument('--report', help='Save diagnostic report to file')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config_path = os.path.join(args.model_dir, 'config.json')
        config = load_json(config_path)
        
        # Determine epoch
        if args.epoch:
            best_epoch = args.epoch
        elif os.path.isfile(os.path.join(args.model_dir, 'loss.json')):
            data = load_json(os.path.join(args.model_dir, 'loss.json'))
            best = max(data, key=lambda x: x.get('val_edit', 0))
            best_epoch = best['epoch']
        else:
            import re
            regex = re.compile(r'checkpoint_(\d+)\.pt')
            last_epoch = -1
            for file_name in os.listdir(args.model_dir):
                m = regex.match(file_name)
                if m:
                    epoch = int(m.group(1))
                    last_epoch = max(last_epoch, epoch)
            best_epoch = last_epoch
        
        print(f"Using epoch: {best_epoch}")
        
        # Get dataset
        dataset = args.dataset if args.dataset else config['dataset']
        
        # Load classes
        classes = load_classes(os.path.join('data', dataset, 'elements.txt'))
        
        # Load original model
        print("Loading original model...")
        original_model = F3Set(
            len(classes), 
            config['feature_arch'], 
            config['temporal_arch'], 
            clip_len=config['clip_len'],
            step=config['stride'], 
            window=config['window'], 
            use_ctx=config['use_ctx'],
            multi_gpu=False
        )
        
        checkpoint_path = os.path.join(args.model_dir, f'checkpoint_{best_epoch:03d}.pt')
        original_model.load(torch.load(checkpoint_path, map_location='cpu'))
        
        # Create dataset
        split_path = os.path.join('data', dataset, f'{args.split}.json')
        video_data = ActionSeqVideoDataset(
            classes, split_path, args.frame_dir, config['clip_len'],
            overlap_len=config['clip_len'] // 2, 
            crop_dim=config['crop_dim'],
            stride=config['stride']
        )
        
        # Load test clip
        test_clip = load_test_clip(video_data, args.video_name, args.clip_index)
        
        # Convert and validate
        results = convert_and_validate_model(
            original_model, config, test_clip, classes, args.output
        )
        
        if results:
            print("\n" + "="*80)
            print("CONVERSION SUMMARY")
            print("="*80)
            
            print("✅ Conversion completed successfully!")
            print(f"Original model inference: {results['inference_times']['original']:.2f}ms")
            print(f"Mobile model inference: {results['inference_times']['mobile']:.2f}ms")
            print(f"Speedup: {results['inference_times']['original']/results['inference_times']['mobile']:.2f}x")
            
            print(f"\nOutput files:")
            for format_name, path in results['output_files'].items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path) / (1024 * 1024)
                    print(f"  {format_name.capitalize()}: {path} ({file_size:.1f} MB)")
            
            print(f"\nValidation results:")
            comp = results['comparison_results']
            print(f"  Max coarse difference: {comp['coarse_diff_max']:.6f}")
            print(f"  Max fine difference: {comp['fine_diff_max']:.6f}")
            print(f"  Detection mismatches: {comp['detection_mismatches']}/{comp['total_frames']}")
            print(f"  Zero frames: {comp['zero_frames_count']}")
            
            print(f"\nDiagnosis:")
            for item in results['diagnosis']:
                print(f"  {item}")
            
            # Save diagnostic report if requested
            if args.report:
                report = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'video_name': test_clip['video'],
                        'clip_start': test_clip['start'],
                        'epoch': best_epoch,
                        'config': config
                    },
                    'conversion_results': {
                        'successful': True,
                        'output_files': results['output_files'],
                        'inference_times': results['inference_times']
                    },
                    'validation_results': results['comparison_results'],
                    'diagnosis': results['diagnosis'],
                    'pytorch_version': torch.__version__
                }
                
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\n📄 Diagnostic report saved: {args.report}")
        
        else:
            print("\n❌ Conversion failed validation")
            return 1
        
        print("\n🎉 Mobile conversion and validation complete!")
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())