#!/usr/bin/env python3
"""
Enhanced F3Set Mobile vs Original Model Diagnostic Script
Compares PyTorch original model with mobile-converted model to identify inference discrepancies
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import time
from datetime import datetime
import traceback
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('.')

from train_f3set_f3ed import F3Set
from dataset.frame_process import ActionSeqVideoDataset
from util.io import load_json
from util.dataset import load_classes
from util.eval import non_maximum_suppression_np

class F3SetMobileModelDiagnostic:
    """Wrapper for mobile model to match original model interface"""
    
    def __init__(self, mobile_model_path, device='cpu'):
        self.device = device
        if mobile_model_path.endswith('.ptl'):
            from torch.jit.mobile import _load_for_lite_interpreter
            self.model = _load_for_lite_interpreter(mobile_model_path)
        else:
            self.model = torch.jit.load(mobile_model_path, map_location=device)
    
    def predict(self, frame, hand, use_amp=True):
        """Match the original model's predict interface"""
        with torch.no_grad():
            # Debug input shapes
            print(f"Mobile model input - frame: {frame.shape}, hand: {hand.shape}")
            
            # Ensure correct input shapes for mobile model
            if frame.dim() == 4:  # (L, C, H, W)
                frame = frame.unsqueeze(0)  # Add batch dimension -> (1, L, C, H, W)
            if hand.dim() == 1:  # (2,)
                hand = hand.unsqueeze(0)    # Add batch dimension -> (1, 2)
            
            # Ensure correct data types
            frame = frame.float()
            hand = hand.float()
            
            # Move to correct device
            frame = frame.to(self.device)
            hand = hand.to(self.device)
            
            print(f"Mobile model adjusted input - frame: {frame.shape}, hand: {hand.shape}")
            print(f"Mobile model input dtypes - frame: {frame.dtype}, hand: {hand.dtype}")
            
            # Run mobile model
            try:
                outputs = self.model(frame, hand)
                print(f"Mobile model raw output type: {type(outputs)}")
                
                if isinstance(outputs, tuple):
                    print(f"Mobile model output tuple length: {len(outputs)}")
                    for i, out in enumerate(outputs):
                        print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
                
                # Extract outputs - mobile model returns (coarse_cls, coarse_scores, fine_scores)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    coarse_cls = outputs[0].cpu().numpy()
                    coarse_scores = outputs[1].cpu().numpy() 
                    fine_scores = outputs[2].cpu().numpy()
                else:
                    raise ValueError(f"Unexpected mobile model output format: {type(outputs)}, length: {len(outputs) if isinstance(outputs, tuple) else 'N/A'}")
                
                print(f"Mobile model extracted shapes - cls: {coarse_cls.shape}, scores: {coarse_scores.shape}, fine: {fine_scores.shape}")
                
                return coarse_cls, coarse_scores, fine_scores
                
            except Exception as e:
                print(f"❌ Mobile model forward pass failed: {e}")
                print(f"Input shapes were - frame: {frame.shape}, hand: {hand.shape}")
                raise

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
    
    # Debug tensor info
    print(f"Selected clip info:")
    print(f"  Video: {selected_clip['video']}")
    print(f"  Start frame: {selected_clip['start']}")
    print(f"  Frame tensor shape: {selected_clip['frame'].shape}")
    print(f"  Frame tensor dtype: {selected_clip['frame'].dtype}")
    print(f"  Hand tensor shape: {selected_clip['hand'].shape}")
    print(f"  Hand tensor dtype: {selected_clip['hand'].dtype}")
    print(f"  Hand tensor values: {selected_clip['hand']}")
    
    return selected_clip

def analyze_input_preprocessing(frame_tensor, hand_tensor):
    """Analyze input preprocessing to ensure consistency"""
    print("\n" + "="*60)
    print("INPUT PREPROCESSING ANALYSIS")
    print("="*60)
    
    # Frame tensor analysis
    print(f"Frame tensor shape: {frame_tensor.shape}")
    print(f"Frame tensor dtype: {frame_tensor.dtype}")
    
    frame_data = frame_tensor.cpu().numpy()
    print(f"Frame stats:")
    print(f"  Min: {frame_data.min():.6f}")
    print(f"  Max: {frame_data.max():.6f}")
    print(f"  Mean: {frame_data.mean():.6f}")
    print(f"  Std: {frame_data.std():.6f}")
    
    # Check for ImageNet normalization
    expected_range = (-2.5, 2.5)  # Rough range after ImageNet normalization
    if frame_data.min() < expected_range[0] or frame_data.max() > expected_range[1]:
        print("⚠️  Frame values outside expected ImageNet normalized range!")
    
    # Hand tensor analysis
    print(f"\nHand tensor shape: {hand_tensor.shape}")
    print(f"Hand tensor dtype: {hand_tensor.dtype}")
    print(f"Hand values: {hand_tensor.cpu().numpy()}")
    
    # Validate hand encoding
    hand_data = hand_tensor.cpu().numpy().flatten()
    if len(hand_data) != 2:
        print("⚠️  Hand tensor should have exactly 2 values!")
    if not all(val in [0.0, 1.0] for val in hand_data):
        print("⚠️  Hand values should be binary (0.0 or 1.0)!")

def debug_original_model_inputs(original_model, frame_tensor, hand_tensor):
    """Debug what input shapes the original model expects"""
    print("\n" + "="*60)
    print("DEBUGGING ORIGINAL MODEL INPUT REQUIREMENTS")
    print("="*60)
    
    print(f"Testing different input combinations...")
    
    # Test different frame tensor shapes
    frame_variations = []
    if frame_tensor.dim() == 4:  # (L, C, H, W)
        frame_variations.append(("4D (L,C,H,W)", frame_tensor))
        frame_variations.append(("5D (1,L,C,H,W)", frame_tensor.unsqueeze(0)))
    elif frame_tensor.dim() == 5:  # (1, L, C, H, W)
        frame_variations.append(("5D (1,L,C,H,W)", frame_tensor))
        frame_variations.append(("4D (L,C,H,W)", frame_tensor.squeeze(0)))
    
    # Test different hand tensor shapes
    hand_variations = []
    if hand_tensor.dim() == 1:  # (2,)
        hand_variations.append(("1D (2,)", hand_tensor))
        hand_variations.append(("2D (1,2)", hand_tensor.unsqueeze(0)))
        hand_variations.append(("1D float", hand_tensor.float()))
        hand_variations.append(("2D float", hand_tensor.unsqueeze(0).float()))
    elif hand_tensor.dim() == 2:  # (1, 2)
        hand_variations.append(("2D (1,2)", hand_tensor))
        hand_variations.append(("1D (2,)", hand_tensor.squeeze(0)))
        hand_variations.append(("2D float", hand_tensor.float()))
        hand_variations.append(("1D float", hand_tensor.squeeze(0).float()))
    
    successful_combinations = []
    
    for frame_desc, frame_var in frame_variations:
        for hand_desc, hand_var in hand_variations:
            try:
                print(f"Trying: frame {frame_desc} {frame_var.shape} {frame_var.dtype}, hand {hand_desc} {hand_var.shape} {hand_var.dtype}")
                
                # Test the prediction
                outputs = original_model.predict(frame_var, hand_var)
                
                print(f"  ✅ SUCCESS!")
                successful_combinations.append((frame_desc, hand_desc, frame_var.shape, hand_var.shape))
                
                # Return the first successful combination
                return frame_var, hand_var, outputs
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:100]}...")
                continue
    
    if successful_combinations:
        print(f"\nSuccessful combinations found:")
        for frame_desc, hand_desc, frame_shape, hand_shape in successful_combinations:
            print(f"  Frame: {frame_desc} {frame_shape}, Hand: {hand_desc} {hand_shape}")
    else:
        print(f"\n❌ No successful input combinations found!")
        
        # Try to inspect the model forward method
        print(f"\nTrying to inspect model structure...")
        try:
            if hasattr(original_model, '_model'):
                if hasattr(original_model._model, 'forward'):
                    import inspect
                    sig = inspect.signature(original_model._model.forward)
                    print(f"Model forward signature: {sig}")
        except:
            pass
    
    return None, None, None

def compare_model_outputs(orig_outputs, mobile_outputs, frame_offset=0, tolerance=1e-4):
    """Detailed comparison of model outputs"""
    print("\n" + "="*60)
    print("MODEL OUTPUT COMPARISON")
    print("="*60)
    
    orig_coarse_cls, orig_coarse_scores, orig_fine_scores = orig_outputs
    mobile_coarse_cls, mobile_coarse_scores, mobile_fine_scores = mobile_outputs
    
    # Shape comparison
    print("SHAPE COMPARISON:")
    print(f"Original - Coarse cls: {orig_coarse_cls.shape}, Coarse scores: {orig_coarse_scores.shape}, Fine scores: {orig_fine_scores.shape}")
    print(f"Mobile   - Coarse cls: {mobile_coarse_cls.shape}, Coarse scores: {mobile_coarse_scores.shape}, Fine scores: {mobile_fine_scores.shape}")
    
    shape_mismatch = False
    if orig_coarse_scores.shape != mobile_coarse_scores.shape:
        print("⚠️  Coarse scores shape mismatch!")
        shape_mismatch = True
    if orig_fine_scores.shape != mobile_fine_scores.shape:
        print("⚠️  Fine scores shape mismatch!")
        shape_mismatch = True
    
    if shape_mismatch:
        print("❌ Cannot proceed with detailed comparison due to shape mismatch")
        return None
    
    # Detailed frame-by-frame comparison
    batch_size, num_frames = orig_coarse_scores.shape[:2]
    
    print(f"\nFRAME-BY-FRAME COMPARISON (showing first 10 frames):")
    print("Format: frame_idx | orig_bg orig_fg | mobile_bg mobile_fg | bg_diff fg_diff | detection_orig detection_mobile")
    print("-" * 100)
    
    significant_diffs = []
    detection_mismatches = []
    
    for b in range(batch_size):
        for f in range(min(num_frames, 10)):  # Show first 10 frames
            abs_frame = frame_offset + f
            
            # Coarse scores comparison
            orig_bg, orig_fg = orig_coarse_scores[b, f, 0], orig_coarse_scores[b, f, 1]
            mobile_bg, mobile_fg = mobile_coarse_scores[b, f, 0], mobile_coarse_scores[b, f, 1]
            
            bg_diff = abs(orig_bg - mobile_bg)
            fg_diff = abs(orig_fg - mobile_fg)
            
            # Detection decisions (fg > bg)
            orig_detection = 1 if orig_fg > orig_bg else 0
            mobile_detection = 1 if mobile_fg > mobile_bg else 0
            
            detection_match = orig_detection == mobile_detection
            significant_diff = bg_diff > tolerance or fg_diff > tolerance
            
            status = ""
            if not detection_match:
                status += "🔴 DETECTION_MISMATCH "
                detection_mismatches.append({
                    'frame': abs_frame,
                    'orig_detection': orig_detection,
                    'mobile_detection': mobile_detection,
                    'orig_scores': (orig_bg, orig_fg),
                    'mobile_scores': (mobile_bg, mobile_fg)
                })
            if significant_diff:
                status += "⚠️  LARGE_DIFF "
                significant_diffs.append({
                    'frame': abs_frame,
                    'bg_diff': bg_diff,
                    'fg_diff': fg_diff
                })
            
            print(f"{abs_frame:3d} | {orig_bg:7.4f} {orig_fg:7.4f} | {mobile_bg:7.4f} {mobile_fg:7.4f} | "
                  f"{bg_diff:6.4f} {fg_diff:6.4f} | {orig_detection:10d} {mobile_detection:12d} | {status}")
    
    # Fine scores comparison (summary)
    fine_diff = np.abs(orig_fine_scores - mobile_fine_scores)
    max_fine_diff = fine_diff.max()
    mean_fine_diff = fine_diff.mean()
    
    print(f"\nFINE SCORES COMPARISON:")
    print(f"Max difference: {max_fine_diff:.6f}")
    print(f"Mean difference: {mean_fine_diff:.6f}")
    print(f"95th percentile difference: {np.percentile(fine_diff, 95):.6f}")
    
    # Statistical summary
    coarse_diff = np.abs(orig_coarse_scores - mobile_coarse_scores)
    
    print(f"\nSTATISTICAL SUMMARY:")
    print(f"Coarse scores - Max diff: {coarse_diff.max():.6f}, Mean diff: {coarse_diff.mean():.6f}")
    print(f"Fine scores   - Max diff: {max_fine_diff:.6f}, Mean diff: {mean_fine_diff:.6f}")
    print(f"Frames with detection mismatch: {len(detection_mismatches)}/{num_frames}")
    print(f"Frames with significant score differences (>{tolerance}): {len(significant_diffs)}/{num_frames}")
    
    return {
        'coarse_diff_max': float(coarse_diff.max()),
        'coarse_diff_mean': float(coarse_diff.mean()),
        'fine_diff_max': float(max_fine_diff),
        'fine_diff_mean': float(mean_fine_diff),
        'detection_mismatches': detection_mismatches,
        'significant_diffs': significant_diffs,
        'total_frames': num_frames
    }

def analyze_nms_behavior(coarse_scores, window_size=5, prefix=""):
    """Analyze NMS behavior on coarse scores"""
    print(f"\n{prefix}NMS ANALYSIS (window={window_size}):")
    
    # Apply NMS
    nms_scores = non_maximum_suppression_np(coarse_scores[0], window_size)
    
    # Count detections before and after NMS
    pre_nms_detections = np.sum(coarse_scores[0, :, 1] > coarse_scores[0, :, 0])
    post_nms_detections = np.sum(nms_scores[:, 1] > nms_scores[:, 0])
    
    print(f"Detections before NMS: {pre_nms_detections}")
    print(f"Detections after NMS: {post_nms_detections}")
    print(f"NMS suppression rate: {(pre_nms_detections - post_nms_detections) / max(pre_nms_detections, 1) * 100:.1f}%")
    
    # Show specific frames with detections
    detection_frames = []
    for i in range(len(nms_scores)):
        if nms_scores[i, 1] > nms_scores[i, 0]:
            detection_frames.append(i)
    
    print(f"Detection frames after NMS: {detection_frames[:20]}")  # Show first 20
    
    return {
        'pre_nms_count': int(pre_nms_detections),
        'post_nms_count': int(post_nms_detections),
        'detection_frames': detection_frames
    }

def apply_tennis_rules_debug(fine_scores, coarse_predictions, classes):
    """Apply tennis rules with debug output"""
    print("\nTENNIS RULES APPLICATION:")
    
    num_frames, num_classes = fine_scores.shape
    fine_predictions = np.zeros_like(fine_scores, int)
    
    action_groups = [
        (0, 2, "Hit types"),
        (2, 5, "Hit subtypes"), 
        (5, 8, "Hit directions"),
        (16, 24, "Shot types"),
        (25, 29, "Error types")
    ]
    
    total_actions = 0
    
    for i in range(num_frames):
        if coarse_predictions[i] == 0:
            continue
            
        frame_actions = []
        
        # Apply rules for each action group
        for start, end, group_name in action_groups:
            if start < num_classes and end <= num_classes:
                group_scores = fine_scores[i, start:end]
                if len(group_scores) > 0:
                    max_idx = np.argmax(group_scores)
                    fine_predictions[i, start + max_idx] = 1
                    frame_actions.append(f"{group_name}[{start + max_idx}]")
        
        # Special rules
        if 24 < num_classes and fine_scores[i, 24] > 0.5:
            fine_predictions[i, 24] = 1
            frame_actions.append("Approach[24]")
        
        # Additional rules for non-serve shots
        if 5 < num_classes and fine_predictions[i, 5] != 1:
            additional_groups = [
                (8, min(10, num_classes), "Hand type"),
                (10, min(16, num_classes), "Shot variations")
            ]
            
            for start, end, group_name in additional_groups:
                if start < end:
                    group_scores = fine_scores[i, start:end]
                    if len(group_scores) > 0:
                        max_idx = np.argmax(group_scores)
                        fine_predictions[i, start + max_idx] = 1
                        frame_actions.append(f"{group_name}[{start + max_idx}]")
        
        if frame_actions:
            total_actions += len(frame_actions)
            if i < 10:  # Show first 10 action frames
                print(f"  Frame {i}: {', '.join(frame_actions)}")
    
    print(f"Total actions detected: {total_actions}")
    return fine_predictions

def diagnose_mobile_model(original_model, mobile_model, test_clip, classes, config):
    """Main diagnostic function"""
    print("\n" + "="*80)
    print("F3SET MOBILE MODEL DIAGNOSIS")
    print("="*80)
    
    video_name = test_clip['video']
    clip_start = test_clip['start']
    frame_tensor = test_clip['frame']
    hand_tensor = test_clip['hand']
    
    print(f"Analyzing clip from video: {video_name}")
    print(f"Clip start frame: {clip_start}")
    print(f"Clip length: {frame_tensor.shape[1]} frames")
    
    # Fix tensor shapes for both models
    print(f"Original frame shape: {frame_tensor.shape}")
    print(f"Original hand shape: {hand_tensor.shape}")
    
    # Ensure correct tensor shapes
    if frame_tensor.dim() == 4:  # (L, C, H, W) -> (1, L, C, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)
    if hand_tensor.dim() == 1:  # (2,) -> (1, 2)
        hand_tensor = hand_tensor.unsqueeze(0)
    
    print(f"Adjusted frame shape: {frame_tensor.shape}")
    print(f"Adjusted hand shape: {hand_tensor.shape}")
    
    # Analyze input preprocessing
    analyze_input_preprocessing(frame_tensor, hand_tensor)
    
    # Run inference on both models
    print(f"\nRunning inference on both models...")
    
    # Debug original model input requirements first
    frame_tensor_fixed, hand_tensor_fixed, orig_outputs = debug_original_model_inputs(
        original_model, frame_tensor, hand_tensor
    )
    
    if orig_outputs is None:
        print("❌ Could not find working input combination for original model")
        return None
    
    # Time the original model inference
    start_time = time.time()
    orig_outputs = original_model.predict(frame_tensor_fixed, hand_tensor_fixed)
    orig_time = (time.time() - start_time) * 1000
    print(f"✅ Original model inference successful: {orig_time:.2f}ms")
    
    # Mobile model
    start_time = time.time()
    try:
        mobile_outputs = mobile_model.predict(
            frame_tensor.squeeze(0) if frame_tensor.dim() == 5 else frame_tensor,
            hand_tensor.squeeze(0) if hand_tensor.dim() == 2 and hand_tensor.shape[0] == 1 else hand_tensor
        )
        mobile_time = (time.time() - start_time) * 1000
        print(f"✅ Mobile model inference successful")
    except Exception as e:
        print(f"❌ Mobile model inference failed: {e}")
        return None
    
    print(f"Original model inference time: {orig_time:.2f}ms")
    print(f"Mobile model inference time: {mobile_time:.2f}ms")
    
    # Validate output shapes before comparison
    print(f"\nValidating output shapes...")
    print(f"Original outputs: coarse_cls={orig_outputs[0].shape}, coarse_scores={orig_outputs[1].shape}, fine_scores={orig_outputs[2].shape}")
    print(f"Mobile outputs: coarse_cls={mobile_outputs[0].shape}, coarse_scores={mobile_outputs[1].shape}, fine_scores={mobile_outputs[2].shape}")
    
    # Compare outputs
    comparison_results = compare_model_outputs(orig_outputs, mobile_outputs, clip_start)
    
    if comparison_results is None:
        return None
    
    # Analyze NMS behavior for both models
    analyze_nms_behavior(orig_outputs[1], config.get('window', 5), "ORIGINAL - ")
    analyze_nms_behavior(mobile_outputs[1], config.get('window', 5), "MOBILE - ")
    
    # Apply tennis rules to both and compare
    print("\n" + "="*60)
    print("TENNIS RULES COMPARISON")
    print("="*60)
    
    # Get coarse predictions after NMS for both models
    orig_nms = non_maximum_suppression_np(orig_outputs[1][0], config.get('window', 5))
    mobile_nms = non_maximum_suppression_np(mobile_outputs[1][0], config.get('window', 5))
    
    orig_coarse_pred = np.argmax(orig_nms, axis=1)
    mobile_coarse_pred = np.argmax(mobile_nms, axis=1)
    
    print("ORIGINAL MODEL:")
    orig_fine_pred = apply_tennis_rules_debug(orig_outputs[2][0], orig_coarse_pred, classes)
    
    print("\nMOBILE MODEL:")
    mobile_fine_pred = apply_tennis_rules_debug(mobile_outputs[2][0], mobile_coarse_pred, classes)
    
    # Compare final predictions
    coarse_match = np.array_equal(orig_coarse_pred, mobile_coarse_pred)
    fine_match = np.array_equal(orig_fine_pred, mobile_fine_pred)
    
    print(f"\nFINAL PREDICTION COMPARISON:")
    print(f"Coarse predictions match: {coarse_match}")
    print(f"Fine predictions match: {fine_match}")
    
    if not coarse_match:
        coarse_diff_count = np.sum(orig_coarse_pred != mobile_coarse_pred)
        print(f"Coarse prediction differences: {coarse_diff_count}/{len(orig_coarse_pred)} frames")
    
    if not fine_match:
        fine_diff_count = np.sum(orig_fine_pred != mobile_fine_pred)
        print(f"Fine prediction differences: {fine_diff_count}/{orig_fine_pred.size} elements")
    
    # Generate diagnosis summary
    diagnosis = generate_diagnosis(comparison_results, orig_outputs, mobile_outputs)
    
    return {
        'comparison_results': comparison_results,
        'diagnosis': diagnosis,
        'inference_times': {'original': orig_time, 'mobile': mobile_time},
        'final_match': {'coarse': coarse_match, 'fine': fine_match}
    }

def generate_diagnosis(comparison_results, orig_outputs, mobile_outputs):
    """Generate diagnosis of the discrepancies"""
    diagnosis = []
    
    # Check for major issues
    if comparison_results['coarse_diff_max'] > 0.1:
        diagnosis.append("🔴 CRITICAL: Large coarse score differences detected (>0.1)")
        diagnosis.append("   Possible causes: Model weights conversion error, different softmax/sigmoid application")
    
    if comparison_results['fine_diff_max'] > 0.1:
        diagnosis.append("🔴 CRITICAL: Large fine score differences detected (>0.1)")
        diagnosis.append("   Possible causes: Model weights conversion error, different activation functions")
    
    if len(comparison_results['detection_mismatches']) > 0:
        ratio = len(comparison_results['detection_mismatches']) / comparison_results['total_frames']
        if ratio > 0.1:
            diagnosis.append(f"🔴 CRITICAL: Detection mismatches in {ratio*100:.1f}% of frames")
            diagnosis.append("   This explains false positives/negatives in mobile model")
        else:
            diagnosis.append(f"⚠️  WARNING: Minor detection mismatches in {ratio*100:.1f}% of frames")
    
    # Check for precision issues
    if 0.001 < comparison_results['coarse_diff_max'] < 0.01:
        diagnosis.append("⚠️  WARNING: Small but significant numerical differences")
        diagnosis.append("   Possible causes: Float32 vs Float16 precision, different optimization")
    
    if comparison_results['coarse_diff_max'] < 0.001:
        diagnosis.append("✅ GOOD: Coarse scores are numerically very close")
    
    if comparison_results['fine_diff_max'] < 0.001:
        diagnosis.append("✅ GOOD: Fine scores are numerically very close")
    
    # Check for systematic biases
    orig_mean_fg = np.mean(orig_outputs[1][0, :, 1])
    mobile_mean_fg = np.mean(mobile_outputs[1][0, :, 1])
    
    if abs(orig_mean_fg - mobile_mean_fg) > 0.05:
        diagnosis.append(f"⚠️  WARNING: Systematic bias in foreground scores")
        diagnosis.append(f"   Original mean: {orig_mean_fg:.4f}, Mobile mean: {mobile_mean_fg:.4f}")
    
    if not diagnosis:
        diagnosis.append("✅ No significant issues detected - models appear equivalent")
    
    return diagnosis

def save_diagnostic_report(results, output_file, video_name, clip_info):
    """Save detailed diagnostic report"""
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'video_name': video_name,
            'clip_start': clip_info['start'],
            'analysis_type': 'mobile_vs_original_diagnosis'
        },
        'results': results,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDiagnostic report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Diagnose F3Set mobile model vs original")
    parser.add_argument('model_dir', help='Path to original model directory')
    parser.add_argument('mobile_model', help='Path to mobile model (.pt or .ptl)')
    parser.add_argument('frame_dir', help='Path to frame directory')
    parser.add_argument('-s', '--split', choices=['train', 'val', 'test', 'test_one'],
                        default='test_one', help='Dataset split')
    parser.add_argument('-d', '--dataset', help='Dataset name')
    parser.add_argument('--video_name', help='Specific video name to test')
    parser.add_argument('--clip_index', type=int, default=0, 
                        help='Clip index to test (default: 0)')
    parser.add_argument('--epoch', type=int, help='Specific epoch to use')
    parser.add_argument('--output', help='Output file for diagnostic report')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Tolerance for detecting significant differences')
    
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
            # Get last epoch
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
            multi_gpu=False  # Disable for fair comparison
        )
        
        checkpoint_path = os.path.join(args.model_dir, f'checkpoint_{best_epoch:03d}.pt')
        original_model.load(torch.load(checkpoint_path, map_location='cpu'))
        
        # Load mobile model
        print("Loading mobile model...")
        mobile_model = F3SetMobileModelDiagnostic(args.mobile_model, device='cpu')
        
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
        
        # Run diagnosis
        results = diagnose_mobile_model(original_model, mobile_model, test_clip, classes, config)
        
        if results:
            # Print diagnosis
            print("\n" + "="*80)
            print("DIAGNOSIS SUMMARY")
            print("="*80)
            
            for item in results['diagnosis']:
                print(item)
            
            print(f"\nInference time comparison:")
            print(f"Original: {results['inference_times']['original']:.2f}ms")
            print(f"Mobile: {results['inference_times']['mobile']:.2f}ms")
            print(f"Speedup: {results['inference_times']['original']/results['inference_times']['mobile']:.2f}x")
            
            # Save report if requested
            if args.output:
                save_diagnostic_report(results, args.output, test_clip['video'], test_clip)
        
        print("\nDiagnosis complete!")
        
    except Exception as e:
        print(f"Error during diagnosis: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())