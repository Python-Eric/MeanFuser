#!/usr/bin/env python3
"""
Generate Gaussian Mixture Noise (GMN) parameters from navtrain expert trajectories.

Pipeline:
  1. Load all navtrain scenes, extract expert trajectories [N, 8, 3] (x, y, heading)
  2. Apply diff_traj() normalization → [N, 8, 4] (x_diff_norm, y_diff_norm, sin, cos)
  3. K-means clustering (K=8) in the normalized delta space
  4. Save cluster_trajs (absolute coords), center_points (delta space), center_std
  5. Compare with existing navtrain_8_mean_std.pkl

Usage:
  cd /home/wyh/project/End2end/MeanFuser
  export OPENSCENE_DATA_ROOT="/home/wyh/project/End2end/dataset"
  python tools/gaussian_mixed_noise/generate_gmn.py [--num_clusters 8] [--std 0.1] [--output OUTPUT_PATH]
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from pyquaternion import Quaternion

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from nuplan.common.actor_state.state_representation import StateSE2
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

# ─── Normalization constants (from navsim/agents/meanfuser/utils.py) ───
X_DIFF_MIN = -1.2698211669921875
X_DIFF_MAX = 7.475563049316406
X_DIFF_MEAN = 2.950225591659546

Y_DIFF_MIN = -5.012081146240234
Y_DIFF_MAX = 4.8563690185546875
Y_DIFF_MEAN = 0.0607292577624321

# ─── Navtrain log names (from scene_filter/navtrain.yaml) ───
NAVTRAIN_LOG_NAMES = [
    '2021.10.05.07.49.39_veh-52_00934_01406',
    '2021.07.09.02.42.50_veh-35_00038_02629',
    '2021.07.09.17.06.37_veh-35_02609_05015',
    '2021.10.11.08.31.07_veh-50_02360_02684',
    '2021.06.09.17.37.09_veh-12_04489_04816',
    '2021.07.09.16.12.19_veh-26_04434_04498',
    '2021.10.11.08.31.07_veh-50_00282_00680',
    '2021.06.14.16.48.02_veh-12_04783_04967',
    '2021.07.09.01.37.16_veh-26_01726_01793',
    '2021.10.01.17.52.06_veh-28_01034_01107',
    '2021.08.17.17.17.01_veh-45_02098_02251',
    '2021.10.06.17.08.46_veh-28_00498_00621',
    '2021.08.31.14.01.15_veh-40_00573_00681',
    '2021.09.15.12.32.43_veh-28_01070_01157',
    '2021.06.14.14.25.15_veh-26_04542_04617',
    '2021.07.16.01.22.41_veh-14_04315_07102',
    '2021.07.09.15.53.28_veh-38_03528_04262',
    '2021.08.24.17.01.06_veh-45_00228_00689',
    '2021.06.14.13.27.42_veh-35_02283_02603',
    '2021.08.24.14.35.46_veh-45_00011_00162',
    '2021.10.06.17.43.07_veh-28_00508_00877',
    '2021.06.14.16.32.09_veh-35_00283_00357',
]


def diff_traj_np(traj: np.ndarray) -> np.ndarray:
    """
    Numpy version of diff_traj from utils.py.
    Input:  traj [N, 8, 3] (x, y, heading)
    Output: delta [N, 8, 4] (x_diff_norm, y_diff_norm, sin, cos)
    """
    sin_h = np.sin(traj[..., 2:3])
    cos_h = np.cos(traj[..., 2:3])

    # x diff
    x = traj[..., 0:1]
    # 先增加起点，[N, 1, 1]，值为0， [N, 8, 1] --->[N, 9, 1]
    # 再计算diff，得到[N, 8, 1] 
    x_diff = np.diff(x, axis=1, prepend=np.zeros((traj.shape[0], 1, 1)))
    x_diff = x_diff - X_DIFF_MEAN
    x_diff_range = max(abs(X_DIFF_MAX - X_DIFF_MEAN), abs(X_DIFF_MIN - X_DIFF_MEAN))
    x_diff_norm = x_diff / x_diff_range

    # y diff
    y = traj[..., 1:2]
    y_diff = np.diff(y, axis=1, prepend=np.zeros((traj.shape[0], 1, 1)))
    y_diff = y_diff - Y_DIFF_MEAN
    y_diff_range = max(abs(Y_DIFF_MAX - Y_DIFF_MEAN), abs(Y_DIFF_MIN - Y_DIFF_MEAN))
    y_diff_norm = y_diff / y_diff_range

    return np.concatenate([x_diff_norm, y_diff_norm, sin_h, cos_h], axis=-1)


def load_navtrain_trajectories(data_root: str, max_scenes: int = None) -> np.ndarray:
    """Load all navtrain expert trajectories. Returns [N, 8, 3] array.
    
    Optimized: extracts ego_pose directly from scene_frames_dicts,
    bypassing expensive Scene construction (map API, sensors, annotations).
    """

    data_path = Path(data_root) / "navsim_logs" / "trainval"
    sensor_blobs_path = Path(data_root)

    # Read full log_names from yaml if available, otherwise use hardcoded list
    yaml_path = project_root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    log_names = None
    if yaml_path.exists():
        import yaml
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        log_names = cfg.get('log_names', None)
    if log_names is None:
        log_names = NAVTRAIN_LOG_NAMES

    print(f"Using {len(log_names)} navtrain logs")

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=max_scenes,
        log_names=log_names,
    )

    scene_loader = SceneLoader(
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    print(f"Total navtrain scenes: {len(scene_loader)}")

    num_history = scene_filter.num_history_frames  # 4
    num_traj_frames = 8  # future trajectory frames
    start_idx = num_history - 1  # current frame index (T=0)

    all_trajs = []
    for token in tqdm(scene_loader.tokens, desc="Extracting trajectories"):
        try:
            frame_list = scene_loader.scene_frames_dicts[token]

            # Extract global ego poses: current frame + 8 future frames = 9 poses
            global_poses = []
            for frame_idx in range(start_idx, start_idx + num_traj_frames + 1):
                frame = frame_list[frame_idx]
                translation = frame["ego2global_translation"]
                quat = Quaternion(*frame["ego2global_rotation"])
                yaw = quat.yaw_pitch_roll[0]
                global_poses.append([translation[0], translation[1], yaw])

            global_poses = np.array(global_poses, dtype=np.float64)

            # Convert to relative coordinates (same as Scene.get_future_trajectory)
            origin = StateSE2(*global_poses[0])
            local_poses = convert_absolute_to_relative_se2_array(origin, global_poses[1:])
            all_trajs.append(local_poses.astype(np.float32))  # (8, 3)
        except Exception as e:
            print(f"Error token {token}: {e}")
            continue

    trajs = np.array(all_trajs, dtype=np.float32)  # [N, 8, 3]
    print(f"Collected {trajs.shape[0]} trajectories, shape={trajs.shape}")
    return trajs


def generate_gmn(trajs: np.ndarray, num_clusters: int = 8, std_value: float = 0.1):
    """
    Run K-means on normalized delta trajectories and generate GMN parameters.

    Args:
        trajs: [N, 8, 3] absolute trajectories (x, y, heading)
        num_clusters: K for K-means
        std_value: fixed std for each Gaussian component

    Returns:
        dict with cluster_trajs, center_points, center_std
    """
    N = trajs.shape[0]
    print(f"\n=== Generating GMN with K={num_clusters}, std={std_value} ===")

    # Step 1: Normalize to delta space
    delta_trajs = diff_traj_np(trajs)  # [N, 8, 4]
    delta_flat = delta_trajs.reshape(N, -1)  # [N, 32]
    print(f"Delta trajectories shape: {delta_trajs.shape}")

    # Step 2: K-means clustering in delta space
    print(f"Running K-means (K={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300)
    # cluster = KMeans(n_clusters=num_clusters).fit(delta_flat).cluster_centers_
    
    labels = kmeans.fit_predict(delta_flat) # 返回每条轨迹属于哪一个簇

    # Step 3: Compute cluster parameters
    cluster_trajs = []
    center_points = []
    center_stds = []
    for k in range(num_clusters):
        mask = labels == k
        count = mask.sum()

        # Absolute trajectory center (mean of all trajectories in cluster)
        abs_center = trajs[mask].mean(axis=0)  # [8, 3]
        cluster_trajs.append(abs_center)

        # Center point: mean of all delta vectors in this cluster across SAMPLES and TIMESTEPS
        # NOTE: this differs from using the K-means center directly because sin/cos are nonlinear.
        # The x_diff/y_diff components are the same either way (linear operations),
        # but sin/cos must be averaged from the actual sample deltas, not from the cluster center trajectory.
        cluster_deltas = delta_trajs[mask]  # [Nk, 8, 4]
        center_point = cluster_deltas.mean(axis=(0, 1))  # [4]
        center_points.append(center_point)

        # Fixed std
        center_stds.append(np.full(4, std_value, dtype=np.float32))

        print(f"  Cluster {k}: {count} samples ({count/N*100:.1f}%), "
              f"mean_speed={abs_center[-1, 0]/4:.2f} m/s, "
              f"center_point={center_point}")

    result = {
        'cluster_trajs': torch.tensor(np.array(cluster_trajs), dtype=torch.float32),  # [K, 8, 3]
        'center_points': torch.tensor(np.array(center_points), dtype=torch.float32),  # [K, 4]
        'center_std': torch.tensor(np.array(center_stds), dtype=torch.float32),        # [K, 4]
    }

    return result, labels


def compare_with_existing(new_data: dict, existing_path: str):
    """Compare generated GMN with existing pkl."""

    print(f"\n=== Comparing with existing: {existing_path} ===")
    with open(existing_path, 'rb') as f:
        old_data = pickle.load(f)

    for key in ['cluster_trajs', 'center_points', 'center_std']:
        old_v = old_data[key]
        new_v = new_data[key]
        print(f"\n--- {key} ---")
        print(f"  Old shape: {old_v.shape}, New shape: {new_v.shape}")

        if old_v.shape != new_v.shape:
            print(f"  ⚠️  Shape mismatch!")
            continue

        diff = (old_v - new_v).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")

        print(f"  Old:\n{old_v}")
        print(f"  New:\n{new_v}")


def main():
    parser = argparse.ArgumentParser(description="Generate GMN parameters from navtrain")
    parser.add_argument('--data_root', type=str,
                        default=os.environ.get('OPENSCENE_DATA_ROOT', '/home/wyh/project/End2end/dataset'),
                        help='Dataset root directory')
    parser.add_argument('--num_clusters', type=int, default=8, help='Number of K-means clusters')
    parser.add_argument('--std', type=float, default=0.1, help='Fixed std for each Gaussian')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pkl path (default: tools/gaussian_mixed_noise/navtrain_{K}_mean_std_generated.pkl)')
    parser.add_argument('--max_scenes', type=int, default=None, help='Max scenes to load (for debugging)')
    parser.add_argument('--no_compare', action='store_true', help='Skip comparison with existing pkl')
    args = parser.parse_args()

    # Load trajectories
    trajs = load_navtrain_trajectories(args.data_root, args.max_scenes)

    # Print trajectory statistics
    print(f"\n=== Trajectory Statistics ===")
    print(f"  x range: [{trajs[:,:,0].min():.2f}, {trajs[:,:,0].max():.2f}]")
    print(f"  y range: [{trajs[:,:,1].min():.2f}, {trajs[:,:,1].max():.2f}]")
    print(f"  heading range: [{trajs[:,:,2].min():.4f}, {trajs[:,:,2].max():.4f}]")

    # Verify diff_traj normalization constants
    x_diffs = np.diff(trajs[:,:,0], axis=1, prepend=np.zeros((trajs.shape[0], 1)))
    y_diffs = np.diff(trajs[:,:,1], axis=1, prepend=np.zeros((trajs.shape[0], 1)))
    print(f"\n=== Diff Statistics (verify normalization constants) ===")
    print(f"  x_diff: min={x_diffs.min():.4f} (const={X_DIFF_MIN:.4f}), "
          f"max={x_diffs.max():.4f} (const={X_DIFF_MAX:.4f}), "
          f"mean={x_diffs.mean():.4f} (const={X_DIFF_MEAN:.4f})")
    print(f"  y_diff: min={y_diffs.min():.4f} (const={Y_DIFF_MIN:.4f}), "
          f"max={y_diffs.max():.4f} (const={Y_DIFF_MAX:.4f}), "
          f"mean={y_diffs.mean():.4f} (const={Y_DIFF_MEAN:.4f})")

    # Generate GMN
    gmn_data, labels = generate_gmn(trajs, args.num_clusters, args.std)

    # Save
    if args.output is None:
        args.output = str(project_root / f"tools/gaussian_mixed_noise/navtrain_{args.num_clusters}_mean_std_generated.pkl")
    with open(args.output, 'wb') as f:
        pickle.dump(gmn_data, f)
    print(f"\nSaved to: {args.output}")

    # Compare with existing
    if not args.no_compare:
        existing = str(project_root / "tools/gaussian_mixed_noise/navtrain_8_mean_std.pkl")
        if os.path.exists(existing):
            compare_with_existing(gmn_data, existing)
        else:
            print(f"No existing pkl found at {existing}, skipping comparison")

    # Print cluster distribution
    print(f"\n=== Cluster Distribution ===")
    for k in range(args.num_clusters):
        count = (labels == k).sum()
        print(f"  Cluster {k}: {count} ({count/len(labels)*100:.1f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
