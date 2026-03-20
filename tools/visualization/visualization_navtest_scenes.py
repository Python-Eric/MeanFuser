import os
NAVSIM_WORKSPACE = os.environ.get('NAVSIM_WORKSPACE', None)
os.environ['NAVSIM_CACHE_ROOT'] = f"{NAVSIM_WORKSPACE}/cache"
os.environ['OPENSCENE_DATA_ROOT'] = f"{NAVSIM_WORKSPACE}/dataset"
os.environ['NUPLAN_MAPS_ROOT'] = f"{NAVSIM_WORKSPACE}/dataset/maps"

from typing import Tuple
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from pathlib import Path
import concurrent.futures
from multiprocessing import cpu_count
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.visualization.bev import add_configured_bev_on_ax
from navsim.visualization.config import BEV_PLOT_CONFIG


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    # ax.set_ylim(-margin_x / 2, margin_x / 2)
    ax.set_ylim(-8, 40)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def add_icon(ax, xy=(0, 1.461), zoom=0.95):
    img = plt.imread('assets/car_icon.png')
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=False, pad=0, zorder=100)
    ax.add_artist(ab)
    return ab

def plot_bev_with_agent(scene, agent, agent_trajectory) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    # agent_inputs = scene.get_agent_input()

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx], add_ego=False)
    add_icon(ax)

    legend_handles = []

    if agent_trajectory is not None:
        agent_legend = Line2D([0], [0], 
                                 color='#0AD0BF',  # 使用cmap中间颜色
                                 linewidth=2, 
                                 alpha=0.9, 
                                 label='Agent')
        legend_handles.append(agent_legend)

        agent_trajectory = np.concatenate([np.array([[0, 0]]), agent_trajectory[:, :2]])
        xy = agent_trajectory[:, [1, 0]] 
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        color_param = np.linspace(0.0, 1.0, len(segments))
        cmap = LinearSegmentedColormap.from_list('color_agent', ['#0AD0BF', '#0AD0BF', '#0AD0BF'])
        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=2,
            alpha=1,
            zorder=5,
        )
        lc.set_array(color_param)
        ax.add_collection(lc)
    
    poses = np.concatenate([np.array([[0, 0]]), human_trajectory.poses[:8, :2]])
    human_line = ax.plot(
        poses[:, 1],
        poses[:, 0],
        color='#FF6249',
        alpha=1,
        linewidth=2,
        zorder=4,
        label='Human',
    )[0]
    legend_handles.append(human_line)

    configure_bev_ax(ax)
    configure_ax(ax)
    ax.legend(handles=legend_handles)
    return fig, ax


def process_token(token, scene_loader, agent, dataset, bev_save_path):
    
    scene = scene_loader.get_scene_from_token(token)
    scene_token = scene.scene_metadata.scene_token
    timestamp = np.round(scene.frames[3].timestamp/1e6, 1)

    save_path = os.path.join(bev_save_path, f"{scene_token}_{timestamp}_{token}.png")
    agent_inputs, targets  = dataset._load_scene_with_token(token)
    agent_output = agent.compute_trajectory(agent_inputs, targets)
    agent_trajectory = agent_output['trajectory'][0].numpy()
    
    fig, ax = plot_bev_with_agent(scene, agent, agent_trajectory)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)


if __name__ == "__main__":

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../navsim/planning/script/config/pdm_scoring")
    
    # Please change the ckpt_path to your own path
    ckpt_path = "exp/meanfuser_checkpoints/meanfuser_pdms_89.0.ckpt"
    # Please change the save_path to your own path
    save_path = "exp/meanfuser_checkpoints/visualization/"
    
    NAVSIM_CACHE_ROOT = os.environ.get("NAVSIM_CACHE_ROOT")
    overrides = [
        "train_test_split=navtest",
        "agent=meanfuser_agent",
        "agent.config.num_proposals=8",
        f"agent.checkpoint_path={ckpt_path}",
        f"cache_path={NAVSIM_CACHE_ROOT}/trainval_v1_cache"
    ]
    cfg = hydra.compose(config_name="default_run_pdm_score", overrides=overrides)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=agent.get_sensor_config(),
        )

    test_dataset = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=scene_filter.log_names,
                is_training=False,
            )

    tokens = scene_loader.tokens
    print(f"Num Tokens: {len(tokens)}")
    
    bev_save_path = os.path.join(save_path, "bev_scenes")
    os.makedirs(bev_save_path, exist_ok=True)

    # Just for testing
    for token in tqdm(tokens[:1]):
        process_token(token, scene_loader, agent, test_dataset, bev_save_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        tasks = [
            executor.submit(
                process_token, 
                token,
                scene_loader, 
                agent, 
                test_dataset, 
                bev_save_path, 
                ) 
            for token in tokens
        ]
        
        for future in tqdm(
            concurrent.futures.as_completed(tasks), 
            total=len(tasks),
            desc="Processing tokens"
        ):
            try:
                token = future.result()
            except Exception as e:
                print(f"Error processing token: {e}")
