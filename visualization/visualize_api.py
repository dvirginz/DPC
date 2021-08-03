import sys
import os
from visualization.mesh_container import MeshContainer
from visualization.mesh_visualizer import MeshVisualizer
from matplotlib import image
from numpy import asarray
import torch
import numpy as np


def log_prefix(model,idx=0):
    return "_".join([model.hparams.dataset_name, model.hparams.mode, "e%02d" % model.current_epoch, "b%02d" % model.hparams.batch_idx, "s%02d" % model.batch["source"]["id"][idx], "t%02d" % model.batch["target"]["id"][idx], model.hparams.arch])

def visualize_P(model, batch, mode, extra_text, scalar_maps, source_face, source_vert, target_face, target_vert, P, mesh_or_pc, 
    fwd_or_bac, horiz_space=0.1, grayed_indices=None, target_grayed=None, idx=None, 
    color_map=None, img_log_name='', img_description='', pic_folder=None, use_log_prefix=True, write_image=True, write_html=True, vis_mitsuba=False):
    pic_folder = os.path.join(model.hparams.default_root_dir,"visualization",mode)

    for i in range(len(P)):
        image_path = f"{pic_folder}/{extra_text}_{log_prefix(model,i)}.png"

        argmax_idx = 1 if fwd_or_bac == "fwd" else 0

        if scalar_maps is None:
            if isinstance(P[i], torch.Tensor):
                scalar_maps_i = P[i].argmax(dim=argmax_idx).detach().cpu().numpy() if len(P.shape) == 3 else P
            elif isinstance(P[i], np.ndarray):
                scalar_maps_i = P[i].argmax(axis=argmax_idx) if len(P.shape) == 3 else P
            else:
                assert False, "Type error"
        else:
            scalar_maps_i = scalar_maps[i].detach().cpu().numpy()

        viser = MeshVisualizer(dataset=model.hparams.dataset_name,display_up=hasattr(model.hparams,'display_id'))
        fig, source_colors, target_colors = viser.visualize_mesh_pair(
            source_mesh=MeshContainer(source_vert[i], source_face[i]),
            target_mesh=MeshContainer(target_vert[i], target_face[i]),
            corr_map=scalar_maps_i, ## in case P is already the corr_map
            color_map=color_map,
            save_path=image_path,
            mesh_or_pc=mesh_or_pc,
            fwd_or_bac=fwd_or_bac,
            horiz_space=horiz_space,
            grayed_indices=grayed_indices,
            target_grayed=target_grayed,
            rotate_pc_for_vis=model.hparams.rotate_pc_for_vis,
            rotate_pc_angles=model.hparams.rotate_pc_angles,
            write_image=write_image,
            write_html=write_html,
            vis_mitsuba=vis_mitsuba,
        )
        model.logger.experiment.add_image(tag=os.path.basename(image_path),img_tensor=asarray(image.imread(image_path)),global_step=model.global_step,dataformats='HWC')


    return image_path, source_colors, target_colors


def visualize_pcs_same_fig(model, mode, extra_text, pcs, pcs_color, idx=None, img_log_name='', img_description='', pic_folder=None, use_log_prefix=True, write_image=True, write_html=True):
    pic_folder = os.path.join(model.hparams.default_root_dir, "visualization", mode) if pic_folder is None else pic_folder

    for i in range(pcs[0].shape[0]):
        ii = i if idx is None else idx

        image_path = f"{pic_folder}/{extra_text}_{log_prefix(model,i)}.png"

        viser = MeshVisualizer(dataset=model.hparams.dataset_name, display_up=hasattr(model.hparams, 'display_id'))
        fig = viser.visualize_pcs_same_fig(save_path=image_path, pcs=[pc[i] for pc in pcs], pcs_color=pcs_color, write_image=write_image, write_html=write_html)
        model.logger.experiment.add_image(tag=os.path.basename(image_path),img_tensor=asarray(image.imread(image_path)),global_step=model.global_step,dataformats='HWC')

    return image_path


def visualize_reconstructions(model, batch, mode="train"):
    source = batch["source"]
    target = batch["target"]

    s_pos, s_cross_recon, s_cross_recon_hard = source["pos"], source["cross_recon"], source["cross_recon_hard"]

    if model.hparams.use_self_recon:
        s_self_recon = source["self_recon"]

    batch_size = len(s_pos)
    s_pcs_range_list = []
    for i in range(batch_size):
        pcs_list = [s_pos[i], s_cross_recon[i], s_cross_recon_hard[i]]
        if model.hparams.use_self_recon:
            pcs_list.append(s_self_recon[i])
        pcs_range_curr = MeshVisualizer.get_pcs_range(pcs_list)
        s_pcs_range_list.append(np.expand_dims(pcs_range_curr, axis=0))
    s_pcs_range = np.vstack(s_pcs_range_list)

    t_pos, t_cross_recon, t_cross_recon_hard = target["pos"], target["cross_recon"], target["cross_recon_hard"]

    if model.hparams.use_self_recon:
        t_self_recon = target["self_recon"]

    batch_size = len(t_pos)
    t_pcs_range_list = []
    for i in range(batch_size):
        pcs_list = [t_pos[i], t_cross_recon[i], t_cross_recon_hard[i]]
        if model.hparams.use_self_recon:
            pcs_list.append(t_self_recon[i])
        pcs_range_curr = MeshVisualizer.get_pcs_range(pcs_list)
        t_pcs_range_list.append(np.expand_dims(pcs_range_curr, axis=0))
    t_pcs_range = np.vstack(s_pcs_range_list)

    pos_color = (0, 0, 1)  # blue
    cross_recon_color = (1, 0, 0)  # red
    cross_recon_hard_color = (0, 0.45, 0)  # green
    self_recon_color = (1, 0, 1)  # magenta
    pcs_range_color = (1, 1, 1)  # white

    batch_for_vis = len(model.hparams.vis_idx_list) == 0 or model.hparams.batch_idx in model.hparams.vis_idx_list
    write_image = model.hparams.write_image and batch_for_vis
    write_html = model.hparams.write_html and batch_for_vis

    pcs_color = [pos_color, pcs_range_color]
    pcs = [s_pos, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "source", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [cross_recon_color, pcs_range_color]
    pcs = [s_cross_recon, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "source_cross_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [cross_recon_hard_color, pcs_range_color]
    pcs = [s_cross_recon_hard, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "source_cross_recon_hard", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    if model.hparams.use_self_recon:
        pcs_color = [self_recon_color, pcs_range_color]
        pcs = [s_self_recon, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
        visualize_pcs_same_fig(model, mode, "source_self_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [pos_color, cross_recon_color, pcs_range_color]
    pcs = [s_pos, s_cross_recon, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "source-source_cross_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [pos_color, cross_recon_hard_color, pcs_range_color]
    pcs = [s_pos, s_cross_recon_hard, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "source-source_cross_recon_hard", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    if model.hparams.use_self_recon:
        pcs_color = [pos_color, self_recon_color, pcs_range_color]
        pcs = [s_pos, s_self_recon, s_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
        visualize_pcs_same_fig(model, mode, "source-source_self_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [pos_color, pcs_range_color]
    pcs = [t_pos, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "target", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [cross_recon_color, pcs_range_color]
    pcs = [t_cross_recon, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "target_cross_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [cross_recon_hard_color, pcs_range_color]
    pcs = [t_cross_recon_hard, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "target_cross_recon_hard", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    if model.hparams.use_self_recon:
        pcs_color = [self_recon_color, pcs_range_color]
        pcs = [t_self_recon, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
        visualize_pcs_same_fig(model, mode, "target_self_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [pos_color, cross_recon_color, pcs_range_color]
    pcs = [t_pos, t_cross_recon, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "target-target_cross_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    pcs_color = [pos_color, cross_recon_hard_color, pcs_range_color]
    pcs = [t_pos, t_cross_recon_hard, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
    visualize_pcs_same_fig(model, mode, "target-target_cross_recon_hard", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)

    if model.hparams.use_self_recon:
        pcs_color = [pos_color, self_recon_color, pcs_range_color]
        pcs = [t_pos, t_self_recon, t_pcs_range]  # hack: add pcs_range as a pc to have the same range for different plots
        visualize_pcs_same_fig(model, mode, "target-target_self_recon", pcs, pcs_color, img_log_name='', write_image=write_image, write_html=write_html)


def visualize_pair_corr(model, batch, mode="train", extra_text="", scalar_maps=None):
    source = batch["source"]
    target = batch["target"]
    source_comf = batch["P_normalized"].max(2)[0]
    target_comf = batch["P_normalized"].max(1)[0]
    none_face_list = [None for i in range(source["pos"].shape[0])]

    batch_for_vis = len(model.hparams.vis_idx_list) == 0 or model.hparams.batch_idx in model.hparams.vis_idx_list or str(model.hparams.batch_idx) in model.hparams.vis_idx_list

    write_image = model.hparams.write_image and batch_for_vis
    write_html = model.hparams.write_html and batch_for_vis

    _, source_colors_fwd, target_colors_fwd = visualize_P(
        model, batch, mode, "s_t_fwd", scalar_maps, none_face_list, source["pos"], none_face_list, target["pos"], batch["P_normalized"], "pc", "fwd", color_map=None, img_log_name='', write_image=write_image, write_html=write_html)
    _, source_colors_bac, target_colors_bac = visualize_P(
        model, batch, mode, "s_t_bac", scalar_maps, none_face_list, source["pos"], none_face_list, target["pos"], batch["P_normalized"], "pc", "bac", color_map=None, img_log_name='', write_image=write_image, write_html=write_html)
