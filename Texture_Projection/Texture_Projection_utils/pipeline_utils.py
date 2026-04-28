import torch
import numpy as np
from PIL import Image

class ViewProcessor:
    def __init__(self, render, bake_exp=4.0):
        self.render = render
        self.bake_exp = bake_exp

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type="pl")
            normal_maps.append(normal_map)
        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(elev, azim, return_type="pl")
            position_maps.append(position_map)
        return position_maps
    
    def render_alpha_multiview(self, camera_elevs, camera_azims):
        alpha_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            alpha_map = self.render.render_alpha(elev, azim, return_type="th")
            alpha_np = self._convert_alpha_to_rgb(alpha_map)
            alpha_maps.append(Image.fromarray(alpha_np.astype(np.uint8)))
        return alpha_maps
    
    def _convert_alpha_to_rgb(self, alpha_tensor):
        alpha_np = alpha_tensor.cpu().numpy() * 255
        alpha_np = alpha_np.squeeze()
        if alpha_np.ndim == 2:
            alpha_np = np.stack([alpha_np] * 3, axis=-1)
        return alpha_np

    def bake_view_selection(self, candidate_camera_elevs, candidate_camera_azims, candidate_view_weights, max_selected_view_num):
        original_resolution = self.render.default_resolution
        self.render.set_default_render_resolution(512)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = [], [], []
        viewed_tri_idxs, viewed_masks = [], []

        face_areas = self.render.get_face_areas(from_one_index=True)
        total_area = face_areas.sum()
        face_area_ratios = face_areas / total_area

        for elev, azim in zip(candidate_camera_elevs, candidate_camera_azims):
            viewed_tri_idx = self.render.render_alpha(elev, azim, return_type="np")
            viewed_tri_idxs.append(set(np.unique(viewed_tri_idx.flatten())))
            viewed_masks.append(viewed_tri_idx[0, :, :, 0] > 0)

        is_selected = [False] * len(candidate_camera_elevs)
        total_viewed_tri_idxs = set()
        
        for idx in range(min(6, len(candidate_camera_elevs))):
            selected_camera_elevs.append(candidate_camera_elevs[idx]); selected_camera_azims.append(candidate_camera_azims[idx])
            selected_view_weights.append(candidate_view_weights[idx]); is_selected[idx] = True
            total_viewed_tri_idxs.update(viewed_tri_idxs[idx])

        for _ in range(max_selected_view_num - len(selected_view_weights)):
            max_inc, max_idx = 0, -1
            for idx in range(len(candidate_camera_elevs)):
                if is_selected[idx]: continue
                new_inc_area = face_area_ratios[list(viewed_tri_idxs[idx] - total_viewed_tri_idxs)].sum()
                if new_inc_area > max_inc: max_inc, max_idx = new_inc_area, idx
            if max_inc > 0.002:
                is_selected[max_idx] = True
                selected_camera_elevs.append(candidate_camera_elevs[max_idx]); selected_camera_azims.append(candidate_camera_azims[max_idx])
                selected_view_weights.append(candidate_view_weights[max_idx]); total_viewed_tri_idxs.update(viewed_tri_idxs[max_idx])
            else: break

        self.render.set_default_render_resolution(original_resolution)
        return selected_camera_elevs, selected_camera_azims, selected_view_weights

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights):
        textures, cos_maps = [], []
        for view, el, az, w in zip(views, camera_elevs, camera_azims, view_weights):
            tex, cos, _ = self.render.back_project(view, el, az)
            textures.append(tex); cos_maps.append(w * (cos**self.bake_exp))
        texture, mask = self.render.fast_bake_texture(textures, cos_maps)
        return texture, mask

    def texture_inpaint(self, texture, mask, mask_gray=None, default=None):
        if default is not None:
            mask_bool = mask.astype(bool)
            texture[~mask_bool] = torch.tensor(default, dtype=texture.dtype, device=texture.device)
        else:
            texture_np = self.render.uv_inpaint(texture, mask)
            texture = torch.from_numpy(texture_np / 255.0).float().to(texture.device)
        return texture
