import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from ..mesh.structure import Texture
from ..camera.generator import generate_orbit_views_c2ws, generate_intrinsics, generate_box_views_c2ws, generate_orbit_views_c2ws_from_elev_azim
from ..render.nvdiffrast.renderer_base import NVDiffRendererBase
from ..io.mesh_loader import load_whole_mesh
from ..utils.parse_color import parse_color

class VideoExporter:
    def __init__(self) -> None:
        self.mesh_renderer = NVDiffRendererBase(device='cuda')

    def export_condition(
        self,
        mesh_path:str,
        geometry_scale=0.90,
        n_views=6, n_rows=2, n_cols=3, H=512, W=512,
        scale=1.0, fov_deg=49.1, perspective=False, orbit=False, c2ws=None,
        normal_map_strength=1.0,
        background:Optional[Union[str, float, List[float], Tuple[float]]]='grey',
        return_info=False,
        return_image=True,
        return_mesh=False,
        return_camera=False,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, Image.Image]]:
        mesh_trimesh = load_whole_mesh(mesh_path)
        texture_mesh = Texture.from_trimesh(mesh_trimesh)
        map_normal = texture_mesh.map_normal
        if map_normal is not None:
            map_normal = map_normal.to(device='cuda')
            
        map_kd = texture_mesh.map_Kd
        if map_kd is not None:
            map_kd = map_kd.to(device='cuda')
            
        map_ks = texture_mesh.map_Ks
        if map_ks is not None:
            map_ks = map_ks.to(device='cuda')
        mesh = texture_mesh.mesh
        mesh = mesh.scale_to_bbox(scale=geometry_scale).apply_transform()
        mesh = mesh.to(device='cuda')

        if c2ws is not None:
            pass
        elif orbit:
            c2ws = generate_orbit_views_c2ws(n_views + 1, radius=2.8, height=0.0, theta_0=0.0, degree=True)[:n_views]
        else:
            # Original Grid 6 views
            cam_elevs = [20, 20, 20, 20, -20, -20]
            cam_azims = [0, 90, 180, 270, 330, 30]
            c2ws = generate_orbit_views_c2ws_from_elev_azim(radius=2.8, elevation=cam_elevs, azimuth=cam_azims)
        
        if perspective:
            intrinsics = generate_intrinsics(fov_deg, fov_deg, fov=True, degree=True)
            self.mesh_renderer.enable_perspective()
        else:
            intrinsics = generate_intrinsics(scale, scale, fov=False, degree=False)
            self.mesh_renderer.enable_orthogonal()
            
        c2ws = c2ws.to(device='cuda')
        intrinsics = intrinsics.to(device='cuda')
        
        # Consistent background vector
        dark_grey_vec = torch.tensor([0.25, 0.25, 0.25], device='cuda')
        
        background_vec = parse_color(background)
        if background_vec is not None:
            background_vec = background_vec.to(dtype=torch.float32, device='cuda')

        results_list = []
        for i in range(c2ws.shape[0]):
            # Render a single view chunk
            chunk_out = self.mesh_renderer.simple_rendering(
                mesh, None, None, None,
                c2ws[i:i+1], intrinsics, (H, W), # assuming intrinsics is (1, 3, 3) or handles broadcasting
                render_world_normal=True,
                render_world_position=True,
                map_normal=map_normal,
                normal_map_strength=normal_map_strength,
                enable_antialis=False,
                render_map_kd=(map_kd is not None),
                map_kd=map_kd,
                render_map_ks=(map_ks is not None),
                map_ks=map_ks,
                background=background_vec,
            )
            
            # Post-process the chunk (shading, alpha, etc.)
            alpha = chunk_out['alpha']
            ccm = chunk_out['world_position'].mul(0.5).add(0.5)
            ccm = ccm * alpha + dark_grey_vec * (1.0 - alpha)
            
            normal = chunk_out['world_normal'].mul(0.5).add(0.5)
            normal = normal * alpha + dark_grey_vec * (1.0 - alpha)

            normal_bump = chunk_out['world_normal_bump'].mul(0.5).add(0.5)
            normal_bump = normal_bump * alpha + dark_grey_vec * (1.0 - alpha)

            albedo_res = chunk_out.get('map_kd', None)
            mr_res = chunk_out.get('map_ks', None)

            # Albedo and MR background mixing
            if albedo_res is not None:
                if albedo_res.shape[-1] == 4:
                    albedo_res = albedo_res[..., :3]
                albedo_res = albedo_res * alpha + dark_grey_vec * (1.0 - alpha)
            
            if mr_res is not None:
                # Use dark grey for data maps to provide neutral contrast (neither 0 nor 1)
                mr_res = mr_res * alpha + dark_grey_vec * (1.0 - alpha)

            mr_out = mr_res
            if mr_out is not None:
                mr_out = mr_out.clone()
                mr_out[..., 0] = 0.0
                
            results_chunk = {
                'alpha': alpha,
                'ccm': ccm,
                'normal': normal,
                'normal_bump': normal_bump,
                'albedo': albedo_res, 
                'mr': mr_out,
            }
            results_list.append(results_chunk)
            
            # Tiny cleanup
            del chunk_out, alpha, ccm, normal, normal_bump, albedo_res, mr_res
            if i % 2 == 0: torch.cuda.empty_cache() # Occasional flush

        # Concatenate all results
        final_out = {}
        keys = results_list[0].keys()
        for k in keys:
            tensors = [r[k] for r in results_list if r[k] is not None]
            if tensors:
                final_out[k] = torch.cat(tensors, dim=0)
            else:
                final_out[k] = None
                
        return final_out
