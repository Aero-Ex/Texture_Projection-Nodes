import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from ..mesh.structure import Texture
from ..camera.generator import generate_orbit_views_c2ws, generate_intrinsics, generate_box_views_c2ws, generate_orbit_views_c2ws_from_elev_azim
from ..render.nvdiffrast.renderer_base import NVDiffRendererBase
from ..io.mesh_loader import load_whole_mesh
from ..utils.parse_color import parse_color

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    import nvdiffrast.torch as dr
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = torch.nn.functional.normalize(cube_to_dir(s, gx, gy), dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

class NativeEnvMap:
    def __init__(self, image: torch.Tensor):
        self.cubemap = latlong_to_cubemap(image, [512, 512])
        
    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, rotation_mtx=None, intensity=1.0, specular=True):
        import nvdiffrast.torch as dr
        wo = torch.nn.functional.normalize(view_pos - gb_pos, dim=-1)
        
        # ks in ORM format: R=Occlusion, G=Roughness, B=Metallic
        roughness = ks[..., 1:2]
        metallic = ks[..., 2:3]
        
        diffuse_color = kd * (1.0 - metallic)
        # 0.04 is a standard dielectric specular F0
        specular_color = kd * metallic + 0.04 * (1.0 - metallic)
        
        # Calculate reflection vector
        dot_nd_wo = torch.clamp(torch.sum(gb_normal * wo, dim=-1, keepdim=True), min=0.0)
        wi = torch.nn.functional.normalize(2.0 * dot_nd_wo * gb_normal - wo, dim=-1)
        
        # Sample environment using nvdiffrast directly
        # The cubemap has shape [6, 512, 512, 3]
        
        wi_query = wi.contiguous()
        normal_query = gb_normal.contiguous()
        
        if rotation_mtx is not None:
            # rotation_mtx: (B, 3, 3)
            # wi: (B, H, W, 3)
            B, H, W, _ = wi.shape
            # Rotate world-space query vectors into camera space
            # v_cam = R_world_to_cam * v_world = R_cam_to_world^T * v_world
            # For row vectors: v_cam^T = v_world^T * R_cam_to_world
            wi_query = torch.matmul(wi_query.view(B, -1, 3), rotation_mtx).view(B, H, W, 3)
            normal_query = torch.matmul(normal_query.view(B, -1, 3), rotation_mtx).view(B, H, W, 3)

        specular_light = dr.texture(self.cubemap.unsqueeze(0), wi_query.contiguous(), boundary_mode='cube')
        diffuse_light = dr.texture(self.cubemap.unsqueeze(0), normal_query.contiguous(), boundary_mode='cube')
        
        if specular:
            res = diffuse_color * diffuse_light + specular_color * specular_light
        else:
            res = kd * diffuse_light
            
        return res * intensity

def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return torch.clamp(mapped, 0.0, 1.0)

class VideoExporter:
    _env_cache = {}
    
    def __init__(self) -> None:
        self.mesh_renderer = NVDiffRendererBase(device='cuda')

    def export_condition(
        self,
        mesh_path:str,
        hdri_path:str="",
        render_rgb_hdri:str="false",
        lighting_mode:str="hdri",
        lighting_intensity:float=1.0,
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

            shaded = None
            if render_rgb_hdri == "true":
                map_kd_out = torch.clamp(albedo_res if albedo_res is not None else torch.tensor([0.8, 0.8, 0.8], device='cuda').expand_as(normal_bump), 0.0, 1.0)
                map_ks_out = torch.clamp(mr_res if mr_res is not None else torch.zeros_like(normal_bump), 0.0, 1.0)
                
                if lighting_mode == "uniform_ambient":
                    shaded = map_kd_out
                else:
                    # HDRI Loading & Caching
                    img = cv2.imread(hdri_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if img is None:
                        shaded = map_kd_out
                    else:
                        if hdri_path in VideoExporter._env_cache:
                            envmap = VideoExporter._env_cache[hdri_path]
                        else:
                            if img.shape[2] == 4:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)[..., :3]
                            else:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            env_tensor = torch.from_numpy(img_rgb).cuda()
                            envmap = NativeEnvMap(env_tensor)
                            VideoExporter._env_cache[hdri_path] = envmap
                        
                        gb_basecolor = map_kd_out[..., :3] ** 2.2
                        gb_orm = map_ks_out[..., :3]
                        view_pos = c2ws[i, :3, 3].unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(1, H, W, 3)
                        rotation_mtx = c2ws[i, :3, :3].unsqueeze(0)
                        
                        shaded = envmap.shade(
                            chunk_out['world_position'][..., :3],
                            chunk_out['world_normal_bump'][..., :3],
                            gb_basecolor,
                            gb_orm,
                            view_pos,
                            rotation_mtx=rotation_mtx,
                            intensity=lighting_intensity,
                            specular=True,
                        )
                        shaded = aces_tonemapping(shaded)
                        shaded = torch.clamp(shaded ** (1.0 / 2.2), 0.0, 1.0)
                        
                if background_vec is not None:
                    if shaded.shape[-1] == 4:
                        shaded = shaded[..., :3]
                    shaded = shaded * alpha + dark_grey_vec * (1.0 - alpha)

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
                'rgb': shaded,
            }
            results_list.append(results_chunk)
            
            # Tiny cleanup
            del chunk_out, alpha, ccm, normal, normal_bump, albedo_res, mr_res, shaded
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
