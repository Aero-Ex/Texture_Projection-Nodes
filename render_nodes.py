import os
import sys
import torch
import numpy as np
from PIL import Image
import trimesh

from .Texture_Projection.Texture_Projection_utils.texkit._vendor.video.export_nvdiffrast_video import VideoExporter
from .Texture_Projection.Renderer.DifferentiableRenderer.MeshRender import MeshRender
from .Texture_Projection.Texture_Projection_utils.pipeline_utils import ViewProcessor
from .Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import convert_obj_to_glb

def resolve_mesh_path(p):
    if p is None: return p
    if isinstance(p, list) and len(p) > 0: p = p[0]
    if not isinstance(p, str):
        if type(p).__name__ == "File3D":
            if hasattr(p, "get_source") and isinstance(p.get_source(), str): p = p.get_source()
            elif hasattr(p, "save_to"):
                import folder_paths
                tmp = os.path.join(folder_paths.get_temp_directory(), f"mesh_{os.urandom(4).hex()}.glb")
                return p.save_to(tmp)
            elif hasattr(p, "_source") and isinstance(p._source, str): p = p._source
        if hasattr(p, "export"):
            import folder_paths
            tmp = os.path.join(folder_paths.get_temp_directory(), f"mesh_{os.urandom(4).hex()}.glb")
            p.export(tmp, file_type="glb")
            return tmp
        if isinstance(p, dict): return resolve_mesh_path(p.get("mesh") or p.get("glb_path") or p.get("path") or p)
    if not isinstance(p, str): return p
    import folder_paths
    pts = [p] + [os.path.join(getattr(folder_paths, f"get_{d}_directory")(), p) for d in ("input", "output", "temp")]
    return next((os.path.abspath(x) for x in pts if os.path.exists(x)), p)

class Texture_ProjectionRenderConditions:
    """
    renders Normal, CCM, and Mask images from a 3D mesh using the local Grid renderer.
    """
    _exporter = None
    
    @classmethod
    def get_exporter(cls):
        if cls._exporter is None:
            cls._exporter = VideoExporter()
        return cls._exporter
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "tests/case_1/mesh.obj"}),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distances": ("STRING", {"default": "2.8, 2.8, 2.8, 2.8, 2.8, 2.8"}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
            },
            "optional": {
                "mesh": ("*",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("normal_batch", "normal_bump_batch", "ccm_batch", "mask_batch", "albedo_batch", "roughness_batch", "metallic_batch", "mesh_name")
    FUNCTION = "render"
    CATEGORY = "Texture_Projection/Render"

    def render(self, mesh_path, resolution, camera_type, camera_distances, geometry_scale, camera_elevations, camera_azimuths, mesh=None):
        mesh_path = resolve_mesh_path(mesh if mesh is not None else mesh_path)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found at: {mesh_path}")

        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

        import sys
        import subprocess
        from .Texture_Projection.Texture_Projection_utils.texkit._vendor.camera.generator import generate_orbit_views_c2ws_from_elev_azim
        
        try:
            cam_elevs = [float(x.strip()) for x in camera_elevations.split(",")]
            cam_azims = [float(x.strip()) for x in camera_azimuths.split(",")]
            cam_dists = [float(x.strip()) for x in camera_distances.split(",")]
            
            # Pad or truncate cam_dists to match cam_elevs length
            if len(cam_dists) == 1:
                cam_dists = cam_dists * len(cam_elevs)
            elif len(cam_dists) < len(cam_elevs):
                cam_dists = cam_dists + [cam_dists[-1]] * (len(cam_elevs) - len(cam_dists))
            elif len(cam_dists) > len(cam_elevs):
                cam_dists = cam_dists[:len(cam_elevs)]
                
        except Exception as e:
            print(f"Texture_Projection Error: Failed to parse camera parameters - {e}")
            sys.stdout.flush()
            raise e
            
        c2ws = generate_orbit_views_c2ws_from_elev_azim(radius=cam_dists, elevation=cam_elevs, azimuth=cam_azims)

        video_exporter = self.get_exporter()
        
        out = video_exporter.export_condition(
            mesh_path,
            geometry_scale=geometry_scale,
            H=resolution,
            W=resolution,
            perspective=(camera_type == "perspective"),
            fov_deg=49.13,
            c2ws=c2ws,
        )

        def out_to_tensor(tensor_grid):
            # tensor_grid is (B, H, W, C)
            if tensor_grid is not None:
                return tensor_grid.cpu()
            return torch.zeros((len(cam_elevs), resolution, resolution, 3))

        normal_batch = out_to_tensor(out['normal'])
        normal_bump_batch = out_to_tensor(out['normal_bump'])
        ccm_batch = out_to_tensor(out['ccm'])
        mask_batch = out_to_tensor(out['alpha'])
        albedo_batch = out_to_tensor(out.get('albedo'))
        mr_raw = out.get('mr')
        if mr_raw is not None:
             # out_to_tensor expects (B, H, W, C)
             # mr_raw[..., 1] is roughness, mr_raw[..., 2] is metallic
             roughness_batch = out_to_tensor(mr_raw[..., 1:2].repeat(1, 1, 1, 3))
             metallic_batch = out_to_tensor(mr_raw[..., 2:3].repeat(1, 1, 1, 3))
        else:
             roughness_batch = torch.zeros((len(cam_elevs), resolution, resolution, 3))
             metallic_batch = torch.zeros((len(cam_elevs), resolution, resolution, 3))

        return (normal_batch, normal_bump_batch, ccm_batch, mask_batch, albedo_batch, roughness_batch, metallic_batch, mesh_name)

class Texture_ProjectionBakeTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "output/textured_mesh.obj"}),
                "image_batch": ("IMAGE",),
                "bake_size": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distances": ("STRING", {"default": "2.8, 2.8, 2.8, 2.8, 2.8, 2.8"}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
                "output_dir": ("STRING", {"default": "baked"}),
                "blending_sharpness": (["sharp (cos^8)", "ultra_sharp (cos^16)", "smooth (cos^4)"], {"default": "sharp (cos^8)"}),
                "debug_overlay": (["disable", "enable"], {"default": "disable"}),
            },
            "optional": {
                "mesh": ("*",),
                "roughness_batch": ("IMAGE",),
                "metallic_batch": ("IMAGE",),
                "normal_batch": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("glb_path", "texture_map", "verification_batch", "roughness_map", "metallic_map", "normal_map")
    FUNCTION = "bake"
    CATEGORY = "Texture_Projection/Bake"

    def bake(self, mesh_path, image_batch, bake_size, camera_type, camera_distances, geometry_scale, camera_elevations, camera_azimuths, output_dir, blending_sharpness="sharp (cos^8)", debug_overlay="disable", mesh=None, roughness_batch=None, metallic_batch=None, normal_batch=None):
        # Unwrap original_mesh to avoid unnecessary serialization to disk and loss of UVs
        original_mesh = mesh
        if isinstance(original_mesh, list) and len(original_mesh) > 0: original_mesh = original_mesh[0]
        if isinstance(original_mesh, dict): original_mesh = original_mesh.get("mesh") or original_mesh.get("glb_path") or original_mesh.get("path") or original_mesh

        mesh_path_resolved = resolve_mesh_path(mesh if mesh is not None else mesh_path)
        
        import sys
        import folder_paths
        
        output_base = folder_paths.get_output_directory()
        
        # Clean prefix: handle cases where user passed "output/baked", "baked", or custom subfolders/stems
        prefix = output_dir.strip() if output_dir else "baked"
        if not os.path.isabs(prefix):
            # Strip redundant leading 'output/' if user supplied it
            if prefix.startswith("output/") or prefix.startswith("output\\"):
                prefix = prefix[7:]
            # Ensure filename stem is present if only folder was specified
            if not os.path.basename(prefix):
                prefix = os.path.join(prefix, "textured_mesh")
            elif not os.path.splitext(prefix)[1] and not prefix.endswith("_mesh") and not prefix.endswith("textured_mesh"):
                prefix = os.path.join(prefix, "textured_mesh")
        
        # Use official ComfyUI get_save_image_path for safe paths and incrementing non-overwriting counters
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, output_base)
        os.makedirs(full_output_folder, exist_ok=True)
        
        file_basename = f"{filename}_{counter:05}_"
        glb_path = os.path.join(full_output_folder, f"{file_basename}.glb")
        
        mesh_path_resolved = resolve_mesh_path(mesh_path_resolved)
        mesh_path_resolved = os.path.abspath(mesh_path_resolved)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if image_batch is None or image_batch.shape[0] == 0:
            print("Texture_Projection Error: Empty image batch.")
            sys.stdout.flush()
            return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)))

        # Parse camera parameters
        try:
            cam_elevs = [float(x.strip()) for x in camera_elevations.split(",")]
            cam_azims = [float(x.strip()) for x in camera_azimuths.split(",")]
            cam_dists = [float(x.strip()) for x in camera_distances.split(",")]
            
            # Pad or truncate cam_dists to match cam_elevs length
            if len(cam_dists) == 1:
                cam_dists = cam_dists * len(cam_elevs)
            elif len(cam_dists) < len(cam_elevs):
                cam_dists = cam_dists + [cam_dists[-1]] * (len(cam_elevs) - len(cam_dists))
            elif len(cam_dists) > len(cam_elevs):
                cam_dists = cam_dists[:len(cam_elevs)]
                
            # Standard Grid weights/exp
            cam_weights = [1.0] * len(cam_elevs)
            if len(cam_weights) >= 6:
                cam_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05] + [0.0] * (len(cam_elevs)-6)
        except Exception as e:
            print(f"Texture_Projection Error: Failed to parse camera parameters - {e}")
            sys.stdout.flush()
            return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)))

        # 1. Initialize Renderer and Processor
        renderer = MeshRender(
            default_resolution=bake_size,
            camera_distance=cam_dists[0],
            camera_type=camera_type,
            texture_size=bake_size,
            bake_mode="back_sample",
            shader_type="face",
            raster_mode="cr",
            device=device
        )
        if camera_type == "orth":
            renderer.set_orth_scale(2.0)
        view_processor = ViewProcessor(render=renderer)
        
        # 2. Load Mesh
        # Use the original passed-in mesh if possible, otherwise load from path
        if original_mesh is not None and hasattr(original_mesh, "vertices") and hasattr(original_mesh, "faces"):
            mesh = original_mesh
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
                if isinstance(mesh, list): mesh = mesh[0]
        else:
            if not os.path.exists(mesh_path_resolved):
                print(f"Texture_Projection Error: Mesh not found at {mesh_path_resolved}")
                sys.stdout.flush()
                return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, bake_size, bake_size, 3)))
                
            mesh = trimesh.load(mesh_path_resolved)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
                if isinstance(mesh, list): mesh = mesh[0]
            
        # Auto-detect and fix inside-out meshes (where normals point inward towards centroid)
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces") and len(mesh.faces) > 0:
            v_cent = mesh.vertices.mean(axis=0)
            out_vec = mesh.vertices - v_cent
            dot = np.sum(mesh.vertex_normals * out_vec, axis=1)
            if (dot < 0).mean() > 0.6:
                print(f"[Texture_Projection] Warning: Mesh was detected as INSIDE-OUT (inverted normals). Auto-flipping outward...")
                mesh.faces = mesh.faces[:, [0, 2, 1]]
                mesh.vertex_normals = None

        # Ensure UVs are loaded (trimesh often fails for GLB without materials)
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            from .Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import load_mesh as load_mesh_utils
            source_for_uvs = mesh_path_resolved if (original_mesh is None or not hasattr(original_mesh, "vertices")) else original_mesh
            _, _, vtx_uv, _, _ = load_mesh_utils(source_for_uvs)
            if vtx_uv is not None:
                # If it's a SimpleVisuals/ColorVisuals, convert to TextureVisuals
                mesh.visual = trimesh.visual.texture.TextureVisuals(uv=vtx_uv)
            
        # Explicitly apply NVDiffrast scale_to_bbox behavior to keep Renderer/Baker geometries structurally matched
        vertices = mesh.vertices
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        # Replicate scale_to_bbox(largest=True, scale=geometry_scale)
        scale = (bbox_max - bbox_min) / (2.0 * geometry_scale)
        scale_factor = scale.max()
        mesh.vertices = (vertices - center) / scale_factor
            
        renderer.load_mesh(mesh=mesh, auto_center=False)

        # 3. Process Images
        num_views = len(cam_elevs)

        def prepare_input_images(batch):
            if batch is None or batch.shape[0] == 0:
                return None
            imgs = []
            b_size = batch.shape[0]
            for i in range(num_views):
                idx = min(i, b_size - 1)
                img_tensor = batch[idx]
                img_np = (img_tensor[..., :3].cpu().numpy() * 255).astype(np.uint8)
                imgs.append(Image.fromarray(img_np).convert("RGB"))
            return imgs

        input_images = prepare_input_images(image_batch)
        input_roughness = prepare_input_images(roughness_batch)
        input_metallic = prepare_input_images(metallic_batch)
        input_normal = prepare_input_images(normal_batch)
        
        # 4. Bake and Stitch + Verification
        textures, cos_maps = [], []
        roughness_textures = [] if input_roughness is not None else None
        metallic_textures = [] if input_metallic is not None else None
        normal_textures = [] if input_normal is not None else None
        verif_images = []
        
        # Resolve blending exponent based on blending_sharpness
        if "ultra" in str(blending_sharpness).lower() or "16" in str(blending_sharpness):
            blend_power = 16.0
        elif "smooth" in str(blending_sharpness).lower() or "4" in str(blending_sharpness):
            blend_power = 4.0
        else:
            blend_power = 8.0

        for i, (img, elev, azim, dist, weight) in enumerate(zip(input_images, cam_elevs, cam_azims, cam_dists, cam_weights)):
            img_resized = img.resize((bake_size, bake_size))
            tex, cos, _ = renderer.back_project(img_resized, elev, azim, camera_distance=dist)
            textures.append(tex)
            cos_maps.append(weight * (cos ** blend_power))

            if input_roughness is not None:
                r_img_resized = input_roughness[i].resize((bake_size, bake_size))
                r_tex, _, _ = renderer.back_project(r_img_resized, elev, azim, camera_distance=dist)
                roughness_textures.append(r_tex)

            if input_metallic is not None:
                m_img_resized = input_metallic[i].resize((bake_size, bake_size))
                m_tex, _, _ = renderer.back_project(m_img_resized, elev, azim, camera_distance=dist)
                metallic_textures.append(m_tex)

            if input_normal is not None:
                n_img_resized = input_normal[i].resize((bake_size, bake_size))
                n_tex, _, _ = renderer.back_project(n_img_resized, elev, azim, camera_distance=dist)
                normal_textures.append(n_tex)
            
            if debug_overlay == "enable":
                # Use input image resolution for verification overlay
                v_h, v_w = image_batch.shape[1], image_batch.shape[2]
                norm_render = renderer.render_normal(elev, azim, camera_distance=dist, resolution=(v_h, v_w), return_type="th")
                alpha_mask = renderer.render_alpha(elev, azim, camera_distance=dist, resolution=(v_h, v_w), return_type="th")
                
                # Ensure they are [H, W, C]
                if norm_render.dim() == 4: norm_render = norm_render.squeeze(0)
                if alpha_mask.dim() == 4: alpha_mask = alpha_mask.squeeze(0)
                
                # Combine RGB from normal render and Alpha from alpha_mask to create RGBA
                overlay = torch.cat((norm_render, alpha_mask), dim=-1)
                verif_images.append(overlay.cpu())

        texture, trust_map = renderer.fast_bake_texture(textures, cos_maps)
        
        # 5. Inpaint
        texture = view_processor.texture_inpaint(texture, trust_map)
        renderer.set_texture(texture, force_set=True)

        roughness_texture = None
        if roughness_textures is not None:
            r_baked, _ = renderer.fast_bake_texture(roughness_textures, cos_maps)
            roughness_texture = view_processor.texture_inpaint(r_baked, trust_map)

        metallic_texture = None
        if metallic_textures is not None:
            m_baked, _ = renderer.fast_bake_texture(metallic_textures, cos_maps)
            metallic_texture = view_processor.texture_inpaint(m_baked, trust_map)

        normal_texture = None
        if normal_textures is not None:
            n_baked, _ = renderer.fast_bake_texture(normal_textures, cos_maps)
            normal_texture = view_processor.texture_inpaint(n_baked, trust_map)
            renderer.set_texture_normal(normal_texture, force_set=True)

        # Set metallic/roughness to renderer if present
        if roughness_texture is not None or metallic_texture is not None:
            # Channel 0: metallic, Channel 1: roughness
            m_chan = metallic_texture[..., 0:1] if metallic_texture is not None else torch.zeros((bake_size, bake_size, 1), device=device)
            r_chan = roughness_texture[..., 0:1] if roughness_texture is not None else torch.full((bake_size, bake_size, 1), 0.5, device=device)
            tex_mr = torch.cat([m_chan, r_chan, torch.ones_like(m_chan)], dim=-1)
            renderer.set_texture_mr(tex_mr, force_set=True)
        
        # 6. Save directly to GLB
        success = renderer.save_glb(glb_path, downsample=False)
        if not success or not os.path.exists(glb_path):
            print(f"Texture_Projection Error: GLB export failed.")
        sys.stdout.flush()
            
        # Format outputs
        out_tex = texture.cpu().unsqueeze(0) # [1, H, W, C]
        out_roughness = roughness_texture.cpu().unsqueeze(0) if roughness_texture is not None else torch.full((1, bake_size, bake_size, 3), 0.5)
        out_metallic = metallic_texture.cpu().unsqueeze(0) if metallic_texture is not None else torch.zeros((1, bake_size, bake_size, 3))
        out_normal = normal_texture.cpu().unsqueeze(0) if normal_texture is not None else torch.tensor([0.5, 0.5, 1.0]).view(1, 1, 1, 3).repeat(1, bake_size, bake_size, 1)

        if len(verif_images) > 0:
            verif_batch = torch.stack(verif_images) # [B, 512, 512, 3]
        else:
            verif_batch = torch.zeros((1, 512, 512, 3))
            
        # Return path relative to output directory for UI compatibility
        try:
            rel_glb_path = os.path.relpath(glb_path, output_base)
            if not rel_glb_path.startswith(".."):
                glb_path = rel_glb_path
        except:
            pass
            
        return (glb_path, out_tex, verif_batch, out_roughness, out_metallic, out_normal)

class Texture_ProjectionMeshDirectoryLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "input/meshes"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_directory"
    CATEGORY = "Texture_Projection/Utils"

    def load_directory(self, directory_path):
        import glob
        directory_path = resolve_mesh_path(directory_path)
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Texture_ProjectionMeshDirectoryLoader: Directory not found - {directory_path}")
            return ([],)
        
        files = []
        for ext in ("*.obj", "*.glb", "*.gltf", "*.fbx"):
            files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        files.sort()
        if len(files) == 0:
            print(f"Texture_ProjectionMeshDirectoryLoader: No meshes found in {directory_path}")
            
        return (files,)

class Texture_ProjectionBatchDatasetGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "input/meshes"}),
                "output_dir": ("STRING", {"default": "output/dataset"}),
                "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distances": ("STRING", {"default": "2.8, 2.8, 2.8, 2.8, 2.8, 2.8"}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "generate_dataset"
    CATEGORY = "Texture_Projection/Dataset"

    def generate_dataset(self, directory_path, output_dir, resolution, camera_type, camera_distances, geometry_scale, camera_elevations, camera_azimuths):
        import glob
        import gc
        import sys
        directory_path = resolve_mesh_path(directory_path)
        if not os.path.exists(directory_path):
            print(f"Texture_ProjectionBatchDatasetGenerator: Path not found - {directory_path}")
            return ("Path not found",)
            
        files = []
        if os.path.isdir(directory_path):
            for ext in ("*.obj", "*.glb", "*.gltf", "*.fbx"):
                files.extend(glob.glob(os.path.join(directory_path, ext)))
            files.sort()
        elif os.path.isfile(directory_path):
            files.append(directory_path)
        
        if len(files) == 0:
            print(f"Texture_ProjectionBatchDatasetGenerator: No meshes found in {directory_path}")
            return ("No meshes found",)

        render_node = Texture_ProjectionRenderConditions()
        saver_node = Texture_ProjectionDatasetSaver()
        
        # Pre-emptive VRAM cleanup before loop
        gc.collect()
        torch.cuda.empty_cache()
        
        for idx, mesh_path in enumerate(files):
            mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
            
            # Robust resume check: verify the final expected output file exists
            check_path = os.path.abspath(os.path.join(output_dir, mesh_name, f"{mesh_name}_metallic_grid.png"))
            if os.path.exists(check_path):
                print(f"Texture_ProjectionBatchDatasetGenerator: Skipping {mesh_name} (Metallic grid already exists at {check_path})")
                sys.stdout.flush()
                continue

            print(f"BatchDatasetGenerator: Processing mesh {idx+1}/{len(files)}: {mesh_path}")
            sys.stdout.flush()
            
            try:
                outs = render_node.render(
                    mesh_path=mesh_path,
                    resolution=resolution,
                    camera_type=camera_type,
                    camera_distances=camera_distances,
                    geometry_scale=geometry_scale,
                    camera_elevations=camera_elevations,
                    camera_azimuths=camera_azimuths,
                )
                
                # Unpack the new return tuple
                normals, bumps, ccms, masks, albedos, roughness, metallic, mesh_name = outs
                
                saver_node.save_dataset(
                    output_dir=output_dir,
                    prefix=mesh_name,
                    normal_batch=normals,
                    normal_bump_batch=bumps,
                    ccm_batch=ccms,
                    mask_batch=masks,
                    albedo_batch=albedos,
                    roughness_batch=roughness,
                    metallic_batch=metallic
                )
                
                # Protect VRAM aggressively
                del outs, normals, bumps, ccms, masks, albedos, roughness, metallic
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"BatchDatasetGenerator: Error processing {mesh_path} - {e}")
                sys.stdout.flush()
                
        return (f"Saved {len(files)} meshes",)

class Texture_ProjectionDatasetSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_dir": ("STRING", {"default": "output/dataset"}),
                "prefix": ("STRING", {"forceInput": True}),
                "normal_batch": ("IMAGE",),
                "normal_bump_batch": ("IMAGE",),
                "ccm_batch": ("IMAGE",),
                "mask_batch": ("IMAGE",),
                "albedo_batch": ("IMAGE",),
                "roughness_batch": ("IMAGE",),
                "metallic_batch": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_folder",)
    OUTPUT_NODE = True
    FUNCTION = "save_dataset"
    CATEGORY = "Texture_Projection/Dataset"

    def save_dataset(self, output_dir, prefix, normal_batch, normal_bump_batch, ccm_batch, mask_batch, albedo_batch, roughness_batch, metallic_batch):
        out_path = os.path.abspath(os.path.join(output_dir, prefix))
        os.makedirs(out_path, exist_ok=True)

        batches = {
            "normal": normal_batch,
            "normal_bump": normal_bump_batch,
            "ccm": ccm_batch,
            "mask": mask_batch,
            "albedo": albedo_batch,
            "roughness": roughness_batch,
            "metallic": metallic_batch,
        }

        # Validate that batches are not None and determine batch size
        valid_batch_size = 0
        for name, batch in batches.items():
            if batch is not None and batch.shape[0] > 0:
                valid_batch_size = max(valid_batch_size, batch.shape[0])

        if valid_batch_size == 0:
            print("DatasetSaver: Error, all input batches are empty.")
            return ("",)

        for suffix, batch in batches.items():
            if batch is None or batch.shape[0] == 0:
                continue
            
            B, H, W, C = batch.shape
            cols = 3
            rows = (B + cols - 1) // cols
            
            grid_h = rows * H
            grid_w = cols * W
            
            # Create an empty canvas
            grid_tensor = torch.zeros((grid_h, grid_w, C), dtype=batch.dtype, device=batch.device)
            
            for i in range(B):
                r = i // cols
                c = i % cols
                grid_tensor[r*H:(r+1)*H, c*W:(c+1)*W, :] = batch[i]
            
            if C == 4:
                img_np = (grid_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(img_np, mode="RGBA")
            elif C == 3:
                img_np = (grid_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(img_np, mode="RGB")
            elif C == 1:
                img_np = (grid_tensor.squeeze(-1).cpu().numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(img_np, mode="L")
            else:
                img_np = (grid_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(img_np)
            
            filename = f"{prefix}_{suffix}_grid.png"
            file_path = os.path.join(out_path, filename)
            img.save(file_path)

        print(f"Texture_ProjectionDatasetSaver: Successfully saved {valid_batch_size} views as grids for {prefix} to {out_path}")
        import sys
        sys.stdout.flush()
            
        return (out_path,)

class Texture_ProjectionHighToLowBake:
    """
    Bakes geometric details (tangent-space normal map, height, AO) and PBR materials 
    (diffuse, roughness, metallic) from a high-poly mesh onto an unwrapped low-poly mesh.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "high_poly_mesh_path": ("STRING", {"default": "tests/high_poly.glb"}),
                "low_poly_mesh_path": ("STRING", {"default": "tests/low_poly.glb"}),
                "bake_resolution": ([512, 1024, 2048, 4096], {"default": 2048}),
                "ray_max_distance": ("FLOAT", {"default": 0.08, "min": 0.001, "max": 1.0, "step": 0.001}),
                "cage_offset": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.001}),
                "bake_diffuse": (["enable", "disable"], {"default": "enable"}),
                "bake_normal": (["enable", "disable"], {"default": "enable"}),
                "bake_roughness": (["enable", "disable"], {"default": "enable"}),
                "bake_metallic": (["enable", "disable"], {"default": "enable"}),
                "bake_height": (["disable", "enable"], {"default": "disable"}),
                "bake_ao": (["disable", "enable"], {"default": "disable"}),
                "normal_format": (["OpenGL (Y+)", "DirectX (Y-)"], {"default": "OpenGL (Y+)"}),
                "auto_align": (["enable", "disable"], {"default": "enable"}),
                "output_dir": ("STRING", {"default": "baked_lowpoly"}),
            },
            "optional": {
                "high_poly_mesh": ("*",),
                "low_poly_mesh": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "MESH")
    RETURN_NAMES = ("glb_path", "diffuse_map", "normal_map", "roughness_map", "metallic_map", "height_map", "ao_map", "low_poly_mesh")
    FUNCTION = "bake"
    CATEGORY = "Texture_Projection"

    def bake(self, high_poly_mesh_path, low_poly_mesh_path, bake_resolution, ray_max_distance, cage_offset,
             bake_diffuse, bake_normal, bake_roughness, bake_metallic, bake_height, bake_ao, normal_format,
             output_dir="baked_lowpoly", auto_align="enable", high_poly_mesh=None, low_poly_mesh=None):
        
        import folder_paths
        from .high_to_low_baker import bake_high_to_low_poly
        from .Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import save_glb_mesh, load_mesh as load_mesh_utils

        # 1. Resolve High-Poly Mesh
        if isinstance(high_poly_mesh, list) and len(high_poly_mesh) > 0: high_poly_mesh = high_poly_mesh[0]
        if isinstance(high_poly_mesh, dict): high_poly_mesh = high_poly_mesh.get("mesh") or high_poly_mesh.get("glb_path") or high_poly_mesh.get("path") or high_poly_mesh
        high_path_resolved = resolve_mesh_path(high_poly_mesh if high_poly_mesh is not None else high_poly_mesh_path)

        if high_poly_mesh is not None and hasattr(high_poly_mesh, "vertices") and hasattr(high_poly_mesh, "faces"):
            high_mesh = high_poly_mesh
        else:
            if not os.path.exists(high_path_resolved):
                raise FileNotFoundError(f"High-poly mesh not found: {high_path_resolved}")
            high_mesh = trimesh.load(high_path_resolved)
            if isinstance(high_mesh, trimesh.Scene):
                high_mesh = high_mesh.dump(concatenate=True)
                if isinstance(high_mesh, list): high_mesh = high_mesh[0]

        # 2. Resolve Low-Poly Mesh
        if isinstance(low_poly_mesh, list) and len(low_poly_mesh) > 0: low_poly_mesh = low_poly_mesh[0]
        if isinstance(low_poly_mesh, dict): low_poly_mesh = low_poly_mesh.get("mesh") or low_poly_mesh.get("glb_path") or low_poly_mesh.get("path") or low_poly_mesh
        low_path_resolved = resolve_mesh_path(low_poly_mesh if low_poly_mesh is not None else low_poly_mesh_path)

        if low_poly_mesh is not None and hasattr(low_poly_mesh, "vertices") and hasattr(low_poly_mesh, "faces"):
            low_mesh = low_poly_mesh
        else:
            if not os.path.exists(low_path_resolved):
                raise FileNotFoundError(f"Low-poly mesh not found: {low_path_resolved}")
            low_mesh = trimesh.load(low_path_resolved)
            if isinstance(low_mesh, trimesh.Scene):
                low_mesh = low_mesh.dump(concatenate=True)
        # Auto-detect and fix inside-out meshes
        if hasattr(low_mesh, "vertices") and hasattr(low_mesh.faces, "__len__") and len(low_mesh.faces) > 0:
            v_cent = low_mesh.vertices.mean(axis=0)
            out_vec = low_mesh.vertices - v_cent
            dot = np.sum(low_mesh.vertex_normals * out_vec, axis=1)
            if (dot < 0).mean() > 0.6:
                print(f"[Texture_Projection] Warning: Low-poly mesh was detected as INSIDE-OUT. Auto-flipping outward...")
                low_mesh.faces = low_mesh.faces[:, [0, 2, 1]]
                low_mesh.vertex_normals = None

        # Extract and preserve UVs across all operations
        saved_uv = None
        if hasattr(low_mesh, 'visual') and hasattr(low_mesh.visual, 'uv') and low_mesh.visual.uv is not None:
            saved_uv = np.asarray(low_mesh.visual.uv, dtype=np.float32)
        elif hasattr(low_mesh, 'vertex_attributes') and ('texcoord' in low_mesh.vertex_attributes or 'uv' in low_mesh.vertex_attributes):
            saved_uv = np.asarray(low_mesh.vertex_attributes.get('texcoord') or low_mesh.vertex_attributes.get('uv'), dtype=np.float32)
        else:
            source_for_uvs = low_path_resolved if (low_poly_mesh is None or not hasattr(low_poly_mesh, "vertices")) else low_poly_mesh
            _, _, vtx_uv, _, _ = load_mesh_utils(source_for_uvs)
            if vtx_uv is not None:
                saved_uv = np.asarray(vtx_uv, dtype=np.float32)

        # Auto-align low-poly geometry to match high-poly bounding box & centroid
        if auto_align == "enable":
            h_min, h_max = high_mesh.bounds[0], high_mesh.bounds[1]
            l_min, l_max = low_mesh.bounds[0], low_mesh.bounds[1]
            h_center = (h_min + h_max) / 2.0
            l_center = (l_min + l_max) / 2.0
            h_ext = np.maximum(h_max - h_min, 1e-6)
            l_ext = np.maximum(l_max - l_min, 1e-6)
            # Use uniform isotropic scaling to avoid squishing facial features
            scale_factor = float(np.median(h_ext / l_ext))
            new_vertices = (low_mesh.vertices - l_center) * scale_factor + h_center
            low_mesh = trimesh.Trimesh(
                vertices=new_vertices,
                faces=low_mesh.faces,
                visual=trimesh.visual.texture.TextureVisuals(uv=saved_uv) if saved_uv is not None else None,
                process=False
            )
        elif saved_uv is not None:
            low_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=saved_uv)

        # Compute smooth vertex normals on low_mesh to ensure continuous organic shading
        low_mesh.fix_normals()
        low_mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=len(low_mesh.vertices),
            faces=low_mesh.faces,
            face_normals=low_mesh.face_normals
        )

        # 3. Setup output paths via native ComfyUI folder_paths
        output_base = folder_paths.get_output_directory()
        prefix = output_dir.strip() if output_dir else "baked_lowpoly"
        if not os.path.isabs(prefix):
            if prefix.startswith("output/") or prefix.startswith("output\\"):
                prefix = prefix[7:]
            if not os.path.basename(prefix):
                prefix = os.path.join(prefix, "lowpoly_baked")
            elif not os.path.splitext(prefix)[1] and not prefix.endswith("_baked") and not prefix.endswith("lowpoly_baked"):
                prefix = os.path.join(prefix, "lowpoly_baked")

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, output_base)
        os.makedirs(full_output_folder, exist_ok=True)
        file_basename = f"{filename}_{counter:05}_"
        glb_path = os.path.join(full_output_folder, f"{file_basename}.glb")

        import time
        t_start = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 4. Execute High-to-Low Poly Baking Engine
        baked_maps = bake_high_to_low_poly(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            resolution=bake_resolution,
            ray_max_dist=ray_max_distance,
            cage_offset=cage_offset,
            bake_diffuse=(bake_diffuse == "enable"),
            bake_normal=(bake_normal == "enable"),
            bake_roughness=(bake_roughness == "enable"),
            bake_metallic=(bake_metallic == "enable"),
            bake_height=(bake_height == "enable"),
            bake_ao=(bake_ao == "enable"),
            normal_format=normal_format,
            device=device
        )

        diffuse_np = baked_maps["diffuse"]
        normal_np = baked_maps["normal"]
        roughness_np = baked_maps["roughness"]
        metallic_np = baked_maps["metallic"]
        height_np = baked_maps["height"]
        ao_np = baked_maps["ao"]

        # 5. Export Low-Poly PBR GLB
        vtx_pos = np.asarray(low_mesh.vertices, dtype=np.float32)
        pos_idx = np.asarray(low_mesh.faces, dtype=np.int32)
        vtx_uv = np.asarray(low_mesh.visual.uv, dtype=np.float32)
        uv_idx = pos_idx

        success = save_glb_mesh(
            glb_path,
            vtx_pos,
            pos_idx,
            vtx_uv,
            uv_idx,
            diffuse_np,
            metallic=metallic_np,
            roughness=roughness_np,
            normal=normal_np if bake_normal == "enable" else None
        )

        if not success or not os.path.exists(glb_path):
            print(f"Texture_ProjectionHighToLowBake Error: Failed to save GLB to {glb_path}")
        else:
            print(f"[HighToLow Baker] Complete! Saved GLB to {glb_path} (Total time: {time.perf_counter() - t_start:.2f}s)")
        sys.stdout.flush()

        # Format image outputs for ComfyUI [1, H, W, C]
        def to_image_tensor(arr):
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return torch.from_numpy(arr).unsqueeze(0).float()

        out_diffuse_th = to_image_tensor(diffuse_np)
        out_normal_th = to_image_tensor(normal_np)
        out_roughness_th = to_image_tensor(roughness_np)
        out_metallic_th = to_image_tensor(metallic_np)
        out_height_th = to_image_tensor(height_np)
        out_ao_th = to_image_tensor(ao_np)

        # Relative path for UI
        try:
            rel_glb_path = os.path.relpath(glb_path, output_base)
            if not rel_glb_path.startswith(".."):
                glb_path = rel_glb_path
        except:
            pass

        return (glb_path, out_diffuse_th, out_normal_th, out_roughness_th, out_metallic_th, out_height_th, out_ao_th, low_mesh)

NODE_CLASS_MAPPINGS = {
    "Texture_ProjectionRenderConditions": Texture_ProjectionRenderConditions,
    "Texture_ProjectionBakeTextures": Texture_ProjectionBakeTextures,
    "Texture_ProjectionHighToLowBake": Texture_ProjectionHighToLowBake,
    "Texture_ProjectionDatasetSaver": Texture_ProjectionDatasetSaver,
    "Texture_ProjectionMeshDirectoryLoader": Texture_ProjectionMeshDirectoryLoader,
    "Texture_ProjectionBatchDatasetGenerator": Texture_ProjectionBatchDatasetGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texture_ProjectionRenderConditions": "Texture_Projection Render Conditions",
    "Texture_ProjectionBakeTextures": "Texture_Projection Bake Textures",
    "Texture_ProjectionHighToLowBake": "Texture_Projection High-to-Low Poly Baker",
    "Texture_ProjectionDatasetSaver": "Texture_Projection Dataset Saver",
    "Texture_ProjectionMeshDirectoryLoader": "Texture_Projection Mesh Directory Loader",
    "Texture_ProjectionBatchDatasetGenerator": "Texture_Projection Batch Dataset Generator",
}

