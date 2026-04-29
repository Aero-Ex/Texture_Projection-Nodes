import os
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
                "camera_distance": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 10.0, "step": 0.001}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
                "hdri_path": ("STRING", {"default": "/home/aero/Desktop/stuttgart_hillside_4k.exr"}),
                "render_rgb_hdri": (["true", "false"], {"default": "false"}),
                "lighting_mode": (["hdri", "uniform_ambient"], {"default": "hdri"}),
                "lighting_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mesh": ("*",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("normal_batch", "normal_bump_batch", "ccm_batch", "mask_batch", "rgb_batch", "albedo_batch", "roughness_batch", "metallic_batch", "mesh_name")
    FUNCTION = "render"
    CATEGORY = "Texture_Projection/Render"

    def render(self, mesh_path, resolution, camera_type, camera_distance, geometry_scale, camera_elevations, camera_azimuths, hdri_path, render_rgb_hdri, lighting_mode, lighting_intensity, mesh=None):
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
        except Exception as e:
            print(f"Texture_Projection Error: Failed to parse camera parameters - {e}")
            sys.stdout.flush()
            raise e
            
        c2ws = generate_orbit_views_c2ws_from_elev_azim(radius=camera_distance, elevation=cam_elevs, azimuth=cam_azims)

        video_exporter = self.get_exporter()
        
        out = video_exporter.export_condition(
            mesh_path,
            hdri_path=hdri_path,
            render_rgb_hdri=render_rgb_hdri,
            lighting_mode=lighting_mode,
            lighting_intensity=lighting_intensity,
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

        # HDRI PBR Render Pass via PyTorch Deferred Shading
        if render_rgb_hdri == "true":
            rgb_batch = out_to_tensor(out.get('rgb'))
        else:
            rgb_batch = torch.zeros((len(cam_elevs), resolution, resolution, 3))

        return (normal_batch, normal_bump_batch, ccm_batch, mask_batch, rgb_batch, albedo_batch, roughness_batch, metallic_batch, mesh_name)

class Texture_ProjectionBakeTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_path": ("STRING", {"default": "output/textured_mesh.obj"}),
                "image_batch": ("IMAGE",),
                "bake_size": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distance": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 10.0, "step": 0.001}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
                "output_dir": ("STRING", {"default": "output/baked"}),
                "debug_overlay": (["disable", "enable"], {"default": "disable"}),
            },
            "optional": {
                "mesh": ("*",),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("glb_path", "texture_map", "verification_batch")
    FUNCTION = "bake"
    CATEGORY = "Texture_Projection/Bake"

    def bake(self, mesh_path, image_batch, bake_size, camera_type, camera_distance, geometry_scale, camera_elevations, camera_azimuths, output_dir, debug_overlay, mesh=None):
        mesh_path = resolve_mesh_path(mesh if mesh is not None else mesh_path)
        
        import sys
        import folder_paths
        # Use ComfyUI's official output directory as the base
        output_base = folder_paths.get_output_directory()
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(output_base, output_dir)
        
        mesh_path = resolve_mesh_path(mesh_path)
        mesh_path = os.path.abspath(mesh_path)
        
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if image_batch is None or image_batch.shape[0] == 0:
            print("Texture_Projection Error: Empty image batch.")
            sys.stdout.flush()
            return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)))

        # Parse camera parameters
        try:
            cam_elevs = [float(x.strip()) for x in camera_elevations.split(",")]
            cam_azims = [float(x.strip()) for x in camera_azimuths.split(",")]
            # Standard Grid weights/exp
            cam_weights = [1.0] * len(cam_elevs)
            if len(cam_weights) >= 6:
                cam_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05] + [0.0] * (len(cam_elevs)-6)
        except Exception as e:
            print(f"Texture_Projection Error: Failed to parse camera parameters - {e}")
            sys.stdout.flush()
            return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)))

        # 1. Initialize Renderer and Processor
        renderer = MeshRender(
            default_resolution=bake_size,
            camera_distance=camera_distance,
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
        if not os.path.exists(mesh_path):
            print(f"Texture_Projection Error: Mesh not found at {mesh_path}")
            sys.stdout.flush()
            return ("", torch.zeros((1, bake_size, bake_size, 3)), torch.zeros((1, 512, 512, 3)))
            
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            if isinstance(mesh, list): mesh = mesh[0]
            
        # Ensure UVs are loaded (trimesh often fails for GLB without materials)
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            from .Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import load_mesh as load_mesh_utils
            _, _, vtx_uv, _, _ = load_mesh_utils(mesh_path)
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
        input_images = []
        batch_size = image_batch.shape[0]
        num_views = len(cam_elevs)
        
        for i in range(num_views):
            idx = min(i, batch_size - 1)
            img_tensor = image_batch[idx] # [H, W, C]
            
            # Simple RGB conversion (removing the automatic masking as requested)
            img_np = (img_tensor[..., :3].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np).convert("RGB")
            input_images.append(img)
        
        # 4. Bake and Stitch + Verification
        textures, cos_maps = [], []
        verif_images = []
        
        for i, (img, elev, azim, weight) in enumerate(zip(input_images, cam_elevs, cam_azims, cam_weights)):
            img_resized = img.resize((bake_size, bake_size))
            tex, cos, _ = renderer.back_project(img_resized, elev, azim)
            textures.append(tex)
            cos_maps.append(weight * (cos ** 4.0))
            
            if debug_overlay == "enable":
                # Use input image resolution for verification overlay
                v_h, v_w = image_batch.shape[1], image_batch.shape[2]
                norm_render = renderer.render_normal(elev, azim, resolution=(v_h, v_w), return_type="th")
                alpha_mask = renderer.render_alpha(elev, azim, resolution=(v_h, v_w), return_type="th")
                
                # Ensure they are [H, W, C]
                if norm_render.dim() == 4: norm_render = norm_render.squeeze(0)
                if alpha_mask.dim() == 4: alpha_mask = alpha_mask.squeeze(0)
                
                bg_img = img_tensor[idx, ..., :3].to(device)
                overlay = bg_img * (1.0 - alpha_mask * 0.5) + norm_render * (alpha_mask * 0.5)
                verif_images.append(overlay.cpu())

        texture, trust_map = renderer.fast_bake_texture(textures, cos_maps)
        
        # 5. Inpaint
        mask_np = (trust_map.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = view_processor.texture_inpaint(texture, mask_np)
        
        # 6. Save and Convert
        renderer.set_texture(texture, force_set=True)
        obj_path = os.path.join(output_dir, "textured_mesh.obj")
        print(f"Texture_Projection: Exporting OBJ to {obj_path}")
        renderer.save_mesh(obj_path, downsample=False)
        
        glb_path = obj_path.replace(".obj", ".glb")
        print(f"Texture_Projection: Converting to GLB via trimesh...")
        success = convert_obj_to_glb(obj_path, glb_path)
        
        if success and os.path.exists(glb_path):
            print(f"Texture_Projection: GLB SAVED SUCCESSFULLY: {glb_path}")
        else:
            print(f"Texture_Projection Error: GLB conversion failed.")
        
        sys.stdout.flush()
            
        # Format outputs
        out_tex = texture.cpu().unsqueeze(0) # [1, H, W, C]
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
            
        return (glb_path, out_tex, verif_batch)

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
                "camera_distance": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 10.0, "step": 0.001}),
                "geometry_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.001}),
                "camera_elevations": ("STRING", {"default": "20, 20, 20, 20, -20, -20"}),
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 330, 30"}),
                "hdri_path": ("STRING", {"default": "/home/aero/Desktop/stuttgart_hillside_4k.exr"}),
                "render_rgb_hdri": (["true", "false"], {"default": "false"}),
                "lighting_mode": (["hdri", "uniform_ambient"], {"default": "hdri"}),
                "lighting_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "generate_dataset"
    CATEGORY = "Texture_Projection/Dataset"

    def generate_dataset(self, directory_path, output_dir, resolution, camera_type, camera_distance, geometry_scale, camera_elevations, camera_azimuths, hdri_path, render_rgb_hdri, lighting_mode, lighting_intensity):
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
                    camera_distance=camera_distance,
                    geometry_scale=geometry_scale,
                    camera_elevations=camera_elevations,
                    camera_azimuths=camera_azimuths,
                    hdri_path=hdri_path, 
                    render_rgb_hdri=render_rgb_hdri,
                    lighting_mode=lighting_mode,
                    lighting_intensity=lighting_intensity
                )
                
                # Unpack the new return tuple
                normals, bumps, ccms, masks, rgbs, albedos, roughness, metallic, mesh_name = outs
                
                saver_node.save_dataset(
                    output_dir=output_dir,
                    prefix=mesh_name,
                    normal_batch=normals,
                    normal_bump_batch=bumps,
                    ccm_batch=ccms,
                    mask_batch=masks,
                    rgb_batch=rgbs,
                    albedo_batch=albedos,
                    roughness_batch=roughness,
                    metallic_batch=metallic
                )
                
                # Protect VRAM aggressively
                del outs, normals, bumps, ccms, masks, rgbs, albedos, roughness, metallic
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
                "rgb_batch": ("IMAGE",),
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

    def save_dataset(self, output_dir, prefix, normal_batch, normal_bump_batch, ccm_batch, mask_batch, rgb_batch, albedo_batch, roughness_batch, metallic_batch):
        out_path = os.path.abspath(os.path.join(output_dir, prefix))
        os.makedirs(out_path, exist_ok=True)

        batches = {
            "normal": normal_batch,
            "normal_bump": normal_bump_batch,
            "ccm": ccm_batch,
            "mask": mask_batch,
            "rgb": rgb_batch,
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

NODE_CLASS_MAPPINGS = {
    "Texture_ProjectionRenderConditions": Texture_ProjectionRenderConditions,
    "Texture_ProjectionBakeTextures": Texture_ProjectionBakeTextures,
    "Texture_ProjectionDatasetSaver": Texture_ProjectionDatasetSaver,
    "Texture_ProjectionMeshDirectoryLoader": Texture_ProjectionMeshDirectoryLoader,
    "Texture_ProjectionBatchDatasetGenerator": Texture_ProjectionBatchDatasetGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Texture_ProjectionRenderConditions": "Texture_Projection Render Conditions",
    "Texture_ProjectionBakeTextures": "Texture_Projection Bake Textures",
    "Texture_ProjectionDatasetSaver": "Texture_Projection Dataset Saver",
    "Texture_ProjectionMeshDirectoryLoader": "Texture_Projection Mesh Directory Loader",
    "Texture_ProjectionBatchDatasetGenerator": "Texture_Projection Batch Dataset Generator",
}
