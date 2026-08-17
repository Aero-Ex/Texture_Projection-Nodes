import os
from PIL import Image
import math
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict, Any

try:
    import bpy
except ImportError:
    bpy = None

def _safe_extract_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default

def _convert_to_numpy(data: Any, dtype: np.dtype) -> Optional[np.ndarray]:
    if data is None: return None
    return np.asarray(data, dtype=dtype)

def load_mesh(mesh):
    vtx_pos, pos_idx, vtx_uv, uv_idx = None, None, None, None
    if isinstance(mesh, str):
        mesh_path = os.path.abspath(mesh)
        import trimesh
        # Use process=False to avoid stripping data
        m = trimesh.load(mesh_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.to_geometry()
        
        vtx_pos = _safe_extract_attribute(m, "vertices")
        pos_idx = _safe_extract_attribute(m, "faces")
        vtx_uv = _safe_extract_attribute(m, "visual.uv")
        
        if vtx_uv is None and hasattr(m, 'vertex_attributes'):
            vtx_uv = m.vertex_attributes.get('texcoord') or m.vertex_attributes.get('uv')
    else:
        vtx_pos = _safe_extract_attribute(mesh, "vertices")
        pos_idx = _safe_extract_attribute(mesh, "faces")
        vtx_uv = _safe_extract_attribute(mesh, "visual.uv")

    uv_idx = pos_idx if (vtx_uv is not None and uv_idx is None) else uv_idx
    
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)
    
    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data

def _get_base_path_and_name(mesh_path: str) -> Tuple[str, str]:
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name

def _save_texture_map(texture: np.ndarray, base_path: str, suffix: str = "", image_format: str = ".jpg", as_grayscale: bool = False) -> str:
    path = f"{base_path}{suffix}{image_format}"
    processed_texture = (np.clip(texture, 0, 1) * 255).astype(np.uint8) if texture.dtype != np.uint8 else texture
    if as_grayscale or (processed_texture.ndim == 3 and processed_texture.shape[-1] == 1):
        if processed_texture.ndim == 3:
            processed_texture = processed_texture.squeeze(-1)
        img = Image.fromarray(processed_texture, mode="L")
    else:
        img = Image.fromarray(processed_texture, mode="RGB")
    img.save(path)
    return os.path.basename(path)

def _write_mtl_properties(f, properties: Dict[str, Any]):
    for key, value in properties.items():
        if isinstance(value, (list, tuple)): f.write(f"{key} {' '.join(map(str, value))}\n")
        else: f.write(f"{key} {value}\n")

def _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name) -> str:
    # Use buffered writes or np.savetxt for speed
    buffer = StringIO()
    buffer.write(f"mtllib {name}.mtl\no {name}\n")
    for v in vtx_pos: buffer.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for vt in vtx_uv: buffer.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
    buffer.write("s 0\nusemtl Material\n")
    
    # Faces: f v1/vt1 v2/vt2 v3/vt3
    faces = np.stack([pos_idx + 1, uv_idx + 1], axis=-1)
    for f in faces:
        buffer.write(f"f {f[0,0]}/{f[0,1]} {f[1,0]}/{f[1,1]} {f[2,0]}/{f[2,1]}\n")
        
    return buffer.getvalue()

def save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None):
    base_path, name = _get_base_path_and_name(mesh_path)
    obj_content = _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name)
    with open(mesh_path, "w") as f: f.write(obj_content)
    
    texture_maps = {"diffuse": _save_texture_map(texture, base_path)}
    if metallic is not None: texture_maps["metallic"] = _save_texture_map(metallic, base_path, "_metallic", as_grayscale=True)
    if roughness is not None: texture_maps["roughness"] = _save_texture_map(roughness, base_path, "_roughness", as_grayscale=True)
    if normal is not None: texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal")
    
    with open(f"{base_path}.mtl", "w") as f:
        f.write("newmtl Material\n")
        props = {"Kd": [0.8, 0.8, 0.8], "illum": 2, "map_Kd": texture_maps["diffuse"]}
        _write_mtl_properties(f, props)
        if "metallic" in texture_maps: f.write(f"map_Pm {texture_maps['metallic']}\n")
        if "roughness" in texture_maps: f.write(f"map_Pr {texture_maps['roughness']}\n")
        if "normal" in texture_maps: f.write(f"map_Bump -bm 1.0 {texture_maps['normal']}\n")

def save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None):
    save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic, roughness, normal)

def save_glb_mesh(glb_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from PIL import Image
    try:
        glb_path = os.path.abspath(glb_path)
        os.makedirs(os.path.dirname(glb_path), exist_ok=True)
        
        # Unroll vertices/UVs if UV indices differ from vertex position indices (UV seams)
        if uv_idx is not None and (len(vtx_uv) != len(vtx_pos) or not np.array_equal(pos_idx, uv_idx)):
            flat_pos = pos_idx.flatten()
            flat_uv = uv_idx.flatten()
            vertices = vtx_pos[flat_pos]
            uvs = vtx_uv[flat_uv]
            faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        else:
            vertices = vtx_pos
            uvs = vtx_uv
            faces = pos_idx
            
        h, w = texture.shape[0], texture.shape[1]
        base_color_img = Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
        
        material_kwargs = {
            "name": "Material",
            "baseColorTexture": base_color_img,
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        }
        
        if metallic is not None or roughness is not None:
            r_img = Image.fromarray((np.clip(roughness[..., 0], 0, 1) * 255).astype(np.uint8)).resize((w, h)) if roughness is not None else Image.new("L", (w, h), 128)
            m_img = Image.fromarray((np.clip(metallic[..., 0], 0, 1) * 255).astype(np.uint8)).resize((w, h)) if metallic is not None else Image.new("L", (w, h), 0)
            ao_img = Image.new("L", (w, h), 255)
            mr_img = Image.merge("RGB", (ao_img, r_img, m_img))
            material_kwargs["metallicRoughnessTexture"] = mr_img
            
        if normal is not None:
            norm_img = Image.fromarray((np.clip(normal, 0, 1) * 255).astype(np.uint8)).resize((w, h))
            material_kwargs["normalTexture"] = norm_img
            
        pbr_mat = PBRMaterial(**material_kwargs)
        
        visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr_mat)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        mesh.fix_normals()
        mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=len(vertices),
            faces=faces,
            face_normals=mesh.face_normals
        )
        
        mesh.export(glb_path, file_type='glb')
        return os.path.exists(glb_path)
    except Exception as e:
        print(f"GLB Error (in-memory): {e}")
        import sys
        sys.stdout.flush()
        return False

def convert_obj_to_glb(obj_path, glb_path, shade_type="SMOOTH", auto_smooth_angle=60, merge_vertices=False):
    import trimesh
    from PIL import Image
    try:
        obj_path = os.path.abspath(obj_path)
        glb_path = os.path.abspath(glb_path)
        base_path = os.path.splitext(obj_path)[0]
        print(f"GLB Debug: Using trimesh to convert {obj_path}")
        
        # Load mesh with trimesh
        # trimesh handles OBJ+MTL+Textures automatically if they are in the same folder
        mesh = trimesh.load(obj_path, process=False) # process=False to keep exact topography
        
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, we might want to merge it or just export the whole thing
            # For Grid, it's usually a single mesh
            print("GLB Debug: Mesh loaded as Scene, exporting...")
        
        # Check if PBR maps (metallic, roughness, normal) exist
        diffuse_img_path = f"{base_path}.jpg"
        metallic_img_path = f"{base_path}_metallic.jpg"
        roughness_img_path = f"{base_path}_roughness.jpg"
        normal_img_path = f"{base_path}_normal.jpg"
        
        if os.path.exists(metallic_img_path) or os.path.exists(roughness_img_path) or os.path.exists(normal_img_path):
            try:
                from trimesh.visual.material import PBRMaterial
                
                base_color_img = Image.open(diffuse_img_path).convert("RGBA") if os.path.exists(diffuse_img_path) else None
                w, h = (base_color_img.size if base_color_img else (1024, 1024))
                
                # GLTF PBR standard: Green = Roughness, Blue = Metallic, Red = Occlusion (255)
                if os.path.exists(roughness_img_path):
                    r_img = Image.open(roughness_img_path).convert("L").resize((w, h))
                else:
                    r_img = Image.new("L", (w, h), 128)
                    
                if os.path.exists(metallic_img_path):
                    m_img = Image.open(metallic_img_path).convert("L").resize((w, h))
                else:
                    m_img = Image.new("L", (w, h), 0)
                    
                ao_img = Image.new("L", (w, h), 255)
                mr_img = Image.merge("RGB", (ao_img, r_img, m_img))
                
                norm_img = Image.open(normal_img_path).convert("RGB").resize((w, h)) if os.path.exists(normal_img_path) else None
                
                pbr_mat = PBRMaterial(
                    name="Material",
                    baseColorTexture=base_color_img,
                    metallicRoughnessTexture=mr_img,
                    normalTexture=norm_img,
                    metallicFactor=1.0,
                    roughnessFactor=1.0,
                )
                
                if isinstance(mesh, trimesh.Scene):
                    for geom in mesh.geometry.values():
                        if hasattr(geom, 'visual') and hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                            geom.visual.material = pbr_mat
                elif hasattr(mesh, 'visual'):
                    mesh.visual.material = pbr_mat
            except Exception as pe:
                print(f"GLB PBR Material creation warning: {pe}")
        
        # Apply smoothing if requested
        if shade_type == "SMOOTH" or shade_type == "AUTO_SMOOTH":
            # Trimesh doesn't have an exact "auto-smooth" operator like Blender, 
            # but it defaults to smooth shading if vertex normals are present.
            pass
            
        mesh.export(glb_path, file_type='glb')
        
        if os.path.exists(glb_path):
            print(f"GLB Debug: Export successful to {glb_path}")
            return True
        else:
            print("GLB Error: Trimesh export finished but file not found")
            return False
            
    except Exception as e:
        print(f"GLB Error (trimesh): {e}")
        import sys
        sys.stdout.flush()
        return False
