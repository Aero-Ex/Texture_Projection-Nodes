import os
import cv2
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
        
        # If trimesh fails to find UVs in a GLB, try pygltflib
        if vtx_uv is None and mesh_path.lower().endswith(('.glb', '.gltf')):
            try:
                import pygltflib
                gltf = pygltflib.GLTF2().load(mesh_path)
                for g_mesh in gltf.meshes:
                    for primitive in g_mesh.primitives:
                        if primitive.attributes.TEXCOORD_0 is not None:
                            accessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
                            bv = gltf.bufferViews[accessor.bufferView]
                            buffer = gltf.buffers[bv.buffer]
                            raw_data = gltf.decode_data(buffer.uri) if buffer.uri else gltf.binary_blob()
                            
                            start = bv.byteOffset + (accessor.byteOffset or 0)
                            end = start + accessor.count * 8
                            
                            vtx_uv = np.frombuffer(raw_data[start:end], dtype=np.float32).reshape(-1, 2).copy()
                            uv_idx = pos_idx if vtx_uv.shape[0] == vtx_pos.shape[0] else None
                            break
                    if vtx_uv is not None: break
            except Exception as e:
                print(f"pygltflib UV extraction failed: {e}")

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

def _save_texture_map(texture: np.ndarray, base_path: str, suffix: str = "", image_format: str = ".jpg", color_convert: Optional[int] = None) -> str:
    path = f"{base_path}{suffix}{image_format}"
    processed_texture = (texture * 255).astype(np.uint8)
    if color_convert is not None:
        processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        cv2.imwrite(path, processed_texture[..., ::-1])  # RGB to BGR
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
    if metallic is not None: texture_maps["metallic"] = _save_texture_map(metallic, base_path, "_metallic", color_convert=cv2.COLOR_RGB2GRAY)
    if roughness is not None: texture_maps["roughness"] = _save_texture_map(roughness, base_path, "_roughness", color_convert=cv2.COLOR_RGB2GRAY)
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

def convert_obj_to_glb(obj_path, glb_path, shade_type="SMOOTH", auto_smooth_angle=60, merge_vertices=False):
    import trimesh
    try:
        obj_path = os.path.abspath(obj_path)
        glb_path = os.path.abspath(glb_path)
        print(f"GLB Debug: Using trimesh to convert {obj_path}")
        
        # Load mesh with trimesh
        # trimesh handles OBJ+MTL+Textures automatically if they are in the same folder
        mesh = trimesh.load(obj_path, process=False) # process=False to keep exact topography
        
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, we might want to merge it or just export the whole thing
            # For Grid, it's usually a single mesh
            print("GLB Debug: Mesh loaded as Scene, exporting...")
        
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
