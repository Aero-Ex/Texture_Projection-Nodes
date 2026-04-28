from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial, PBRMaterial

def parse_texture_visuals(texture_visuals:TextureVisuals) \
    -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    if isinstance(texture_visuals.material, SimpleMaterial):
        map_Kd = texture_visuals.material.image
        if map_Kd is None:
            color_Kd = texture_visuals.material.diffuse
            if color_Kd is not None:
                map_Kd = Image.new(mode='RGBA', size=(4, 4), color=tuple(color_Kd))
        map_Ks = None
        map_normal = None
    elif isinstance(texture_visuals.material, PBRMaterial):
        map_Kd = texture_visuals.material._data.get('baseColorTexture', None)
        if map_Kd is None:
            map_Kd = texture_visuals.material._data.get('emissiveTexture', None)
        if map_Kd is None:
            color_Kd = texture_visuals.material._data.get('baseColorFactor', None)
            if color_Kd is None:
                color_Kd = texture_visuals.material._data.get('emissiveFactor', None)
            if color_Kd is not None:
                map_Kd = Image.new(mode='RGBA', size=(4, 4), color=tuple(color_Kd))
        map_Ks = texture_visuals.material._data.get('metallicRoughnessTexture', None)
        if map_Ks is None:
            color_m = texture_visuals.material._data.get('metallicFactor', None)
            color_r = texture_visuals.material._data.get('roughnessFactor', None)
            if color_m is not None or color_r is not None:
                if color_m is None: color_m = 0.0
                if color_r is None: color_r = 1.0
                color_Ks = np.asarray([1.0, color_r, color_m], dtype=np.float32)
                color_Ks = (color_Ks.clip(0.0, 1.0) * 255).astype(np.uint8)
                map_Ks = Image.new(mode='RGB', size=(4, 4), color=tuple(color_Ks))
        map_normal = texture_visuals.material._data.get('normalTexture', None)
    else:
        map_Kd, map_Ks, map_normal = None, None, None
    return map_Kd, map_Ks, map_normal
