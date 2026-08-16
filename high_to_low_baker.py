import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh

try:
    from .Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import (
        load_mesh as load_mesh_utils,
        save_glb_mesh
    )
except (ImportError, ValueError):
    from Texture_Projection.Renderer.DifferentiableRenderer.mesh_utils import (
        load_mesh as load_mesh_utils,
        save_glb_mesh
    )


def compute_vertex_tangents(vertices, normals, faces, uvs, uv_faces):
    """
    Computes per-vertex tangent and bitangent vectors using triangle geometry and UV coordinates.
    """
    num_verts = len(vertices)
    tangents = np.zeros((num_verts, 3), dtype=np.float32)
    bitangents = np.zeros((num_verts, 3), dtype=np.float32)

    # If UV faces differ from vertex faces, unroll or use direct per-vertex UVs if matching
    if len(uvs) == num_verts and np.array_equal(faces, uv_faces):
        tri_v = vertices[faces] # [F, 3, 3]
        tri_uv = uvs[faces]     # [F, 3, 2]
    else:
        # Per-corner indexing
        tri_v = vertices[faces]     # [F, 3, 3]
        tri_uv = uvs[uv_faces]      # [F, 3, 2]

    v0 = tri_v[:, 0]
    v1 = tri_v[:, 1]
    v2 = tri_v[:, 2]

    uv0 = tri_uv[:, 0]
    uv1 = tri_uv[:, 1]
    uv2 = tri_uv[:, 2]

    delta_pos1 = v1 - v0
    delta_pos2 = v2 - v0

    delta_uv1 = uv1 - uv0
    delta_uv2 = uv2 - uv0

    denom = (delta_uv1[:, 0] * delta_uv2[:, 1] - delta_uv1[:, 1] * delta_uv2[:, 0])
    denom[np.abs(denom) < 1e-8] = 1e-8
    r = 1.0 / denom[:, None]

    tri_tangent = (delta_pos1 * delta_uv2[:, 1:2] - delta_pos2 * delta_uv1[:, 1:2]) * r
    tri_bitangent = (delta_pos2 * delta_uv1[:, 0:1] - delta_pos1 * delta_uv2[:, 0:1]) * r

    # Accumulate into vertices
    for i in range(3):
        idx = faces[:, i]
        np.add.at(tangents, idx, tri_tangent)
        np.add.at(bitangents, idx, tri_bitangent)

    # Gram-Schmidt orthogonalize tangents against vertex normals
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_len[norm_len < 1e-8] = 1e-8
    normals_unit = normals / norm_len

    dot_nt = np.sum(normals_unit * tangents, axis=-1, keepdims=True)
    tangents_ortho = tangents - normals_unit * dot_nt
    tan_len = np.linalg.norm(tangents_ortho, axis=-1, keepdims=True)
    tan_len[tan_len < 1e-8] = 1e-8
    tangents_unit = tangents_ortho / tan_len

    # Compute bitangent with correct handedness
    cross_nt = np.cross(normals_unit, tangents_unit)
    dot_cross_b = np.sum(cross_nt * bitangents, axis=-1, keepdims=True)
    sign = np.where(dot_cross_b < 0, -1.0, 1.0)
    bitangents_final = cross_nt * sign

    return tangents_unit.astype(np.float32), bitangents_final.astype(np.float32)


def rasterize_lowpoly_uv_space(low_mesh, resolution, device="cuda"):
    """
    Rasterizes the low-poly mesh in UV space to obtain per-texel 3D positions (P_low),
    interpolated normals (N_low), tangents (T_low), and bitangents (B_low).
    """
    try:
        import nvdiffrast.torch as dr
        has_nvdiffrast = True
    except ImportError:
        has_nvdiffrast = False

    # Extract geometry
    vtx_pos = np.asarray(low_mesh.vertices, dtype=np.float32)
    faces = np.asarray(low_mesh.faces, dtype=np.int32)
    
    # Robust UV extraction
    vtx_uv = None
    if hasattr(low_mesh, 'visual') and hasattr(low_mesh.visual, 'uv') and low_mesh.visual.uv is not None and len(low_mesh.visual.uv) > 0:
        vtx_uv = np.asarray(low_mesh.visual.uv, dtype=np.float32)
    elif hasattr(low_mesh, 'vertex_attributes') and ('texcoord' in low_mesh.vertex_attributes or 'uv' in low_mesh.vertex_attributes):
        vtx_uv = np.asarray(low_mesh.vertex_attributes.get('texcoord') or low_mesh.vertex_attributes.get('uv'), dtype=np.float32)

    if vtx_uv is None or len(vtx_uv) == 0:
        print("[HighToLow Baker] Low-poly mesh missing UV coordinates, auto-unwrapping with xatlas...")
        import xatlas
        vmapping, indices, uvs = xatlas.parametrize(vtx_pos, faces)
        vtx_pos = vtx_pos[vmapping]
        faces = indices.astype(np.int32)
        vtx_uv = uvs.astype(np.float32)
        # Update low_mesh in place
        low_mesh = trimesh.Trimesh(vertices=vtx_pos, faces=faces, visual=trimesh.visual.texture.TextureVisuals(uv=vtx_uv), process=False)

    uv_faces = faces
    normals = np.asarray(low_mesh.vertex_normals, dtype=np.float32)
    tangents, bitangents = compute_vertex_tangents(vtx_pos, normals, faces, vtx_uv, uv_faces)

    if has_nvdiffrast and torch.cuda.is_available() and device.startswith("cuda"):
        glctx = dr.RasterizeGLContext() if hasattr(dr, 'RasterizeGLContext') else dr.RasterizeCudaContext()
        
        # Scale UV to clip space: U in [-1, 1], V in [-1, 1] (row 0 at top)
        uv_clip = np.zeros((len(vtx_uv), 4), dtype=np.float32)
        uv_clip[:, 0] = vtx_uv[:, 0] * 2.0 - 1.0
        uv_clip[:, 1] = 1.0 - vtx_uv[:, 1] * 2.0
        uv_clip[:, 2] = 0.0
        uv_clip[:, 3] = 1.0

        uv_clip_th = torch.from_numpy(uv_clip).to(device).unsqueeze(0)
        uv_faces_th = torch.from_numpy(uv_faces).to(device).int()

        vtx_pos_th = torch.from_numpy(vtx_pos).to(device).unsqueeze(0)
        normals_th = torch.from_numpy(normals).to(device).unsqueeze(0)
        tangents_th = torch.from_numpy(tangents).to(device).unsqueeze(0)
        bitangents_th = torch.from_numpy(bitangents).to(device).unsqueeze(0)

        rast_out, _ = dr.rasterize(glctx, uv_clip_th, uv_faces_th, resolution=[resolution, resolution])
        valid_mask_th = (rast_out[0, ..., 3] > 0) # [H, W]

        pos_map, _ = dr.interpolate(vtx_pos_th, rast_out, uv_faces_th)
        norm_map, _ = dr.interpolate(normals_th, rast_out, uv_faces_th)
        tan_map, _ = dr.interpolate(tangents_th, rast_out, uv_faces_th)
        bitan_map, _ = dr.interpolate(bitangents_th, rast_out, uv_faces_th)

        pos_map = pos_map[0] # [H, W, 3]
        norm_map = F.normalize(norm_map[0], dim=-1)
        tan_map = F.normalize(tan_map[0], dim=-1)
        bitan_map = F.normalize(bitan_map[0], dim=-1)

        return (
            pos_map.cpu().numpy(),
            norm_map.cpu().numpy(),
            tan_map.cpu().numpy(),
            bitan_map.cpu().numpy(),
            valid_mask_th.cpu().numpy()
        )
    else:
        # CPU UV rasterization fallback via barycentric grid sampling
        H = resolution
        W = resolution
        pos_map = np.zeros((H, W, 3), dtype=np.float32)
        norm_map = np.zeros((H, W, 3), dtype=np.float32)
        tan_map = np.zeros((H, W, 3), dtype=np.float32)
        bitan_map = np.zeros((H, W, 3), dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)

        # Triangle rasterization
        tri_uvs = vtx_uv[uv_faces] # [F, 3, 2]
        tri_pos = vtx_pos[faces]   # [F, 3, 3]
        tri_norm = normals[faces]  # [F, 3, 3]
        tri_tan = tangents[faces]  # [F, 3, 3]
        tri_bitan = bitangents[faces] # [F, 3, 3]

        for f in range(len(faces)):
            uv0, uv1, uv2 = tri_uvs[f, 0], tri_uvs[f, 1], tri_uvs[f, 2]
            # Convert to pixel space (0 at top)
            px0 = uv0[0] * (W - 1), (1.0 - uv0[1]) * (H - 1)
            px1 = uv1[0] * (W - 1), (1.0 - uv1[1]) * (H - 1)
            px2 = uv2[0] * (W - 1), (1.0 - uv2[1]) * (H - 1)

            min_x = max(0, int(np.floor(min(px0[0], px1[0], px2[0]))))
            max_x = min(W - 1, int(np.ceil(max(px0[0], px1[0], px2[0]))))
            min_y = max(0, int(np.floor(min(px0[1], px1[1], px2[1]))))
            max_y = min(H - 1, int(np.ceil(max(px0[1], px1[1], px2[1]))))

            if min_x > max_x or min_y > max_y: continue

            # Compute edge functions
            det = (px1[1] - px2[1]) * (px0[0] - px2[0]) + (px2[0] - px1[0]) * (px0[1] - px2[1])
            if abs(det) < 1e-6: continue
            inv_det = 1.0 / det

            xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
            xs, ys = xs.flatten(), ys.flatten()

            w0 = ((px1[1] - px2[1]) * (xs - px2[0]) + (px2[0] - px1[0]) * (ys - px2[1])) * inv_det
            w1 = ((px2[1] - px0[1]) * (xs - px2[0]) + (px0[0] - px2[0]) * (ys - px2[1])) * inv_det
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside): continue

            valid_xs = xs[inside]
            valid_ys = ys[inside]
            w0_in = w0[inside, None]
            w1_in = w1[inside, None]
            w2_in = w2[inside, None]

            pos_interp = w0_in * tri_pos[f, 0] + w1_in * tri_pos[f, 1] + w2_in * tri_pos[f, 2]
            norm_interp = w0_in * tri_norm[f, 0] + w1_in * tri_norm[f, 1] + w2_in * tri_norm[f, 2]
            tan_interp = w0_in * tri_tan[f, 0] + w1_in * tri_tan[f, 1] + w2_in * tri_tan[f, 2]
            bitan_interp = w0_in * tri_bitan[f, 0] + w1_in * tri_bitan[f, 1] + w2_in * tri_bitan[f, 2]

            pos_map[valid_ys, valid_xs] = pos_interp
            norm_map[valid_ys, valid_xs] = norm_interp
            tan_map[valid_ys, valid_xs] = tan_interp
            bitan_map[valid_ys, valid_xs] = bitan_interp
            valid_mask[valid_ys, valid_xs] = True

        # Normalize vectors
        norm_len = np.linalg.norm(norm_map, axis=-1, keepdims=True)
        norm_len[norm_len < 1e-8] = 1e-8
        norm_map = norm_map / norm_len

        tan_len = np.linalg.norm(tan_map, axis=-1, keepdims=True)
        tan_len[tan_len < 1e-8] = 1e-8
        tan_map = tan_map / tan_len

        bitan_len = np.linalg.norm(bitan_map, axis=-1, keepdims=True)
        bitan_len[bitan_len < 1e-8] = 1e-8
        bitan_map = bitan_map / bitan_len

        return pos_map, norm_map, tan_map, bitan_map, valid_mask


def sample_texture_bilinear(img_array, uv_coords):
    """
    Bilinearly samples an image array of shape [H, W, C] using UV coordinates [N, 2].
    UV coordinates range from 0 to 1 (U right, V up).
    """
    if img_array is None:
        return None

    H, W = img_array.shape[:2]
    u = uv_coords[:, 0] % 1.0
    v = (1.0 - (uv_coords[:, 1] % 1.0)) # Flip V for image array coordinates

    x = u * (W - 1)
    y = v * (H - 1)

    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    if img_array.ndim == 2:
        img_array = img_array[..., None]

    top_left = img_array[y0, x0]
    top_right = img_array[y0, x1]
    bottom_left = img_array[y1, x0]
    bottom_right = img_array[y1, x1]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx

    sampled = top * (1.0 - wy) + bottom * wy
    return sampled


def fast_gpu_seam_dilate(texture_th, mask_th, iterations=16):
    """
    Fast GPU-accelerated seam dilation for baked textures.
    texture_th: [H, W, C] on CUDA
    mask_th: [H, W] or [H, W, 1] on CUDA
    """
    if mask_th.all():
        return texture_th

    H, W = texture_th.shape[:2]
    C = texture_th.shape[-1]

    tex = texture_th.permute(2, 0, 1).unsqueeze(0).contiguous()
    m = (mask_th > 0).float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.permute(2, 0, 1).unsqueeze(0)

    # Bounded edge dilation to eliminate UV seam borders without smearing into adjacent charts
    kernel = torch.ones((1, 1, 3, 3), device=tex.device, dtype=tex.dtype)
    kernel_c = torch.ones((C, 1, 3, 3), device=tex.device, dtype=tex.dtype)

    for _ in range(iterations):
        if m.all(): break
        valid_neighbors = F.conv2d(m, kernel, padding=1)
        sum_neighbors = F.conv2d(tex * m, kernel_c, padding=1, groups=C)
        avg_neighbors = sum_neighbors / torch.clamp(valid_neighbors, min=1e-5)
        update_mask = (m == 0) & (valid_neighbors > 0)
        tex = torch.where(update_mask, avg_neighbors, tex)
        m = torch.where(update_mask, torch.ones_like(m), m)

    return tex.squeeze(0).permute(1, 2, 0).contiguous()


def extract_mesh_textures(mesh):
    """
    Extracts Diffuse/Albedo, Metallic, Roughness, and Normal textures from a Trimesh object.
    """
    diffuse = None
    roughness = None
    metallic = None
    normal = None

    if hasattr(mesh, 'visual') and mesh.visual is not None:
        material = getattr(mesh.visual, 'material', None)
        if material is not None:
            # Base color
            base_col = getattr(material, 'baseColorTexture', None) or getattr(material, 'image', None)
            if base_col is not None:
                diffuse = np.asarray(base_col.convert("RGB"), dtype=np.float32) / 255.0

            # MetallicRoughness (G = Roughness, B = Metallic in GLTF)
            mr_tex = getattr(material, 'metallicRoughnessTexture', None)
            if mr_tex is not None:
                mr_np = np.asarray(mr_tex.convert("RGB"), dtype=np.float32) / 255.0
                roughness = mr_np[..., 1:2]
                metallic = mr_np[..., 2:3]
            else:
                # Fallback to scalar factors
                r_factor = getattr(material, 'roughnessFactor', 0.5)
                m_factor = getattr(material, 'metallicFactor', 0.0)
                if r_factor is not None:
                    roughness = np.full((128, 128, 1), float(r_factor), dtype=np.float32)
                if m_factor is not None:
                    metallic = np.full((128, 128, 1), float(m_factor), dtype=np.float32)

            # Normal texture
            norm_tex = getattr(material, 'normalTexture', None)
            if norm_tex is not None:
                normal = np.asarray(norm_tex.convert("RGB"), dtype=np.float32) / 255.0

    return diffuse, roughness, metallic, normal


def bake_high_to_low_poly(
    high_mesh,
    low_mesh,
    resolution=2048,
    ray_max_dist=0.05,
    cage_offset=0.005,
    bake_diffuse=True,
    bake_normal=True,
    bake_roughness=True,
    bake_metallic=True,
    bake_height=False,
    bake_ao=False,
    normal_format="OpenGL (Y+)",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main High-to-Low Poly Baking Engine.
    Bakes normal maps, PBR maps, displacement, and AO from high_mesh to low_mesh.
    """
    print(f"[HighToLow Baker] Stage 1/5: Rasterizing low-poly UV space ({resolution}x{resolution})...")
    sys.stdout.flush()

    # 1. Rasterize Low-Poly UV Space
    pos_map, norm_map, tan_map, bitan_map, valid_mask = rasterize_lowpoly_uv_space(
        low_mesh, resolution, device=device
    )

    valid_indices = np.where(valid_mask)
    num_valid = len(valid_indices[0])

    if num_valid == 0:
        raise RuntimeError("Low-poly mesh has no rasterized UV pixels. Check mesh UV unwrapping.")

    print(f"[HighToLow Baker] Stage 2/5: Raycasting against high-poly mesh ({num_valid:,} active texels)...")
    sys.stdout.flush()

    # Low-poly texel properties
    p_low = pos_map[valid_indices]
    n_low = norm_map[valid_indices]
    t_low = tan_map[valid_indices]
    b_low = bitan_map[valid_indices]

    # 2. Raycast from low-poly towards high-poly surface
    ray_origins = p_low + n_low * cage_offset
    ray_dirs = -n_low

    hit_points = np.zeros_like(p_low)
    hit_normals = np.zeros_like(n_low)
    hit_uvs = np.zeros((num_valid, 2), dtype=np.float32)
    hit_mask = np.zeros(num_valid, dtype=bool)

    high_vtx = np.asarray(high_mesh.vertices, dtype=np.float32)
    high_faces = np.asarray(high_mesh.faces, dtype=np.int32)
    high_normals = np.asarray(high_mesh.vertex_normals, dtype=np.float32)
    has_high_uv = hasattr(high_mesh.visual, 'uv') and high_mesh.visual.uv is not None
    high_uvs = np.asarray(high_mesh.visual.uv, dtype=np.float32) if has_high_uv else None

    # Chunked forward ray intersection for steady progress and low memory overhead
    chunk_size = 150000
    num_chunks = int(np.ceil(num_valid / chunk_size))

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(num_valid, c_start + chunk_size)
        c_orig = ray_origins[c_start:c_end]
        c_dir = ray_dirs[c_start:c_end]

        c_locs, c_idx_ray, c_idx_tri = high_mesh.ray.intersects_location(
            ray_origins=c_orig,
            ray_directions=c_dir
        )

        if len(c_locs) > 0:
            dists = np.linalg.norm(c_locs - c_orig[c_idx_ray], axis=-1)
            valid_hits = dists <= (ray_max_dist + cage_offset)

            c_locs = c_locs[valid_hits]
            c_idx_ray = c_idx_ray[valid_hits]
            c_idx_tri = c_idx_tri[valid_hits]

            unique_rays, first_indices = np.unique(c_idx_ray, return_index=True)
            closest_locs = c_locs[first_indices]
            closest_tris = c_idx_tri[first_indices]

            global_ray_idx = c_start + unique_rays
            hit_points[global_ray_idx] = closest_locs
            hit_mask[global_ray_idx] = True

            tri_verts = high_vtx[high_faces[closest_tris]]
            bary = trimesh.triangles.points_to_barycentric(tri_verts, closest_locs)
            bary = np.clip(bary, 0.0, 1.0)
            bary /= np.sum(bary, axis=-1, keepdims=True)

            tri_norm = high_normals[high_faces[closest_tris]]
            interp_norm = (bary[:, 0:1] * tri_norm[:, 0] +
                           bary[:, 1:2] * tri_norm[:, 1] +
                           bary[:, 2:3] * tri_norm[:, 2])
            norm_len = np.linalg.norm(interp_norm, axis=-1, keepdims=True)
            norm_len[norm_len < 1e-8] = 1e-8
            interp_norm = interp_norm / norm_len

            # Backface rejection: only accept hits facing generally in the same direction
            dot_align = np.sum(n_low[global_ray_idx] * interp_norm, axis=-1)
            valid_facing = dot_align > -0.1

            if np.any(valid_facing):
                valid_global_idx = global_ray_idx[valid_facing]
                hit_points[valid_global_idx] = closest_locs[valid_facing]
                hit_normals[valid_global_idx] = interp_norm[valid_facing]
                hit_mask[valid_global_idx] = True

                if has_high_uv:
                    tri_uv = high_uvs[high_faces[closest_tris[valid_facing]]]
                    interp_uv = (bary[valid_facing, 0:1] * tri_uv[:, 0] +
                                 bary[valid_facing, 1:2] * tri_uv[:, 1] +
                                 bary[valid_facing, 2:3] * tri_uv[:, 2])
                    hit_uvs[valid_global_idx] = interp_uv

        pct = int(((c + 1) / num_chunks) * 100)
        print(f"[HighToLow Baker] Inward raycast progress: {pct}% ({min(c_end, num_valid):,}/{num_valid:,} texels)")
        sys.stdout.flush()

    # 3. Bi-directional raycast for any missed points (outward ray check via fast BVH)
    missed_idx = np.where(~hit_mask)[0]
    if len(missed_idx) > 0:
        print(f"[HighToLow Baker] Resolving {len(missed_idx):,} missed texels via outward raycast...")
        sys.stdout.flush()

        out_origins = p_low[missed_idx] - n_low[missed_idx] * cage_offset
        out_dirs = n_low[missed_idx]

        out_locs, out_idx_ray, out_idx_tri = high_mesh.ray.intersects_location(
            ray_origins=out_origins,
            ray_directions=out_dirs
        )

        if len(out_locs) > 0:
            dists = np.linalg.norm(out_locs - out_origins[out_idx_ray], axis=-1)
            valid_hits = dists <= (ray_max_dist + cage_offset)

            out_locs = out_locs[valid_hits]
            out_idx_ray = out_idx_ray[valid_hits]
            out_idx_tri = out_idx_tri[valid_hits]

            unique_out, first_out = np.unique(out_idx_ray, return_index=True)
            closest_locs = out_locs[first_out]
            closest_tris = out_idx_tri[first_out]
            global_out_idx = missed_idx[unique_out]

            tri_verts = high_vtx[high_faces[closest_tris]]
            bary = trimesh.triangles.points_to_barycentric(tri_verts, closest_locs)
            bary = np.clip(bary, 0.0, 1.0)
            bary /= np.sum(bary, axis=-1, keepdims=True)

            tri_norm = high_normals[high_faces[closest_tris]]
            interp_norm = (bary[:, 0:1] * tri_norm[:, 0] +
                           bary[:, 1:2] * tri_norm[:, 1] +
                           bary[:, 2:3] * tri_norm[:, 2])
            norm_len = np.linalg.norm(interp_norm, axis=-1, keepdims=True)
            # Backface rejection - only accept high-poly surfaces facing the same direction
            dot_align = np.sum(n_low[global_out_idx] * interp_norm, axis=-1)
            valid_facing = dot_align > 0.05

            if np.any(valid_facing):
                valid_out_idx = global_out_idx[valid_facing]
                hit_points[valid_out_idx] = closest_locs[valid_facing]
                hit_normals[valid_out_idx] = interp_norm[valid_facing]
                hit_mask[valid_out_idx] = True

                if has_high_uv:
                    tri_uv = high_uvs[high_faces[closest_tris[valid_facing]]]
                    interp_uv = (bary[valid_facing, 0:1] * tri_uv[:, 0] +
                                 bary[valid_facing, 1:2] * tri_uv[:, 1] +
                                 bary[valid_facing, 2:3] * tri_uv[:, 2])
                    hit_uvs[valid_out_idx] = interp_uv

    # 4. Normal-aware spatial cKDTree query for any remaining missed texels
    # Guarantees 100% full surface coverage while preventing ghosting/double projections
    final_missed = np.where(~hit_mask)[0]
    if len(final_missed) > 0:
        print(f"[HighToLow Baker] Resolving {len(final_missed):,} crevice/boundary texels via normal-aware spatial query...")
        sys.stdout.flush()
        from scipy.spatial import cKDTree
        kdtree = cKDTree(high_vtx)
        k_val = min(16, len(high_vtx))
        distances, nearest_idx = kdtree.query(p_low[final_missed], k=k_val, workers=-1)

        if k_val > 1:
            candidate_normals = high_normals[nearest_idx] # [N, K, 3]
            dots = np.sum(candidate_normals * n_low[final_missed, None, :], axis=-1) # [N, K]
            scores = dots - (distances * 5.0)
            best_k = np.argmax(scores, axis=-1)
            chosen_idx = nearest_idx[np.arange(len(final_missed)), best_k]
        else:
            chosen_idx = nearest_idx

        hit_points[final_missed] = high_vtx[chosen_idx]
        hit_normals[final_missed] = high_normals[chosen_idx]
        if has_high_uv:
            hit_uvs[final_missed] = high_uvs[chosen_idx]
        hit_mask[final_missed] = True

    # 100% of valid low-poly texels are now populated
    valid_hit_pixels = valid_indices
    dilated_mask = valid_mask

    print(f"[HighToLow Baker] Stage 3/5: Sampling PBR textures & calculating Tangent-Space Normals ({num_valid:,}/{num_valid:,} texels resolved)...")
    sys.stdout.flush()

    # 4. Extract High-Poly Source Textures
    high_diffuse_tex, high_roughness_tex, high_metallic_tex, high_normal_tex = extract_mesh_textures(high_mesh)

    # Initialize output map arrays
    out_diffuse = np.zeros((resolution, resolution, 3), dtype=np.float32)
    out_normal = np.full((resolution, resolution, 3), [0.5, 0.5, 1.0], dtype=np.float32)
    out_roughness = np.full((resolution, resolution, 1), 0.5, dtype=np.float32)
    out_metallic = np.zeros((resolution, resolution, 1), dtype=np.float32)
    out_height = np.full((resolution, resolution, 1), 0.5, dtype=np.float32)
    out_ao = np.ones((resolution, resolution, 1), dtype=np.float32)

    # 5. Compute Tangent-Space Normal Map
    if bake_normal:
        # N_ts = [dot(T, N_high), dot(B, N_high), dot(N, N_high)]
        ts_x = np.sum(t_low * hit_normals, axis=-1)
        ts_y = np.sum(b_low * hit_normals, axis=-1)
        ts_z = np.sum(n_low * hit_normals, axis=-1)

        ts_norm = np.stack([ts_x, ts_y, ts_z], axis=-1)
        ts_len = np.linalg.norm(ts_norm, axis=-1, keepdims=True)
        ts_len[ts_len < 1e-8] = 1e-8
        ts_norm = ts_norm / ts_len

        # If high-poly has its own normal map, blend using Reoriented Normal Mapping (RNM)
        if high_normal_tex is not None and has_high_uv:
            sampled_high_norm = sample_texture_bilinear(high_normal_tex, hit_uvs)
            n_tex = sampled_high_norm * 2.0 - 1.0
            # RNM blending: N_final = normalize([N_ts.x + n_tex.x, N_ts.y + n_tex.y, N_ts.z * n_tex.z])
            blended_x = ts_norm[:, 0] + n_tex[:, 0]
            blended_y = ts_norm[:, 1] + n_tex[:, 1]
            blended_z = ts_norm[:, 2] * np.clip(n_tex[:, 2], 0.01, 1.0)
            blended = np.stack([blended_x, blended_y, blended_z], axis=-1)
            b_len = np.linalg.norm(blended, axis=-1, keepdims=True)
            b_len[b_len < 1e-8] = 1e-8
            ts_norm = blended / b_len

        # Apply DirectX format if requested (invert Green/Y channel)
        if "DirectX" in normal_format or "Y-" in normal_format:
            ts_norm[:, 1] = -ts_norm[:, 1]

        # Convert to [0, 1] RGB
        norm_rgb = ts_norm * 0.5 + 0.5
        out_normal[valid_indices] = norm_rgb

    # 6. Sample PBR Textures
    if has_high_uv:
        if bake_diffuse and high_diffuse_tex is not None:
            out_diffuse[valid_indices] = sample_texture_bilinear(high_diffuse_tex, hit_uvs)
        elif bake_diffuse:
            # Fallback to white/base color
            out_diffuse[valid_indices] = [0.8, 0.8, 0.8]

        if bake_roughness and high_roughness_tex is not None:
            out_roughness[valid_indices] = sample_texture_bilinear(high_roughness_tex, hit_uvs)

        if bake_metallic and high_metallic_tex is not None:
            out_metallic[valid_indices] = sample_texture_bilinear(high_metallic_tex, hit_uvs)

    # 7. Compute Height / Displacement
    if bake_height:
        delta_p = hit_points - p_low
        dist_signed = np.sum(delta_p * n_low, axis=-1, keepdims=True) # Signed distance
        # Normalize around 0.5 midpoint
        height_val = np.clip((dist_signed / (2.0 * ray_max_dist)) + 0.5, 0.0, 1.0)
        out_height[valid_indices] = height_val

    # 8. Compute Ambient Occlusion / Cavity
    if bake_ao:
        # Distance and curvature based cavity shading
        delta_p = hit_points - p_low
        dist_len = np.linalg.norm(delta_p, axis=-1, keepdims=True)
        dot_n = np.sum(n_low * hit_normals, axis=-1, keepdims=True)
        ao_val = np.clip(dot_n * (1.0 - np.clip(dist_len / ray_max_dist, 0.0, 0.5)), 0.0, 1.0)
        out_ao[valid_indices] = ao_val

    # 9. Apply Fast GPU Seam Dilation
    print(f"[HighToLow Baker] Stage 4/5: Running GPU seam dilation & inpainting...")
    sys.stdout.flush()

    device_th = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    mask_th = torch.from_numpy(dilated_mask).to(device_th)

    def dilate_map(arr, custom_mask=None):
        m = custom_mask if custom_mask is not None else mask_th
        t = torch.from_numpy(arr).to(device_th)
        t_dilated = fast_gpu_seam_dilate(t, m, iterations=16)
        return t_dilated.cpu().numpy()

    # For diffuse, filter out unprojected dark pixels (< 0.02) from the high-poly source
    diff_valid = dilated_mask & (~(out_diffuse[..., :3] < 0.02).all(axis=-1))
    diff_mask_th = torch.from_numpy(diff_valid).to(device_th)

    out_diffuse = dilate_map(out_diffuse, custom_mask=diff_mask_th)
    out_normal = dilate_map(out_normal)
    out_roughness = dilate_map(out_roughness)
    out_metallic = dilate_map(out_metallic)
    if bake_height: out_height = dilate_map(out_height)
    if bake_ao: out_ao = dilate_map(out_ao)

    print(f"[HighToLow Baker] Stage 5/5: Textures baked and ready for export.")
    sys.stdout.flush()

    return {
        "diffuse": out_diffuse,
        "normal": out_normal,
        "roughness": out_roughness,
        "metallic": out_metallic,
        "height": out_height,
        "ao": out_ao
    }
