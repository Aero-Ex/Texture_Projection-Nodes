from typing import Optional, Union
import trimesh

def convert_to_whole_mesh(scene:Union[trimesh.Trimesh, trimesh.Scene]):
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        geometry = scene.dump()
        if len(geometry) == 1:
            mesh = geometry[0]
        else:
            mesh = trimesh.util.concatenate(geometry)
    else:
        raise ValueError(f"Unknown mesh type.")
    # mesh.merge_vertices(merge_tex=False, merge_norm=True)
    return mesh

def load_whole_mesh(mesh_path, limited_faces=None) -> trimesh.Trimesh:
    scene = trimesh.load(mesh_path, process=False)
    mesh = convert_to_whole_mesh(scene)
    return mesh
