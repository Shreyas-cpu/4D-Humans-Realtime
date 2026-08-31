import numpy as np
import trimesh
import pyrender
from hmr2.utils.renderer import Renderer, create_raymond_lights

def render_side_custom(renderer, all_verts, all_cam_t, render_res, focal_length, rot_angle=45, mesh_base_color=(0.95, 0.55, 0.25)):
    r = pyrender.OffscreenRenderer(viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0)
    scene = pyrender.Scene(bg_color=[0.12, 0.14, 0.18, 0.0], ambient_light=(0.3, 0.3, 0.3))
    
    for i, (verts, cam_t) in enumerate(zip(all_verts, all_cam_t)):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * verts.shape[0])
        mesh = trimesh.Trimesh(verts.copy(), renderer.faces.copy(), vertex_colors=vertex_colors)
        # Rotate in place around Y axis
        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
        mesh.apply_transform(rot)
        # Apply camera translation
        mesh.vertices += cam_t.copy()
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(py_mesh, f"mesh_{i}")
        
    camera_pose = np.eye(4)
    camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1], zfar=1e12)
    cam_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(cam_node)
    
    renderer.add_point_lighting(scene, cam_node)
    renderer.add_lighting(scene, cam_node)
    for node in create_raymond_lights():
        scene.add_node(node)
        
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return (color.astype(np.float32) / 255.0)

print("Side render function defined successfully")
