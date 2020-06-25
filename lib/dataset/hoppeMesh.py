import torch
import numpy as np
from scipy.spatial import cKDTree

from trimesh.triangles import points_to_barycentric
from trimesh.visual.color import uv_to_color

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

def save_ply(mesh_path, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param mesh_path: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(mesh_path,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\n'+
                          'property float x\nproperty float y\nproperty float z\n'+
                          'property uchar red\nproperty uchar green\nproperty uchar blue\n'+
                          'end_header').format(points.shape[0])
                      )

class HoppeMesh:
    def __init__(self, 
                 verts, faces, vert_normals, face_normals, 
                 uvs=None, face_uvs=None, texture=None,
                 ignore_vert_idxs=None, ignore_face_idxs=None):
        '''
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        :param points: pts
        :param normals: normals
        '''
        self.verts = verts #[n, 3]
        self.faces = faces  #[m, 3]
        self.vert_normals = vert_normals  #[n, 3]
        self.face_normals = face_normals  #[m, 3]
        self.uvs = uvs  #[n, 2]
        self.face_uvs = face_uvs #[m, 3, 2]
        self.vertex_colors = None #[n, 4] rgba
        self.ignore_vert_idxs = ignore_vert_idxs
        self.ignore_face_idxs = ignore_face_idxs
        if ignore_vert_idxs is None:
            self.kd_tree = cKDTree(self.verts)
        else:
            self.kd_tree = cKDTree(self.verts[~ignore_vert_idxs])
        self.len = len(self.verts)

        if (uvs is not None) and (texture is not None):
            self.vertex_colors = uv_to_color(uvs, texture)
        
    def query(self, points):
        dists, idx = self.kd_tree.query(points, n_jobs=1)
        # FIXME: because the eyebows are removed, cKDTree around eyebows
        # are not accurate. Cause a few false-inside labels here.
        if self.ignore_vert_idxs is None:
            dirs = points - self.verts[idx]
            signs = (dirs * self.vert_normals[idx]).sum(axis=1)
        else:
            dirs = points - self.verts[~self.ignore_vert_idxs][idx]
            signs = (dirs * self.vert_normals[~self.ignore_vert_idxs][idx]).sum(axis=1)
        signs = (signs > 0) * 2 - 1
        return signs * dists

    def contains(self, points):
        sdf = self.query(points)
        labels = sdf < 0 # in is 1.0, out is 0.0
        return labels

    def get_colors(self, points, faces):
        """
        Get colors of surface points from texture image through 
        barycentric interpolation.
        - points: [n, 3]
        - faces: [n, 3]
        - return: [n, 4] rgba
        """
        triangles = self.verts[faces] #[n, 3, 3]
        barycentric = points_to_barycentric(triangles, points) #[n, 3]
        vert_colors = self.vertex_colors[faces] #[n, 3, 4]
        point_colors = (barycentric[:, :, None] * vert_colors).sum(axis=1)
        return point_colors

    def export(self, path):
        if self.vertex_colors is not None:
            save_obj_mesh_with_color(
                path, self.verts, self.faces, self.vertex_colors[:, 0:3]/255.0)
        else:
            save_obj_mesh(
                path, self.verts, self.faces)

    def export_ply(self, path):
        save_ply(path, self.verts, self.vertex_colors[:, 0:3]/255.0)

    def triangles(self):
        return self.verts[self.faces] #[n, 3, 3]



if __name__ == '__main__':
    import trimesh
    from PIL import Image
    from sample import sample_surface

    # load
    mesh_ori = trimesh.load("../test_data/mesh.obj")
    verts = mesh_ori.vertices
    vert_normals = mesh_ori.vertex_normals
    face_normals = mesh_ori.face_normals
    faces = mesh_ori.faces
    uvs = mesh_ori.visual.uv 

    # samples, face_index = trimesh.sample.sample_surface(mesh_ori, 100000)

    # create
    mesh = HoppeMesh(
        verts, vert_normals, face_normals, faces, uvs, 
        texture=Image.open("../test_data/uv_render.png"))

    # export
    mesh.export("../test_data/test.obj")
    mesh.export_ply("../test_data/test.ply")

    samples, face_index = sample_surface(mesh.triangles(), 100000)

    sample_colors = mesh.get_colors(samples, faces[face_index])
    save_ply("../test_data/test_sample.ply", samples, sample_colors[:, 0:3]/255.0)