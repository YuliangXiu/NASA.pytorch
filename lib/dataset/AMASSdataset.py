import os, sys

import numpy as np
from PIL import Image
import trimesh
import glob
import torch
from collections import defaultdict

# amass related libs
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.visualization_tools import imagearray2file
from human_body_prior.body_model.body_model import BodyModel

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .hoppeMesh import HoppeMesh
from .sample import sample_surface
from .mesh_util import obj_loader

import matplotlib.pyplot as plt

def projection(points, calib):
    return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]

class AMASSdataset(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, opt, split, num_betas=16):

        self.opt = opt
        self.num_betas = num_betas
        self.num_poses = 21
        self.bm_path = os.path.dirname(os.path.abspath(__file__)) + \
                                '/amass/body_models/smplh/%s/model.npz'
        self.ds = {}
        for data_fname in glob.glob(os.path.join(
                        self.opt.root, split, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

    def __len__(self):

       return len(self.ds['trans'])

    def __getitem__(self, idx):
        data_dict =  {k: self.ds[k][idx] for k in self.ds.keys()}
        data_dict['root_orient'] = data_dict['pose'][:3]
        data_dict['pose_body'] = data_dict['pose'][3:66]
        data_dict['pose_hand'] = data_dict['pose'][66:]
        data_dict['betas'] = data_dict['betas'][:self.num_betas]

        self.num_poses = data_dict['pose_body'].shape[0] // 3

        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(self.get_sampling_geo(data_dict))

        # data_dict['A'] [22, 4, 4] bone transform matrix wrt root
        # [21, 4, 4] bone(w/o root) transform matrix
        B_inv = data_dict['A'][1:].inverse()
        # root location [4,]
        T_0 = data_dict['A'][0,:,-1]
        # sample points [N, 1, 4, 1]
        X = torch.cat((data_dict['samples_geo'], 
            torch.ones(self.opt.num_sample_geo, 1)),dim=1)[:, None, :, None] 

        # [21, 4, 4] x [N, 1, 4, 1] - > [N, 21, 4, 1] -- > [N, 21, 3]
        data_BX = torch.matmul(B_inv, X)[:, :, :3, 0] 

        # root location [1, 1, 4, 1] -- > [N, 1, 4, 1]
        # [21, 4, 4] x [N, 1, 4, 1] - > [N, 21, 4, 1] -- > [N, 21, 3]
        data_BT = torch.matmul(B_inv, T_0[None, None,:,None].repeat(
            self.opt.num_sample_geo, 1, 1, 1))[:, :, :3, 0] 

        # [N, ]
        targets = data_dict['labels_geo']

        return {'B_inv': B_inv,
                'T_0': T_0,
                'data_BX':data_BX, 
                'data_BT':data_BT, 
                'targets':targets}

    
    def load_mesh(self, data_dict):

        gender_type = "male" if data_dict["gender"] == -1 else "female"

        with torch.no_grad():
            bm = BodyModel(bm_path=self.bm_path%(gender_type), num_betas=self.num_betas, batch_size=1)
            body = bm.forward(pose_body=data_dict['pose_body'].unsqueeze(0), 
                            betas=data_dict['betas'].unsqueeze(0))

        mesh_ori = trimesh.Trimesh(vertices=c2c(body.v)[0], faces=c2c(body.f))

        # move the mesh to the original
        joints = c2c(body.Jtr)[0]
        root_xyz = joints[0]
        mesh_ori.vertices -= root_xyz

        verts = mesh_ori.vertices
        vert_normals = mesh_ori.vertex_normals
        face_normals = mesh_ori.face_normals
        faces = mesh_ori.faces

        mesh = HoppeMesh(
            verts=verts, 
            faces=faces, 
            vert_normals=vert_normals, 
            face_normals=face_normals)

        return {'mesh': mesh, 'A': body.A[0,:self.num_poses+1]}

    def get_sampling_geo(self, data_dict):
        mesh = data_dict['mesh']

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        samples_surface, face_index = sample_surface(
            mesh.triangles(), n_samples_surface)
        offset = np.random.normal(
            scale=self.opt.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.face_normals[face_index] * offset
        
        # Uniform samples in [-1, 1]
        b_min = np.array([-1.0, -1.0, -1.0])
        b_max = np.array([ 1.0,  1.0,  1.0])
        n_samples_space = self.opt.num_sample_geo // 4
        samples_space = np.random.rand(n_samples_space, 3) * (b_max - b_min) + b_min
        
        # total sampled points
        samples = np.concatenate([samples_surface, samples_space], 0)
        np.random.shuffle(samples)

        # labels: in->1.0; out->0.0.
        inside = mesh.contains(samples)

        # balance in and out
        inside_samples = samples[inside > 0.5]
        outside_samples = samples[inside <= 0.5]

        nin = inside_samples.shape[0]
        if nin > self.opt.num_sample_geo // 2:
            inside_samples = inside_samples[:self.opt.num_sample_geo // 2]
            outside_samples = outside_samples[:self.opt.num_sample_geo // 2]
        else:
            outside_samples = outside_samples[:(self.opt.num_sample_geo - nin)]
            
        samples = np.concatenate([inside_samples, outside_samples], 0)
        labels = np.concatenate([
            np.ones(inside_samples.shape[0]), np.zeros(outside_samples.shape[0])])

        return {
            'samples_geo': torch.Tensor(samples), 
            'labels_geo': torch.Tensor(labels),
        }


    def visualize_sampling3D(self, data_dict):
        import vtkplotter
        samples = data_dict[f'samples_geo']
        labels = data_dict[f'labels_geo']
        colors = np.stack([labels, labels, labels], axis=1)
     
        mesh = data_dict['mesh']
        faces = mesh.faces

        # [-1, 1]
        points = samples
        verts = mesh.verts
        
        # create plot
        vp = vtkplotter.Plotter(title="", size=(1500, 1500))
        vis_list = []

        # create a mesh
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 255]
        vis_list.append(mesh)

        # create a pointcloud
        pc = vtkplotter.Points(points, r=12, c=np.float32(colors))
        vis_list.append(pc)
        
        vp.show(*vis_list, bg="white", axes=1, interactive=True)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--num_sample_geo', type=int, default=5000)
    parser.add_argument('--sigma_geo', type=float, default=0.05)
    args = parser.parse_args()
        
    dataset = AMASSdataset(args, split='vald')
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # bdata = next(iter(dataloader))

    # print(bdata['B_inv'].shape)

    # if args.num_sample_geo:
    #     dataset.visualize_sampling(data_dict, '../test_data/proj_geo.jpg', mode='geo')
    # if args.num_sample_color:
    #     dataset.visualize_sampling(data_dict, '../test_data/proj_color.jpg', mode='color')

    # dataset.visualize_sampling3D(data_dict, mode='color')
    # dataset.visualize_sampling3D(data_dict, mode='geo')

    # speed 3.30 iter/s
    # with tinyobj loader 5.27 iter/s
    import tqdm
    for data_dict in tqdm.tqdm(dataset):
        data_dict["mesh"].export("mesh.obj")
        break
