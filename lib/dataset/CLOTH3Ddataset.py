import os, sys

sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '.')

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

from lib.dataset.cloth3d.prepare_data import outfit_types, fabric_types
from lib.dataset.cloth3d.read import DataReader
from lib.dataset.cloth3d.IO import quads2tris

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lib.dataset.hoppeMesh import HoppeMesh
from lib.dataset.sample import sample_surface
from lib.dataset.mesh_util import obj_loader
from lib.common.config import get_cfg_defaults

import matplotlib.pyplot as plt

def projection(points, calib):
    return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]

class CLOTH3Ddataset(Dataset):

    def __init__(self, opt, split, num_betas=10):

        self.opt = opt.dataset
        self.overfit = opt.overfit
        self.num_betas = num_betas
        self.num_poses = 24
        self.neighbors = 3
        self.bm_path = os.path.dirname(os.path.abspath(__file__)) + \
                                '/amass/body_models/smpl/model_%s.pkl'
        self.ds = {}
        for data_fname in glob.glob(os.path.join(
                        self.opt.root, split, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

        self.reader = DataReader()
        self.reader.SRC = os.path.join(self.opt.root, "raw")

        self.samples_body = None
        self.samples_cloth = None

    def __len__(self):

       return len(self.ds['trans'])

    def __getitem__(self, idx):
        if self.overfit:
            idx = 0 # for overfitting
        data_dict =  {k: self.ds[k][idx] for k in self.ds.keys()}
        data_dict['root_orient'] = data_dict['pose'][:3]
        data_dict['pose_body'] = data_dict['pose'][3:-6]
        data_dict['pose_hand'] = data_dict['pose'][-6:]
        data_dict['betas'] = data_dict['betas'][:self.num_betas]

        self.num_poses = data_dict['pose_body'].shape[0] // 3

        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(self.load_cloth(data_dict))
        data_dict.update(self.get_sampling_geo(data_dict))
        data_dict.update(self.get_sampling_cloth(data_dict))

        # data_dict['A'] [22, 4, 4] bone transform matrix wrt root
        # [21, 4, 4] bone(w/o root) transform matrix
        B_inv = data_dict['A'][1:].inverse()

        # root location [4,]
        T_0 = data_dict['A'][0,:,-1]

        # sample points [N, 1, 4, 1]
        X = torch.cat((data_dict['samples_body'], 
            torch.ones(self.opt.num_sample_geo+self.opt.num_verts, 1)),dim=1)[:, None, :, None] 

        # [21, 4, 4] x [N, 1, 4, 1] - > [N, 21, 4, 1] -- > [N, 21, 3]
        data_BX = torch.matmul(B_inv, X)[:, :, :3, 0] 

        # root location [1, 1, 4, 1] -- > [N, 1, 4, 1]
        # [21, 4, 4] x [N, 1, 4, 1] - > [N, 21, 4, 1] -- > [N, 21, 3]
        data_BT = torch.matmul(B_inv, T_0[None, None,:,None].repeat(
            self.opt.num_sample_geo+self.opt.num_verts, 1, 1, 1))[:, :, :3, 0] 

        # [num_verts, 21]
        weights = data_dict['samples_verts_weights']

        values, indices = weights.topk(self.neighbors, 
                                            dim=1, largest=True, sorted=True)
        weights_fill = torch.zeros_like(weights).scatter_(1, indices, values)
        weights_min = weights_fill.min(dim=1, keepdim=True)[0]
        weights_max = weights_fill.max(dim=1, keepdim=True)[0]
        weights_norm = (weights_fill - weights_min) / weights_max * 0.5

        return {'B_inv': B_inv,
                'T_0': T_0,

                'verts_body': data_dict['verts_body'],
                'faces_body': data_dict['faces_body'],

                'verts_cloth': data_dict['verts_cloth'], 
                'faces_cloth': data_dict['faces_cloth'],

                'samples_verts_body': data_dict['samples_verts_body'],
                'samples_verts_cloth': data_dict['samples_verts_cloth'],
                'weights': weights_norm,

                'samples_body': data_dict['samples_body'],
                'targets_body': data_dict['labels_body'],

                'samples_cloth': data_dict['samples_cloth'],
                'targets_cloth': data_dict['labels_cloth'],

                'joints': data_dict['joints'],
                'pose_mat': data_dict['pose_matrot'],
                'A': data_dict['A'],

                'data_BX':data_BX, 
                'data_BT':data_BT}

    def load_cloth(self, data_dict):

        # clothes
        garments_lst = list(np.nonzero(c2c(data_dict['outfit']))[0])
        garments = [outfit_types[idx] for idx in garments_lst]

        frame_id = c2c(data_dict['frame']).item()
        sample_id = "%05d"%(c2c(data_dict['idx']).item())

        V = None
        F = None

        for garment in garments:
            V_ = self.reader.read_garment_vertices(sample_id, garment, frame_id)
            F_ = quads2tris(self.reader.read_garment_topology(sample_id, garment))

            if (V is None) and (F is None):
                V = V_
                F = F_
            else:
                V = np.concatenate((V, V_),0)
                F = np.concatenate((F,F_),0)

        mesh_ori = trimesh.Trimesh(vertices=V, 
                                    faces=F, 
                                    process=False)

        verts = mesh_ori.vertices
        vert_normals = mesh_ori.vertex_normals
        face_normals = mesh_ori.face_normals
        faces = mesh_ori.faces

        mesh = HoppeMesh(
            verts=verts, 
            faces=faces, 
            vert_normals=vert_normals, 
            face_normals=face_normals)


        return {'verts_cloth': V,
                'faces_cloth': F,
                'mesh_cloth': mesh}


    
    def load_mesh(self, data_dict):

        gender_type = "m" if data_dict["gender"] == 1 else "f"

        with torch.no_grad():
            bm = BodyModel(bm_path=self.bm_path%(gender_type), num_betas=self.num_betas, batch_size=1)
            body = bm.forward(root_orient=data_dict['root_orient'].unsqueeze(0), 
                                pose_body=data_dict['pose_body'].unsqueeze(0), 
                                pose_hand=data_dict['pose_hand'].unsqueeze(0), 
                                betas=data_dict['betas'].unsqueeze(0), 
                                trans=data_dict['trans'].unsqueeze(0))

        # move the mesh to the original
        joints = c2c(body.Jtr)[0]
        root_xyz = joints[0].copy()
        joints -= root_xyz
       
        mesh_ori = trimesh.Trimesh(vertices=c2c(body.v)[0]-root_xyz, 
                                faces=c2c(body.f), 
                                process=False)

        verts = mesh_ori.vertices
        vert_normals = mesh_ori.vertex_normals
        face_normals = mesh_ori.face_normals
        faces = mesh_ori.faces

        mesh = HoppeMesh(
            verts=verts, 
            faces=faces, 
            vert_normals=vert_normals, 
            face_normals=face_normals)

        return {'verts_body': verts,
                'faces_body': faces, 
                'mesh_body': mesh,
                'joints': joints[1:self.num_poses+1],
                'A': body.A[0,:self.num_poses+1],
                'weights': body.weights[:,1:]}

    def get_sampling_geo(self, data_dict):
        mesh = data_dict['mesh_body']
        weights = data_dict['weights'] #[6890, 52]

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        samples_surface, face_index = sample_surface(
            mesh.triangles(), n_samples_surface)
        offset = np.random.normal(
            scale=self.opt.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.face_normals[face_index] * offset

        samples_verts_idx = np.random.choice(np.arange(mesh.verts.shape[0]),
                            self.opt.num_verts, replace=False)
        samples_verts = mesh.verts[samples_verts_idx, :]
        samples_verts_weights = weights[samples_verts_idx, :]
                
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

        # add original verts
        samples = np.concatenate([samples, samples_verts], 0)
        labels = np.concatenate([labels, 0.5 * np.ones(self.opt.num_verts)])

        return {
            'samples_body': torch.Tensor(samples), 
            'samples_verts_body': torch.Tensor(samples_verts),
            'samples_verts_weights': torch.Tensor(samples_verts_weights),
            'labels_body': torch.Tensor(labels),
        }

    def get_sampling_cloth(self, data_dict):

        mesh = data_dict['mesh_cloth']

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        samples_surface, face_index = sample_surface(
            mesh.triangles(), n_samples_surface)
        offset = np.random.normal(
            scale=self.opt.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.face_normals[face_index] * offset

        samples_verts_idx = np.random.choice(np.arange(mesh.verts.shape[0]),
                            self.opt.num_verts, replace=False)
        samples_verts = mesh.verts[samples_verts_idx, :]
                
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

        # add original verts
        samples = np.concatenate([samples, samples_verts], 0)
        labels = np.concatenate([labels, 0.5 * np.ones(self.opt.num_verts)])

        return {
            'samples_cloth': torch.Tensor(samples), 
            'samples_verts_cloth': torch.Tensor(samples_verts),
            'labels_cloth': torch.Tensor(labels),
        }


    def visualize_sampling3D(self, data_dict, only_pc=False):
        import vtkplotter
        from matplotlib import cm

        cmap_body = cm.get_cmap('jet')
        cmap_cloth = cm.get_cmap('Paired')

        samples_body = c2c(data_dict['samples_body'])
        labels_body = c2c(data_dict['targets_body'])
        colors_body = cmap_body(labels_body/labels_body.max())[:,:3]

        samples_cloth = c2c(data_dict['samples_cloth'])
        labels_cloth = c2c(data_dict['targets_cloth'])
        colors_cloth = cmap_cloth(labels_cloth/labels_cloth.max())[:,:3]
        
        # create plot
        vp = vtkplotter.Plotter(title="", size=(1500, 1500))
        vis_list = []

        if not only_pc:

            # create a mesh
            body = trimesh.Trimesh(
                    data_dict['verts_body'], 
                    data_dict['faces_body'])
            body.visual.face_colors = [200, 200, 0, 255]
            vis_list.append(body)

            if 'verts_cloth' in data_dict.keys() and \
                'faces_cloth' in data_dict.keys():
                cloth = trimesh.Trimesh(
                    data_dict['verts_cloth'],
                    data_dict['faces_cloth'])

                cloth.visual.face_colors = [0, 200, 0, 255]
                vis_list.append(cloth)
            
        # # create a pointcloud
        # pc = vtkplotter.Points(np.concatenate((samples_body, samples_cloth),0), 
        #                         r=12, 
        #                         c=np.float32(np.concatenate((colors_body, colors_cloth),0)))
        # # pc = vtkplotter.Points(samples_body, 
        # #                         r=12, 
        # #                         c=np.float32(colors_body))
        # vis_list.append(pc)
        
        vp.show(*vis_list, bg="white", axes=1, interactive=True)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', type=str, help='path of the yaml config file')
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)

    # opts = sys.argv[sys.argv.index('--') + 1:]

    # default cfg: defined in 'lib.common.config.py'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    # Now override from a list (opts could come from the command line)
    # opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
    # cfg.merge_from_list(opts)
    cfg.freeze()

    dataset = CLOTH3Ddataset(cfg, split='vald')

    # speed 3.30 iter/s
    # with tinyobj loader 5.27 iter/s
    import tqdm

    for data_dict in tqdm.tqdm(dataset):

        dataset.visualize_sampling3D(data_dict, only_pc=False)

        break
