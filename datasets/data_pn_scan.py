"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import collections
import h5py
import numpy as np
import os
import pickle
from pointnet2_ops import pointnet2_utils
from scipy.linalg import expm, norm
import torch
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pdb


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data[:,:, :3].contiguous(), number) 
    fps_data = torch.gather(data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


def load_scanobjectnn_data(split, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if split == 1:
        DATA_DIR = BASE_DIR + '/../data/h5_files/main_split/'
        h5_name = DATA_DIR + partition + '_objectdataset.h5'
    elif split == 2:
        DATA_DIR = BASE_DIR + '/../data/h5_files/main_split_nobg/'
        h5_name = DATA_DIR + partition + '_objectdataset.h5'
    elif split == 3:
        DATA_DIR = BASE_DIR + '/../data/h5_files/main_split/'
        h5_name = DATA_DIR + partition + '_objectdataset_augmentedrot_scale75.h5'

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()

    if partition == 'test':
        precomputed_path = os.path.join(DATA_DIR, f'{partition}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
        if not os.path.exists(precomputed_path):
            data = torch.from_numpy(data).to(torch.float32).cuda()
            data = fps(data, 1024).cpu().numpy()
            with open(precomputed_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(precomputed_path, 'rb') as f:
                data = pickle.load(f)
    return data, label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):  
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not torch.is_tensor(data[key]):
                data[key] = torch.from_numpy(np.array(data[key]))
        return data


class PointCloudScaling(object):
    def __init__(self, 
                 scale=[2. / 3, 3. / 2], 
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 symmetries=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.from_numpy(np.array(symmetries))
        
    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        symmetries = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.symmetries = self.symmetries.to(device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False, 
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - torch.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim+1]
                data['heights'] = height - torch.min(height)
            
            if self.centering:
                data['pos'] = data['pos'] - torch.mean(data['pos'], axis=0, keepdims=True)
            
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - torch.mean(data, axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data = data / m
        return data


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()
        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data


class ScanObjectNN(Dataset):
    def __init__(self, num_points=2048, split=3, partition='training'):
        self.data, self.label = load_scanobjectnn_data(split, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            # pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud, 'y': label}

        if self.partition == 'training':
            data = PointsToTensor()(data)
            data = PointCloudScaling(scale=[0.9, 1.1])(data)
            data = PointCloudCenterAndNormalize(gravity_dim=1)(data)
            data = PointCloudRotation(angle=[0.0, 1.0, 0.0])(data)
        elif self.partition == 'test':
            data = PointsToTensor()(data)
            data = PointCloudCenterAndNormalize(gravity_dim=1)(data)
        
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = torch.cat((data['pos'], torch.from_numpy(pointcloud[:, 1:2] - pointcloud[:, 1:2].min())), dim=1)
        return data['x'], data['pos'], label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
