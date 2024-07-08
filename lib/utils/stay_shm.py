from lib.utils.milvus_pose import MilvusPose
import numpy as np
from multiprocessing import shared_memory, resource_tracker
from tqdm import tqdm
import os
import time
import  pickle
from lib.utils.geometry_utils import *

dataset_names = ['h36m_fcn_3D', 'pw3d_pare_3D', 'mocap_noise_3D']
# dataset_names = ['aist_vibe_3D'] # npz数据仅仅是源数据，id2sea才需要辨别dataset_names, 在get_id2seas的参数中调整
# train_path = [['/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/groundtruth/h36m_gt_3D_train.npz',
#                '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/detected/h36m_fcn_3D_train.npz'],
#                ['/mnt/new_disk/wangtao/SmoothNet/data/poses/pw3d_pare_3D/groundtruth/pw3d_gt_3D_train.npz',
#                 '/mnt/new_disk/wangtao/SmoothNet/data/poses/pw3d_pare_3D/detected/pw3d_pare_3D_train.npz']]
aist_vibe_3D_train_path = ['aist_vibe_3D', # name
              '/mnt/new_disk/wangtao/SmoothNet/data/poses/aist_vibe_3D/groundtruth/aist_gt_3D_train.npz', # gt
              '/mnt/new_disk/wangtao/SmoothNet/data/poses/aist_vibe_3D/detected/aist_vibe_3D_train.npz' # pred
              ]
h36m_hourglass_2D_path = ['h36m_hourglass_2D',
                          '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_hourglass_2D/groundtruth/h36m_gt_2D_train.npz',
                          '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_hourglass_2D/detected/h36m_hourglass_2D_train.npz']

aist_spin_smpl_path = ['aist_spin_smpl',
                       '/mnt/new_disk/wangtao/SmoothNet/data/poses/aist_spin_smpl/groundtruth/aist_gt_smpl_train.npz',
                       '/mnt/new_disk/wangtao/SmoothNet/data/poses/aist_spin_smpl/detected/aist_spin_smpl_train.npz']
h36m_fcn_3D_path = ['h36m_fcn_3D',
                    '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/groundtruth/h36m_gt_3D_train.npz',
                '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/detected/h36m_fcn_3D_train.npz']
pw3d_pare_3D_path = ['pw3d_pare_3D',
                    '/mnt/new_disk/wangtao/SmoothNet/data/poses/pw3d_pare_3D/groundtruth/pw3d_gt_3D_train.npz',
                '/mnt/new_disk/wangtao/SmoothNet/data/poses/pw3d_pare_3D/detected/pw3d_pare_3D_train.npz']
h36m_ppt_3D_path = ['h36m_ppt_3D',
                    '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_ppt_3D/groundtruth/h36m_gt_3D_train.npz',
                '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_ppt_3D/detected/h36m_ppt_3D_train.npz']
bin_path = "/dev/shm"

def get_id2seas(dnamme=['aist_vibe_3D']):
    mps = MilvusPose(dataset_names=dnamme)
    id2seas = mps.search_load()
    # os.makedirs(bin_path, exist_ok=True)

    for id, id2sea in tqdm(enumerate(id2seas)):
        data = np.array(id2sea)
        shape = data.shape
        dtype = data.dtype
        shared_mem_name = f"id2sea_{dnamme[id]}"
        if os.path.exists(os.path.join(bin_path, shared_mem_name)):
            os.remove(os.path.join(bin_path, shared_mem_name))
        shared_mem = shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize), name=shared_mem_name)
        shm_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
        shm_data[:] = data[:]
        print(shared_mem._name)
        resource_tracker.unregister(shared_mem._name, 'shared_memory')
        with open(os.path.join(bin_path, f"{shared_mem_name}.pkl"), "wb") as f:
            pickle.dump([shape, dtype], f)

def get_train_dataset(train_path: list): # h36m aist
    name = train_path[0]
    for j in range(1,3):
        phase = ['gt', 'pred'][j-1]
        path = train_path[j]
        ori_data = np.load(path, allow_pickle=True)
        if "joints_2d" in ori_data.files:
            ori_data = np.load(path, allow_pickle=True)["joints_2d"]
        elif "joints_3d" in ori_data.files:
            ori_data = np.load(path, allow_pickle=True)["joints_3d"]
        elif "pose" in ori_data.files and name == 'aist_spin_smpl':
            ori_data = np.load(path, allow_pickle=True)["pose"]
            for i in tqdm(range(len(ori_data))):
            # for i in tqdm(range(self.frame_num)):
                ori_data[i] = numpy_axis_to_rot6D(
                    ori_data[i].reshape(-1, 3)).reshape(-1, 24*6)
        else:
            raise ValueError("No joints_2d or joints_3d in npz file")
        data_len = [len(i) for i in ori_data]
        if train_path[0] == 'h36m_hourglass_2D':
            print('hit')
            H36M_IMG_SHAPE = 1000
            ori_data = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in ori_data]
        data = np.concatenate(ori_data, axis=0).astype(np.float32)

        shape = data.shape
        dtype = data.dtype
        shared_mem_name = f"dataset_{name}_{phase}"
        if os.path.exists(os.path.join(bin_path, shared_mem_name)): # delete it
            os.remove(os.path.join(bin_path, shared_mem_name))
        print(shape)
        print(dtype)
        shared_mem = shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize), name=shared_mem_name)
        shm_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
        shm_data[:] = data[:]
        resource_tracker.unregister(shared_mem._name, 'shared_memory')
        with open(os.path.join(bin_path, f"{shared_mem_name}.pkl"), "wb") as f:
            pickle.dump([shape, dtype, data_len], f)

def get_train_dataset_mocap(): # mocap
    path = '/mnt/new_disk/wangtao/SmoothNet/data/poses/mocap_noise_3D/groundtruth/train.npy'
    data = -np.load(path, allow_pickle=True)[:,0]/100
    
    shape = data.shape
    dtype = data.dtype
    shared_mem_name = f"dataset_mocap_gt"
    shared_mem = shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize), name=shared_mem_name)
    shm_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
    shm_data[:] = data[:]
    resource_tracker.unregister(shared_mem._name, 'shared_memory')
    with open(os.path.join(bin_path, f"{shared_mem_name}.pkl"), "wb") as f:
        pickle.dump([shape, dtype], f)

def get_train_dataset_nh36m(): # mocap
    path = '/mnt/new_disk/wangtao/SmoothNet/data/poses/nh36m_noise_3D/groundtruth/nh36m_gt_3D_train.npy'
    data = np.load(path, allow_pickle=True)
    proj_std_ratio = 115
    proj_data_ratio = 4/2000 * proj_std_ratio
    ori_data = [i.reshape(i.shape[0],-1)/proj_data_ratio for i in data]
    data_len = [len(i) for i in ori_data]
    data = np.concatenate(ori_data, axis=0).astype(np.float32)


    shape = data.shape
    dtype = data.dtype
    print(shape)
    print(dtype)
    shared_mem_name = f"dataset_nh36m_gt"
    if os.path.exists(os.path.join(bin_path, shared_mem_name)): # delete it
        os.remove(os.path.join(bin_path, shared_mem_name)) 
    shared_mem = shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize), name=shared_mem_name)
    shm_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
    shm_data[:] = data[:]
    resource_tracker.unregister(shared_mem._name, 'shared_memory')

    with open(os.path.join(bin_path, f"{shared_mem_name}.pkl"), "wb") as f:
        pickle.dump([shape, dtype, data_len], f)


print('------loading dataset1------')
ppath = aist_vibe_3D_train_path
get_train_dataset(ppath)
print('------loading id2seas------')
get_id2seas([ppath[0]])

# mocap
# print('------loading dataset1------')
# # get_train_dataset_mocap()
# print('------loading id2seas------')
# get_id2seas(['mocap_noise_3D'])

# nh36m
# print('------loading dataset1------')
# get_train_dataset_nh36m()
# print('------loading id2seas------')
# get_id2seas(['nh36m_noise_3D'])
print('------loading done------')

while True:
    time.sleep(1)
import pdb;pdb.set_trace()