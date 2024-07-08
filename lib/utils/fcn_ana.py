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
aist_gt_3D_train_path = ['aist_vibe_3D', # name
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
                    '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/groundtruth/h36m_gt_3D_test.npz',
                '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/detected/h36m_fcn_3D_train.npz',
                '/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D/detected/h36m_fcn_3D_test.npz']
bin_path = "/dev/shm"

def get_train_dataset(train_path: str): # h36m aist
    path = train_path
    ori_data = np.load(path, allow_pickle=True)["joints_3d"]
    data_len = [len(i) for i in ori_data]
    H36M_IMG_SHAPE = 1000
    ori_data = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in ori_data]
    data = np.concatenate(ori_data, axis=0).astype(np.float32).reshape(-1,17,3)
    return data
# print('------loading dataset1------')
ppath = h36m_fcn_3D_path
gtrain_data = get_train_dataset(ppath[1])
gtest_data = get_train_dataset(ppath[2])
dtrain_data = get_train_dataset(ppath[1])
dtest_data = get_train_dataset(ppath[2])
all_data = [gtrain_data, gtest_data, dtrain_data, dtest_data]
print(1)
# print('------loading id2seas------')
# get_id2seas([ppath[0]])


