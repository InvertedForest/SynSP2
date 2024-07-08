from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *
from tqdm import tqdm
from lib.dataset.template_skeleton import *
from lib.utils.milvus_pose import MilvusPose
from lib.utils.geometry_utils import *
import pickle
from lib.dataset.shared_memory import SharedMemory
from copy import deepcopy

# MOCAP_TO_AIST =  [9,7,6,1,2,5,26,25,2,17,18,19,13,16]
MOCAP_TO_H36M =  [0,1,2,3,6,7,8,12,14,15,16,24,25,26,17,18,19]
class MOCAPDataset(BaseDataset):

    def __init__(self, cfg, estimator='spin', return_type='3D', phase='train', down_dim=None):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "mocap"
        self.all_name = '_'.join([self.dataset_name, estimator, return_type])
        self.down_dim = down_dim

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
            self.phase = phase
        elif phase == 'milvus':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\']. You can edit the code for additional implements"
            )

        if return_type in ['3D']:  # no 2D
            self.return_type = return_type  # '3D' | '2D' | 'smpl'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\','smpl']. You can edit the code for additional implement"
            )

        self.estimator = estimator
        self.std = cfg.TRAIN.noise_std
        self.noise_type = cfg.TRAIN.noise_type

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + self.estimator + ']')
        print('The type of the data is [' + self.return_type + ']')

        self.coor_dim = 3

        self.base_data_path = cfg.DATASET.BASE_DIR
        self.file_phase = 'train' if self.phase != 'test' else 'test'
        # self.to14 = [8,7,6,1,2,3,26,25,24,17,18,19,14,16]
        self.to14 = [6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 8, 10]
        
        self.proj_std_ratio = 115
        self.proj_data_ratio = 4/2000 * self.proj_std_ratio
        self.std = self.std / self.proj_std_ratio

        # save the fucking time
        if self.phase == 'train' or self.phase == 'milvus':
            self.cflag = True
            self.shm_gt_name = 'dataset_mocap_gt'
            with open(f'/dev/shm/{self.shm_gt_name}.pkl', 'rb') as f:
                shape, dtype = pickle.load(f)
            self.shm_shape = shape
            self.shm_dtype = dtype
            self.real_len = shape[1]-self.slide_window_size+1
            self.frame_num = shape[0] * self.real_len
            self.input_dimension = shape[2] * shape[3]

            
            return


        try:
            ground_truth_data = np.load(f'/mnt/new_disk/wangtao/SmoothNet/data/poses/mocap_noise_3D/groundtruth/{self.file_phase}.npy',
                                        allow_pickle=True)[:,0]/100 # one person
        except:
            raise NotImplementedError("Ground-truth data do not exist!")

        self.data_len = [len(seq)-self.slide_window_size+1 if (len(seq)-self.slide_window_size)>=0 else 0 for seq in ground_truth_data]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        # for i in range(len(self.data_start_num)-2,1):
        #     if self.data_start_num[i]==self.data_start_num[i-1]:
        #         self.data_start_num[i]=self.data_start_num[i+1]

        self.frame_num = sum(self.data_len)
        self.sequence_num = len(ground_truth_data)
        # self.frame_num = 20
        # self.sequence_num = self.frame_num

        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(ground_truth_data)
        print('The sequence number is [' + str(self.sequence_num) + ']')

        
        print('#############################################################')

        da=[-i for i in ground_truth_data] # (19832, 120, 31, 3) -> (19832, 120, 31, 3)
        self.ground_truth_data_joints_3d = [i.reshape(i.shape[0],-1) for i in da] # (19832, 120, 31, 3) -> (19832, 120, 93)

        '''
        import matplotlib.pyplot as plt  
        # data = ground_truth_data[0][0]
        fig = plt.figure()  
        ax = fig.add_subplot(111)
        
        # 画出数据点  
        ax.scatter(data[:, 0], data[:, 1])  
        
        # 在每个点的右上方标出索引  
        for i in range(len(data)):  
            ax.text(data[i, 0] + 0.01, data[i, 1] + 0.01, str(i), fontsize=10)  
        
        # 设置x轴和y轴的标签  
        ax.set_aspect('equal', adjustable='box')
        plt.savefig("test.png")
        plt.close()
        '''
        self.input_dimension = self.ground_truth_data_joints_3d[0].shape[-1]
        


        
    
    def noise(self, data): # (b, 8, 42)
        if self.noise_type == "gaussian":
            # noises = self.std * np.random.randn(*data.shape).astype(np.float32)
            noises = np.random.normal(scale=self.std, size=data.shape)
        elif self.noise_type == "uniform":
            noises = self.std * (np.random.rand(*data.shape).astype(np.float32) - 0.5)
        return data + noises

    
    def get_data(self, index, norm=False): # shm
        position = index // self.real_len

        if self.return_type == '3D':
            gt_shared_mem = SharedMemory(create=False, name=self.shm_gt_name)
            gt_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=gt_shared_mem.buf)[position]
        else:
            raise NotImplementedError


        start_idx = index - self.real_len * position
        end_idx = start_idx + self.slide_window_size

        gt_data = deepcopy(gt_data[start_idx:end_idx, :])
        gt_data = gt_data.reshape(gt_data.shape[0], -1)
        pred_data = self.noise(gt_data[None])[0]

        if not norm:
            return {"gt": gt_data, 
                    "pred": pred_data, 
                    "person_id": position, 
                    "frame_id": start_idx,
                    "all_id": index}
        else:
            norm_gt = self.norm_coor(gt_data[None])[0]
            norm_pred = self.norm_coor(pred_data[None])[0]
            return {"gt": gt_data, 
                    "norm_gt": norm_gt, 
                    "pred": pred_data, 
                    "norm_pred": norm_pred,
                    "person_id": position, 
                    "frame_id": start_idx,
                    "all_id": index}

    def get_test_data(self, index, norm=True):

        gt_data = self.ground_truth_data_joints_3d[index]
        ground_truth_data_len = len(gt_data)


        start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
        gt_data_=[]
        pred_data_=[]
        for idx in start_idx:
            gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])

        gt_data=np.array(gt_data_)
        np.random.seed(index)
        pred_data=self.noise(gt_data)
        
        if not norm:
            return {"gt": gt_data, 
                    "pred": pred_data}
        norm_gt = self.norm_coor(gt_data)
        norm_pred = self.norm_coor(pred_data)
        return {"gt": gt_data, 
                "norm_gt": norm_gt, 
                "pred": pred_data,
                "norm_pred": norm_pred}