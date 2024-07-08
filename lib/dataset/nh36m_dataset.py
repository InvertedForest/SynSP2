from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *
from tqdm import tqdm
from lib.dataset.template_skeleton import *
from lib.utils.milvus_pose import MilvusPose
import pickle
from lib.dataset.shared_memory import SharedMemory
from copy import deepcopy

H36M_IMG_SHAPE=1000

class NH36MDataset(BaseDataset):

    def __init__(self, cfg, estimator='fcn', return_type='3D', phase='train', down_dim=None):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "nh36m"
        self.all_name = '_'.join([self.dataset_name, estimator, return_type])
        self.down_dim = down_dim
        self.std = cfg.TRAIN.noise_std
        self.noise_type = cfg.TRAIN.noise_type

        self.proj_std_ratio = 115
        self.proj_data_ratio = 4/2000 * self.proj_std_ratio
        self.std = self.std / self.proj_std_ratio

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

        if return_type in ['3D']:  
            self.return_type = return_type 
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'2D\',\'3D\',\'smpl\']. You can edit the code for additional implement"
            )

        if estimator in ['noise']:
            self.estimator = estimator  # 'fcn'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'noise\']. You can edit the code for additional implement"
            )
        self.file_phase = 'train' if self.phase != 'test' else 'test'
        print('#############################################################')
        print('You are loading the [' + self.file_phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + str(self.estimator) + ']')
        print('The type of the data is [' + self.return_type + ']')


        if return_type == '3D':
            self.coor_dim = 3
        else:
            raise Exception('Not implemented!')

        self.base_data_path = cfg.DATASET.BASE_DIR
        

        # save the fucking time
        if self.phase == 'train' or self.phase == 'milvus':
            self.cflag = True
            self.shm_gt_name = f'dataset_nh36m_gt'
            with open(f'/dev/shm/{self.shm_gt_name}.pkl', 'rb') as f:
                shape, dtype, data_len = pickle.load(f)
            self.shm_shape = shape
            self.shm_dtype = dtype
            self.ori_len = data_len
            self.start_num = [sum(self.ori_len[0:i]) for i in range(len(self.ori_len)+1)]
            self.data_len=[len-self.slide_window_size if (len-self.slide_window_size)>0 else 0 for len in data_len]
            self.data_start_num = [sum(self.data_len[0:i]) for i in range(len(self.data_len))]
            for i in range(len(self.data_start_num)-2,1):
                if self.data_start_num[i]==self.data_start_num[i-1]:
                    self.data_start_num[i]=self.data_start_num[i+1]
            self.frame_num = sum(self.data_len)
            self.input_dimension = shape[1]
            return

        try:
            ground_truth_data = np.load(os.path.join(
                self.base_data_path,
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.return_type + "_" + self.file_phase + ".npy"),
                                        allow_pickle=True)

        except:
            raise ImportError("Ground-truth data do not exist!")

        self.data_len = [len(seq)-self.slide_window_size if (len(seq)-self.slide_window_size)>0 else 0 for seq in ground_truth_data]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        # for i in range(len(self.data_start_num)-2,1):
        #     if self.data_start_num[i]==self.data_start_num[i-1]:
        #         self.data_start_num[i]=self.data_start_num[i+1]

        self.frame_num = sum(self.data_len)
        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(ground_truth_data)
        print('The sequence number is [' + str(self.sequence_num) + ']')

        
        print('#############################################################')

        self.ground_truth_data_joints_3d = [i.reshape(i.shape[0],-1)/self.proj_data_ratio for i in ground_truth_data] # (19832, 120, 14, 3) -> (19832, 120, 42)
        '''
        import matplotlib.pyplot as plt  
        data = ground_truth_data[0][0]
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

        
        
    # def noise(self, data, scale=0.01, mask_p=1): # (b, 8, 42)
    #     data = data.reshape(data.shape[0], data.shape[1], -1, 3) # (b, 8, 14, 3)
    #     dt = (data.max(axis=2) - data.min(axis=2))[:,:,None] # (b, 8, 1, 3)
    #     noise_dx = np.random.normal(0, (scale*dt), data.shape) # (b, 8, 14, 3)
    #     mask = np.random.uniform(0,1,data.shape) < mask_p  # (b, 8, 14, 3)
    #     noise_dx *= mask
    #     data = data + noise_dx
    #     data = data.reshape(data.shape[0], data.shape[1], -1) # (b, 8, 42)
    #     return data

    # def noise(self, data, scale=0.01, mask_p=1): # (b, 8, 42)
    #     data = data.reshape(data.shape[0], data.shape[1], -1, 3) # (b, 8, 14, 3)
    #     dt = (data.max(axis=2) - data.min(axis=2))[:,:,None] # (b, 8, 1, 3)
    #     noise_dx = np.random.normal(0, (scale*dt), data.shape) # (b, 8, 14, 3)
    #     mask = np.random.uniform(0,1,data.shape) < mask_p  # (b, 8, 14, 3)
    #     noise_dx *= mask
    #     data = data + noise_dx
    #     data = data.reshape(data.shape[0], data.shape[1], -1) # (b, 8, 42)
    #     return data
    def noise(self, data): # (b, 8, 42)
        if self.noise_type == "gaussian":
            noises = self.std * np.random.randn(*data.shape).astype(np.float32)
        elif self.noise_type == "uniform":
            noises = self.std * (np.random.rand(*data.shape).astype(np.float32) - 0.5)
        return data + noises

    def get_data(self, index, norm=False): # shm
        position = bisect.bisect(self.data_start_num, index)-1

        start_idxs = self.start_num[position]
        end_idxs = self.start_num[position+1]
        gt_shared_mem = SharedMemory(create=False, name=self.shm_gt_name)
        gt_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=gt_shared_mem.buf)
        gt_data = gt_data[start_idxs:end_idxs]

        ground_truth_data_len = self.ori_len[position]
        if self.slide_window_size <= ground_truth_data_len:

            start_idx = index - self.data_start_num[position]
            end_idx = start_idx + self.slide_window_size

            gt_data = deepcopy(gt_data[start_idx:end_idx, :])
            pred_data = self.noise(gt_data[None])[0]
            # if self.cflag:
                # self.cflag = False
        else:
            raise NotImplementedError

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