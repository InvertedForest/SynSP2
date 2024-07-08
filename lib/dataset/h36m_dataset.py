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

class H36MDataset(BaseDataset):

    def __init__(self, cfg, estimator='fcn', return_type='3D', phase='train', down_dim=None):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "h36m"
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

        if return_type in ['3D','smpl','2D']:  
            self.return_type = return_type 
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'2D\',\'3D\',\'smpl\']. You can edit the code for additional implement"
            )

        if estimator in ['fcn','vibe','tcmr','hourglass','cpn','hrnet','rle','videoposet27','videoposet81','videoposet243','ppt']:
            self.estimator = estimator  # 'fcn'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'fcn\',\'vibe\',\'tcmr\',\'hourglass\',\'cpn\',\'hrnet\',\'rle\',\'videoposet27\',\'videoposet81\',\'videoposet243\']. You can edit the code for additional implement"
            )
        self.file_phase = 'train' if self.phase != 'test' else 'test'
        print('#############################################################')
        print('You are loading the [' + self.file_phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + self.estimator + ']')
        print('The type of the data is [' + self.return_type + ']')


        if return_type == '2D':
            self.coor_dim = 2
        elif return_type == '3D':
            self.coor_dim = 3
        elif return_type == 'smpl':
            raise Exception('Not implemented!')

        self.base_data_path = cfg.DATASET.BASE_DIR
        self.to14 = [6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 8, 10]

        # save the fucking time
        if self.phase == 'train' or self.phase == 'milvus':
            self.cflag = True
            self.shm_gt_name = f'dataset_{self.all_name}_gt'
            self.shm_pred_name = f'dataset_{self.all_name}_pred'
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
                self.dataset_name + "_" + "gt"+"_"+self.return_type + "_" + self.file_phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise NotImplementedError("Ground-truth data do not exist!")

        try:
            detected_data = np.load(os.path.join(
                self.base_data_path, 
                self.dataset_name+"_"+self.estimator+"_"+self.return_type,
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.return_type + "_" + self.file_phase + ".npz"),
                                        allow_pickle=True)
        except:
            raise NotImplementedError("Detected data do not exist!")

        ground_truth_data_len = sum(
            len(seq) for seq in ground_truth_data["imgname"])
        detected_data_len = sum(len(seq) for seq in detected_data["imgname"])
        self.max_len = max([len(seq) for seq in detected_data["imgname"]])
        print(f'max_len: {self.max_len}')

        if ground_truth_data_len != detected_data_len:
            raise NotImplementedError(
                "Detected data is not the same size with ground_truth data!")

        self.data_len = [len(seq)-self.slide_window_size if (len(seq)-self.slide_window_size)>0 else 0 for seq in ground_truth_data["imgname"]]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        for i in range(len(self.data_start_num)-2,1):
            if self.data_start_num[i]==self.data_start_num[i-1]:
                self.data_start_num[i]=self.data_start_num[i+1]

        self.frame_num = sum(self.data_len)
        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(ground_truth_data["imgname"])
        print('The sequence number is [' + str(self.sequence_num) + ']')

        
        print('#############################################################')

        if self.return_type == '3D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_joints_3d = ground_truth_data["joints_3d"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_joints_3d = detected_data["joints_3d"]

            self.input_dimension = ground_truth_data["joints_3d"][0].shape[1]

        elif self.return_type == 'smpl':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_pose = ground_truth_data["pose"]
            self.ground_truth_data_shape = ground_truth_data["shape"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_pose = detected_data["pose"]
            self.detected_data_shape = detected_data["shape"]

            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
                for i in range(len(self.ground_truth_data_pose)):
                    self.ground_truth_data_pose[i] = numpy_axis_to_rot6D(
                        self.ground_truth_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)

                for i in range(len(self.detected_data_pose)):
                    self.detected_data_pose[i] = numpy_axis_to_rot6D(
                        self.detected_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)
            else:
                self.input_dimension = 3 * 24

        elif self.return_type == '2D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            da = ground_truth_data["joints_2d"]
            self.ground_truth_data_joints_2d = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in da]
            


            self.detected_data_imgname = detected_data["imgname"]
            da = detected_data["joints_2d"]
            self.detected_data_joints_2d = [(i-H36M_IMG_SHAPE/2)/(H36M_IMG_SHAPE/2) for i in da]

            self.input_dimension = ground_truth_data["joints_2d"][0].shape[1]


    def get_data(self, index, norm=False): # shm
        position = bisect.bisect(self.data_start_num, index)-1

        # start_idxs = sum(self.ori_len[0:position])
        # end_idxs = sum(self.ori_len[0:position+1])
        start_idxs = self.start_num[position]
        end_idxs = self.start_num[position+1]
        gt_shared_mem = SharedMemory(create=False, name=self.shm_gt_name)
        pred_shared_mem = SharedMemory(create=False, name=self.shm_pred_name)
        gt_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=gt_shared_mem.buf)
        pred_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=pred_shared_mem.buf)
        gt_data = gt_data[start_idxs:end_idxs]
        pred_data = pred_data[start_idxs:end_idxs]

        ground_truth_data_len = self.ori_len[position]
        if self.slide_window_size <= ground_truth_data_len:

            start_idx = index - self.data_start_num[position]
            end_idx = start_idx + self.slide_window_size

            gt_data = deepcopy(gt_data[start_idx:end_idx, :])
            pred_data = deepcopy(pred_data[start_idx:end_idx, :])
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
    

    def get_data_bak(self, index, norm=False):
        position = bisect.bisect(self.data_start_num, index)-1

        ground_truth_data_len = len(self.ground_truth_data_joints_3d[position])
        # ground_truth_data_len = len(self.ground_truth_data_imgname[position])
        # detected_data_len = len(self.detected_data_imgname[position])

        # if ground_truth_data_len != detected_data_len:
        #     raise NotImplementedError(
        #         "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[position]
            pred_data = self.detected_data_joints_3d[position]
            self.to14 = [6, 5, 4, 1, 2, 3, 13, 12, 11, 14, 15, 16, 8, 10]
        elif self.return_type == '2D':
            gt_data = self.ground_truth_data_joints_2d[position]
            pred_data = self.detected_data_joints_2d[position]

        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[position].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[position].reshape(
                ground_truth_data_len, -1)

        if self.slide_window_size <= ground_truth_data_len:

            start_idx = index - self.data_start_num[position]
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)

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
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])  

        if ground_truth_data_len != detected_data_len:
            raise NotImplementedError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index]
            pred_data = self.detected_data_joints_3d[index]
        elif self.return_type == '2D':
            gt_data = self.ground_truth_data_joints_2d[index]
            pred_data = self.detected_data_joints_2d[index]
        
        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[index].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[index].reshape(
                ground_truth_data_len, -1)

            gt_shape = self.ground_truth_data_shape[index].reshape(
                ground_truth_data_len, -1)
            pred_shape = self.detected_data_shape[index].reshape(
                ground_truth_data_len, -1)
            gt_data = np.concatenate((gt_data, gt_shape), axis=-1)
            pred_data = np.concatenate((pred_data, pred_shape), axis=-1)

        if self.slide_window_size <= ground_truth_data_len:
            start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
            gt_data_=[]
            pred_data_=[]
            for idx in start_idx:
                gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])
                pred_data_.append(pred_data[idx:idx+self.slide_window_size,:])

            gt_data=np.array(gt_data_)
            pred_data=np.array(pred_data_)

        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)[np.newaxis, :]
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)[np.newaxis, :]

        if not norm:
            return {"gt": gt_data, 
                    "pred": pred_data}
        norm_gt = self.norm_coor(gt_data)
        norm_pred = self.norm_coor(pred_data)
        return {"gt": gt_data, 
                "norm_gt": norm_gt, 
                "pred": pred_data,
                "norm_pred": norm_pred}