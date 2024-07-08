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

class AISTDataset(BaseDataset):

    def __init__(self, cfg, estimator='spin', return_type='3D', phase='train', down_dim=None):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "aist"
        self.all_name = '_'.join([self.dataset_name, estimator, return_type])
        self.down_dim = down_dim

        if phase == 'train':
            self.phase = phase  # 'train' | 'test' | 'validate'
        elif phase == 'test':
            self.phase = phase
        elif phase == 'milvus':
            self.phase = phase
        elif phase == 'validate':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\',\'validate\']. You can edit the code for additional implements"
            )

        if return_type in ['3D', 'smpl']:  # no 2D
            self.return_type = return_type  # '3D'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\','smpl']. You can edit the code for additional implement"
            )

        if estimator in ['spin','vibe','tcmr']:
            self.estimator = estimator  # 'spin'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'spin\',\'vibe\']. You can edit the code for additional implement"
            )
        self.file_phase = 'train' if self.phase != 'test' else 'test'
        print('#############################################################')
        print('You are loading the [' + self.file_phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + self.estimator + ']')
        print('The type of the data is [' + self.return_type + ']')

        self.coor_dim = 3
        if self.return_type == 'smpl' and cfg.TRAIN.USE_6D_SMPL:
            self.coor_dim = 6

        self.base_data_path = cfg.DATASET.BASE_DIR
        self.to14 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

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
        self.sequence_num = len(ground_truth_data["imgname"])
        # self.frame_num = 20
        # self.sequence_num = self.frame_num

        print('The frame number is [' + str(self.frame_num) + ']')
        print('The sequence number is [' + str(self.sequence_num) + ']')
        print('#############################################################')

        if self.return_type == '3D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.detected_data_imgname = detected_data["imgname"]

            da = ground_truth_data["joints_3d"]
            da = [i.reshape(i.shape[0],-1,3) for i in da]
            self.ground_truth_data_joints_3d = [i.reshape(i.shape[0],-1) for i in da]


            da = detected_data["joints_3d"]
            da = [i.reshape(i.shape[0],-1, 3) for i in da]
            self.detected_data_joints_3d = [i.reshape(i.shape[0],-1) for i in da]

            self.input_dimension = ground_truth_data["joints_3d"][0].shape[-1]



        elif self.return_type == 'smpl':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_pose = ground_truth_data["pose"]
            self.ground_truth_data_trans = ground_truth_data["trans"]
            self.ground_truth_data_scaling = ground_truth_data["scaling"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_pose = detected_data["pose"]
            self.detected_data_shape = detected_data["shape"]

            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
                self.coor_dim = 6
                for i in tqdm(range(len(self.ground_truth_data_pose))):
                # for i in tqdm(range(self.frame_num)):
                    self.ground_truth_data_pose[i] = numpy_axis_to_rot6D(
                        self.ground_truth_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)

                for i in tqdm(range(len(self.detected_data_pose))):
                # for i in tqdm(range(self.frame_num)):
                    self.detected_data_pose[i] = numpy_axis_to_rot6D(
                        self.detected_data_pose[i].reshape(-1, 3)).reshape(
                            -1, self.input_dimension)
            else:
                self.input_dimension = 3 * 24




    def get_data(self, index, norm=False): # shm
        position = bisect.bisect(self.data_start_num, index)-1

        if self.return_type == '3D':
            start_idxs = self.start_num[position]
            end_idxs = self.start_num[position+1]
            gt_shared_mem = SharedMemory(create=False, name=self.shm_gt_name)
            pred_shared_mem = SharedMemory(create=False, name=self.shm_pred_name)
            gt_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=gt_shared_mem.buf)
            pred_data = np.ndarray(shape=self.shm_shape, dtype=self.shm_dtype, buffer=pred_shared_mem.buf)
            gt_data = gt_data[start_idxs:end_idxs]
            pred_data = pred_data[start_idxs:end_idxs]
        else:
            raise NotImplementedError

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
    

    


    def get_test_data(self, index, norm=True):
        # index = index + 1287
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise NotImplementedError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index]
            pred_data = self.detected_data_joints_3d[index]

        elif self.return_type == 'smpl':
            gt_data = self.ground_truth_data_pose[index].reshape(
                ground_truth_data_len, -1)
            pred_data = self.detected_data_pose[index].reshape(
                ground_truth_data_len, -1)

            gt_trans = self.ground_truth_data_trans[index].reshape(
                ground_truth_data_len, -1)
            gt_scaling = self.ground_truth_data_scaling[index].reshape(
                ground_truth_data_len, -1)

            pred_shape = self.detected_data_shape[index].reshape(
                ground_truth_data_len, -1)

            # gt_data = np.concatenate((gt_data, gt_trans, gt_scaling), axis=-1)
            # pred_data = np.concatenate((pred_data, pred_shape), axis=-1)

        if self.slide_window_size <= ground_truth_data_len:
            start_idx=np.arange(0,ground_truth_data_len-self.slide_window_size+1,self.evaluate_slide_window_step)
            gt_data_=[]
            pred_data_=[]
            pred_shape_=[]
            gt_trans_=[]
            gt_scaling_=[]
            for idx in start_idx:
                gt_data_.append(gt_data[idx:idx+self.slide_window_size,:])
                pred_data_.append(pred_data[idx:idx+self.slide_window_size,:])
                if self.return_type == 'smpl':
                    pred_shape_.append(pred_shape[idx:idx+self.slide_window_size,:])
                    gt_trans_.append(gt_trans[idx:idx+self.slide_window_size,:])
                    gt_scaling_.append(gt_scaling[idx:idx+self.slide_window_size,:])

            gt_data=np.array(gt_data_)
            pred_data=np.array(pred_data_)
            pred_shape=np.array(pred_shape_)
            gt_trans=np.array(gt_trans_)
            gt_scaling=np.array(gt_scaling_)
        else:
            print('loss')
            print(index)
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
            pred_shape=[]
            gt_trans=[]
            gt_scaling=[]
        if not norm:
            return {"gt": gt_data, 
                    "pred": pred_data,
                    "shape": pred_shape,
                    "trans": gt_trans,
                    "scale": gt_scaling}
        norm_gt = self.norm_coor(gt_data)
        norm_pred = self.norm_coor(pred_data)
        return {"gt": gt_data, 
                "norm_gt": norm_gt, 
                "pred": pred_data,
                "norm_pred": norm_pred,
                "shape": pred_shape,
                "trans": gt_trans,
                "scale": gt_scaling}
