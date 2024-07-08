import numpy as np
import os
import bisect
# from lib.utils.geometry_utils import *


H36M_IMG_SHAPE=1000

class H36MDatasetGFPose():

    def __init__(self, cfg=None, estimator='fcn', return_type='3D', phase='test'):

        self.dataset_name = "h36m"

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
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

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + str(self.estimator) + ']')
        print('The type of the data is [' + str(self.return_type) + ']')

        self.slide_window_size = 1
        self.evaluate_slide_window_step = 1


        try:
            ground_truth_data = np.load(os.path.join('/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D',
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
            
            _ground_truth_data = dict()
            _ground_truth_data['imgname'] = [i for i in ground_truth_data['imgname']]
            _ground_truth_data['joints_3d'] = [i for i in ground_truth_data['joints_3d']]
            if self.phase == 'train':
                rate = 0.5
                test_ground_truth_data = np.load(os.path.join('/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D',
                    "groundtruth",
                    self.dataset_name + "_" + "gt"+"_"+self.return_type + "_test.npz"),
                                            allow_pickle=True)
                data_index = np.argwhere(np.random.uniform(0,1,len(test_ground_truth_data["imgname"])) < rate)
                for id in data_index:
                    _ground_truth_data['imgname'].append(test_ground_truth_data['imgname'][id][0])
                    _ground_truth_data['joints_3d'].append(test_ground_truth_data['joints_3d'][id][0])
            ground_truth_data = _ground_truth_data


        except:
            raise ImportError("Ground-truth data do not exist!")

        try:
            detected_data = np.load(os.path.join('/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D',
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.return_type + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
            
            _detected_data = dict()
            _detected_data['imgname'] = [i for i in detected_data['imgname']]
            _detected_data['joints_3d'] = [i for i in detected_data['joints_3d']]
            if self.phase == 'train':
                test_detected_data = np.load(os.path.join('/mnt/new_disk/wangtao/SmoothNet/data/poses/h36m_fcn_3D',
                    "detected",
                    self.dataset_name + "_" + self.estimator+"_"+self.return_type + "_test.npz"),
                                            allow_pickle=True)
                for id in data_index:
                    _detected_data['imgname'].append(test_detected_data['imgname'][id][0])
                    _detected_data['joints_3d'].append(test_detected_data['joints_3d'][id][0])
                detected_data = _detected_data
                
        except:
            raise ImportError("Detected data do not exist!")

        ground_truth_data_len = sum(
            len(seq) for seq in ground_truth_data["imgname"])
        detected_data_len = sum(len(seq) for seq in detected_data["imgname"])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        # self.data_len = [len(seq)-self.slide_window_size if (len(seq)-self.slide_window_size)>0 else 0 for seq in ground_truth_data["imgname"]]
        self.data_len = [len(seq)-self.slide_window_size+1 if (len(seq)-self.slide_window_size)>=0 else 0 for seq in ground_truth_data["imgname"]]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        for i in range(len(self.data_start_num)-2,1):
            if self.data_start_num[i]==self.data_start_num[i-1]:
                self.data_start_num[i]=self.data_start_num[i+1]

        # self.frame_num = ground_truth_data_len # dataset的长度在于有几个样本,很显然要减去8之后的才是
        self.frame_num = sum(self.data_len)
        # print('The frame number is [' + str(self.frame_num) + ']')
        print('The frame number is [' + str(ground_truth_data_len) + ']')

        self.sequence_num = len(ground_truth_data["imgname"])
        print('The sequence number is [' + str(self.sequence_num) + ']')
        
        print('#############################################################')

        if self.return_type == '3D':
            self.ground_truth_data_imgname = ground_truth_data["imgname"]
            self.ground_truth_data_joints_3d = ground_truth_data["joints_3d"]
            self.detected_data_imgname = detected_data["imgname"]
            self.detected_data_joints_3d = detected_data["joints_3d"]

            self.input_dimension = ground_truth_data["joints_3d"][0].shape[1]


    def __len__(self):
        if self.phase == "train":
            return self.frame_num
        elif self.phase == "test":
            return self.sequence_num

    def __getitem__(self, index):
        if self.phase == "train":
            return self.get_data(index)

        elif self.phase == "test":
            return self.get_test_data(index)
    #@profile
    def get_data(self, index):
        position = bisect.bisect(self.data_start_num, index)-1

        ground_truth_data_len = len(self.ground_truth_data_imgname[position])
        detected_data_len = len(self.detected_data_imgname[position])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[position]
            pred_data = self.detected_data_joints_3d[position]
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
        else: #能否去掉？
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
        assert gt_data.shape[0] == 8 or pred_data.shape[0] == 8, 'short for window'
        scale = 0.9+0.2*np.random.random()
        return {"gt": gt_data*scale, "pred": pred_data*scale}

    def get_test_data(self, index):
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
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
        return {"gt": gt_data, "pred": pred_data}
        