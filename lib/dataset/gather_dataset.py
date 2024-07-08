from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *
from tqdm import tqdm
from lib.dataset.template_skeleton import *
from lib.dataset import find_dataset_using_name
from lib.utils.milvus_pose import MilvusPose

class GATHERDataset(BaseDataset):

    def __init__(self, cfg, estimator='all', return_type='3D', phase='train', down_dim=None):
        BaseDataset.__init__(self, cfg)

        self.dataset_name = "gather"
        self.all_name = '_'.join([self.dataset_name, estimator, return_type])
        self.all_dataset_names = ['h36m_fcn_3D', 'aist_vibe_3D', 'mocap_noise_3D']
        self.all_datasets = []
        for name in self.all_dataset_names:
            name_list = name.split('_')
            dataset_class = find_dataset_using_name(name_list[0])
            sub_dataset = dataset_class(cfg,
                                        estimator=name_list[1],
                                        return_type=name_list[2],
                                        phase='milvus')
            self.all_datasets.append(sub_dataset)


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
        
        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step=cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE
        self.step = [0] + [8*(i+1)-1 for i in range(self.slide_window_size//8)] #[0, 7, 15, 23, 31]
        self.return_type = return_type
        template_name = '_'.join([self.dataset_name, self.return_type]).upper()
        self.template_pose = torch.tensor(eval(template_name))
        self.coor_dim = 3
        self.sequence_num  = [dataset.sequence_num for dataset in self.all_datasets]
        data_len = [dataset.frame_num for dataset in self.all_datasets]
        self.data_len = [sum(data_len[:i]) for i in range(len(self.all_datasets))]
        self.frame_num = sum(self.data_len)

    def __len__(self):
        if self.phase == "train":
            return self.frame_num
        elif self.phase == "milvus":
            return self.frame_num
    
    def __getitem__(self, index):

        if self.phase == "train":
            return self.get_train_data(index)

        elif self.phase == "milvus":
            return self.get_data(index, norm=True)
        
    def get_data(self, index, norm=False):
        position = bisect.bisect(self.data_len, index)-1
        dataset = self.all_datasets[position]
        index = index - self.data_len[position]
        data = dataset.get_data(index, norm=False)
        data['dataset_id'] = position
        if not norm:
            return data
        else:
            norm_gt = self.norm_coor(data['gt'][None], dataset.to14)[0]
            norm_pred = self.norm_coor(data['pred'][None], dataset.to14)[0]
            data['norm_gt'] = norm_gt
            data['norm_pred'] = norm_pred
            return data

    def get_train_data(self, index):
        res = self.get_data(index, 1)
        gt_data = res['gt']
        pred_data = [res['pred']]
        search = self.loaded_train_data[index]
        for i in search:
            res = self.get_data(i, 1)
            pred_data.append(res['pred'])
        # randnn = 0.9+0.2*np.random.random()
        randnn = 1
        return {"gt": np.array(gt_data)*randnn, 
                "pred": np.array(pred_data)*randnn}