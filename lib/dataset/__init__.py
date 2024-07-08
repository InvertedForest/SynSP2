import importlib
import numpy as np
from abc import ABC, abstractmethod
from lib.dataset.template_skeleton import *
import torch.utils.data as data
import torch
import pickle
from lib.utils.utils2 import batch_compute_similarity_transform_torch
from lib.dataset.shared_memory import SharedMemory

from copy import deepcopy

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, cfg):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.cfg = cfg
        self.id = 0
        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step=cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE
        self.step = [0] + [8*(i+1)-1 for i in range(self.slide_window_size//8)] #[0, 7, 15, 23, 31]
        self.check = -1
        self.down_dim = cfg.DOWN_DIM

    def add_datasets(self, all_datasets):
        print(f'add dataset {self.all_name} to all datasets')
        self.infor_id2sea = []
        self.all_datasets = all_datasets
        for dataset in all_datasets:
            shm_name = f'id2sea_{dataset.all_name}'
            with open(f'/dev/shm/{shm_name}.pkl', 'rb') as f:
                shape, dtype = pickle.load(f)
                self.infor_id2sea.append([shm_name, shape, dtype])

    def set_check(self, check):
        self.check = check


    def __len__(self):
        if self.check != -1:
            return self.check
        if self.phase == "train":
            return self.frame_num
        elif self.phase == "milvus":
            return self.frame_num
        elif self.phase == "test":
            return self.sequence_num

    # @abstractmethod
    # def __getitem__(self, index):
    #     """Return a data point and its metadata information.
    #     Parameters:
    #         index - - a random integer for data indexing
    #     Returns:
    #         a dictionary of data with their names. It ususally contains the data itself and its metadata information.
    #     """
    #     pass
    def __getitem__(self, index):

        if self.phase == "train":
            return self.get_train_data(index)

        elif self.phase == "milvus":
            return self.get_data(index, norm=True)
        
        elif self.phase == "test":
            return self.get_test_data(index)


    def norm_coor(self, coor):
        coor = coor.reshape(coor.shape[0],coor.shape[1],-1,self.coor_dim)[:,self.step] # (b,4,16,c)
        if self.coor_dim == 6: return coor.reshape(coor.shape[0],coor.shape[1], -1)/2 # align distance
        if self.down_dim:
            down_dim_name = f'to{self.down_dim}'
            if down_dim_name in self.__dict__.keys():
                down_dim_list = eval(f'self.{down_dim_name}')
                self.template_pose = torch.tensor(eval(f'DOWN_{self.down_dim}'))
                coor = coor[:,:,down_dim_list]
            else:
                raise NotImplementedError
        else:
            self.template_pose = torch.tensor(eval(f'{self.dataset_name}_{self.return_type}'.upper()))
        ori_shp = coor.shape
        coor = coor.reshape(ori_shp[0]*ori_shp[1], ori_shp[2], ori_shp[3])
        template = self.template_pose.expand(coor.shape)
        coor = torch.tensor(coor)
        coor = batch_compute_similarity_transform_torch(coor, template)
        coor = coor.reshape(ori_shp)
        return coor # (b,4,16,c)


    # @profile
    def get_train_data(self, index):
        # id2seas = []
        # for infor in self.infor_id2sea:
        #     shm_name, shape, dtype = infor
        #     shared_mem = shared_memory.SharedMemory(create=False, name=shm_name);resource_tracker.unregister(shared_mem._name, 'shared_memory')
        #     gt_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
        #     id2seas.append(gt_data.copy())
            
        # search = id2seas[self.id][index]

        shm_name, shape, dtype = self.infor_id2sea[self.id]
        shared_mem = SharedMemory(create=False, name=shm_name)
        # shared_mem = shared_memory.SharedMemory(create=False, name=shm_name);resource_tracker.unregister(shared_mem._name, 'shared_memory')
        search_infor_data = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
        search = search_infor_data[index]
        
        res = self.get_data(index)
        search_infor_data = res['gt']
        pred_data = res['pred']
        
        # assert len(search) == 3
        search_dst = [int(i[0]) for i in search]
        search_index = [int(i[1]) for i in search]
        search_dis = [i[2] for i in search]
        
        # prevent overfitting
        top_k = 3
        assert top_k <= len(search), 'top_k should be less than search'
        random_index = np.random.randint(1,top_k*2+1)
        if random_index <= top_k:
            random_index = 0
        else:
            random_index = random_index - top_k
        search_dis = search_dis[random_index]
        
        res = self.all_datasets[search_dst[random_index]].get_data(search_index[random_index])
        search_gt = res['gt'].reshape(res['gt'].shape[0],-1, self.coor_dim)
        # search_gt = search_gt[:,self.all_datasets[search_dst[i]].to14,:]
        search_gt = search_gt.reshape(search_gt.shape[0],-1)
            
        # randnn = 0.9+0.2*np.random.random()
        randnn = 1
        return {"gt": np.array(search_infor_data, dtype=np.float32)*randnn, 
                "pred": np.array(pred_data, dtype=np.float32)*randnn,
                "search": np.array(search_gt, dtype=np.float32)*randnn,
                "distance": np.array(search_dis, dtype=np.float32)}


def find_dataset_using_name(dataset_name):
    """
    Import the module "lib/dataset/[dataset_name]_dataset.py".
    """
    dataset_filename = "lib.dataset." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name))

    return dataset