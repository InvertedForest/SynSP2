import os
import logging
from os import path as osp
import time
import yaml
import numpy as np
import torch
import gc
from einops import repeat
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, db
from pymilvus.client.types import LoadState
# from .milvus_pose import MilvusPose


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    # logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    # logdir = f'{cfg.EXP_NAME}_{logtime}'
    logcomment = cfg.comment
    logdir = f'{cfg.EXP_NAME}_{logcomment}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    
    dir_num=0
    logdir_tmp=logdir

    while os.path.exists(logdir_tmp):
        logdir_tmp = logdir + str(dir_num)
        dir_num+=1
    
    logdir=logdir_tmp
    
    os.makedirs(logdir, exist_ok=True)
    #shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg

class DataSearch:
    def __init__(self,dataset, dataset_names) -> None:
        # milvus_pose = MilvusPose(dataset_names=dataset_names)
        # self.db_name = milvus_pose.db_name
        # self.col_name = milvus_pose.dataset_name
        # self.search_params = {
        #     "metric_type": "L2", 
        #     "params": {"range_filter": 0.8, "radius": 10000000}
        #     }
        self.slide_window_size = 8
        self.if_connected = False
        # train load
        self.dataset = dataset
        self.index = dataset.cfg.SEARCH.INDEX

    def init_milvus(self):
        connections.connect("default", host="localhost", port="19530", db_name=self.db_name)
        self.collection = Collection(self.col_name)
        self.collection.load()
        while utility.load_state(self.col_name) != LoadState.Loaded:
            time.sleep(0.1)
        self.if_connected = True

    def collate_fn(self, batch):
        gt = []
        pred = []
        final_res = {'gt': None, 'pred': None}

        for data in batch:
            gt.append(data['gt'])
            search_data = self.loaded_train_data[data['person_id']][data['frame_id']].reshape(-1, *data['gt'].shape) # (3, 8, 32)
            pred_data = np.concatenate([data['pred'][None],search_data], axis=0) # (4, 8, 32)
            pred.append(pred_data)

            
        gt = np.array(gt)
        pred = np.array(pred) # [1024,8,32]

        final_res['pred'] = torch.tensor(pred, dtype=torch.float32)
        final_res['gt'] = torch.tensor(gt, dtype=torch.float32)
        return final_res
    

    def _collate_fn(self, batch):
        gt = []
        pred = []
        final_res = {'gt': None, 'pred': None}

        for data in batch:
            gt.append(data['gt'])
            pred.append(data['pred'])
            
        gt = np.array(gt)
        pred = np.array(pred) # [1024,8,32]
        pred_flat = pred.reshape(pred.shape[0], -1)
        search_data = self.collection0.search(data=pred_flat, anns_field="norm_gt", limit=3, param=self.search_params,output_fields=["person_id","gt_id"])
        res = [[i[j].entity.get('gt_id') for j in range(3)] for i in search_data]
        res = np.array(res)
        # time1 = time.time()
        # print(time.time()-time1)
        res = res.reshape(res.shape[0],res.shape[1],self.slide_window_size,-1) # [1024,3,8,32]
        pred = np.concatenate([pred[:, None], res], axis=1)
        final_res['pred'] = torch.tensor(pred, dtype=torch.float32)
        final_res['gt'] = torch.tensor(gt, dtype=torch.float32)
        return final_res
    
    def unit_search(self, data, unit=5000, search_th=0):
        self.search_params['params']['range_filter'] = search_th
        len_data = len(data)
        unit_num = len_data // unit
        unit_res = len_data % unit
        search_datas = []
        if unit_num != 0:
            for i in range(unit_num):
                search_data = self.collection.search(data=data[i*unit : (i+1)*unit], anns_field="norm_gt", limit=1, param=self.search_params,output_fields=['dataset_id', 'all_id'])
                search_datas.append(search_data)
        if unit_res != 0:
            search_data = self.collection.search(data=data[len_data-unit_res : ], anns_field="norm_gt", limit=1, param=self.search_params,output_fields=['dataset_id', 'all_id'])
            search_datas.append(search_data)

        res = []
        for search_data in search_datas:
            for similar_pose in search_data:
                if len(similar_pose) == 0:
                    res.append([0,0,1])
                    print('loss entity')
                else:
                    similar_pose = similar_pose[0]
                    res.append([similar_pose.entity.get('dataset_id'), similar_pose.entity.get('all_id'), similar_pose.distance])
            # res += [[i[j].entity.get('all_id') for j in range(len(i))] for i in search_data]
        res = np.array(res) # (1024, 3)

        return res
    


    def collate_fn_test(self, batch):
        # if not self.if_connected:
        #     self.init_milvus()
        gt = batch[0]['gt']
        pred = batch[0]['pred']
        final_res = {'gt': None, 'pred': None, 'search': None}
        gt = np.array(gt)
        norm_pred = np.array(batch[0]['norm_pred']) # [1024,8,32]
        pred_flat = norm_pred.reshape(norm_pred.shape[0], -1) # # [1024,256]

        # res = self.unit_search(pred_flat, search_th=self.index) #[b, 3]
        # res_dir = f'searched/{self.index}'
        # res_path = res_dir + f'/{round(pred_flat[-1,0], 9)}.npy'
        # if not os.path.exists(res_dir):
        #     os.makedirs(res_dir, exist_ok=True)
        # if os.path.exists(res_path):
        #     raise ValueError(f'{res_path} exist')
        # np.save(res_path, res)
        
        res_dir = f'./data/poses/aist_vibe_3D/eval_search/{self.index}'
        res_path = res_dir + f'/{round(pred_flat[-1,0], 9)}.npy'
        res = np.load(res_path)
# ##########
#         import time;
#         time1 = time.time()
#         self.unit_search(pred_flat[0, None], search_th=self.index)
#         print(2)
#         print(time.time()-time1)
# ##############
        resu = []
        diss = []
        for id in res:
            if id[0] == -1:
                resu.append(np.zeros_like(pred[0]))
                diss.append(np.inf)
                print('!')
            else:
                dataset = self.dataset.all_datasets[int(id[0])]
                sea = dataset.get_data(int(id[1]))['gt']
                sea = sea.reshape(sea.shape[0], -1, self.dataset.coor_dim)
                # sea = sea[:,dataset.to14]
                sea = sea.reshape(sea.shape[0], -1)
                resu.append(sea)
                diss.append(id[2])
        res =  np.array(resu) # [1024,8,32]
        dis = np.array(diss) # [1024]
        # res = res.reshape(res.shape[0],res.shape[1],self.slide_window_size,-1) # [1024,3,8,32]
        final_res['pred'] = torch.tensor(pred, dtype=torch.float32)
        final_res['gt'] = torch.tensor(gt, dtype=torch.float32)
        final_res['search'] = torch.tensor(res, dtype=torch.float32)
        final_res['distance'] = torch.tensor(dis, dtype=torch.float32)
        if 'shape' in batch[0].keys():
            final_res['shape'] = torch.tensor(batch[0]['shape'], dtype=torch.float32)
        if 'trans' in batch[0].keys():
            final_res['trans'] = torch.tensor(batch[0]['trans'], dtype=torch.float32)
        if 'scale' in batch[0].keys():
            final_res['scale'] = torch.tensor(batch[0]['scale'], dtype=torch.float32)
        return final_res # should be aligned with lib/dataset/__init__.py get_train_data

def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def batch_time_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat

