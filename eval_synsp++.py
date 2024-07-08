import torch
# sys.path.append('/home/wt/docker/SmoothNet')
print()
from lib.dataset import find_dataset_using_name
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import DataSearch, prepare_output_dir, worker_init_fn
from lib.core.evaluate_config import parse_args
from model.Models_inf import Transformer

def main(cfg):
    train_datasets=[]
    test_datasets=[]

    all_estimator=cfg.ESTIMATOR.split(",")
    all_body_representation=cfg.BODY_REPRESENTATION.split(",")
    all_dataset=cfg.DATASET_NAME.split(",")
    dataset_names = [all_dataset[i] + '_' + all_estimator[i] + "_" + all_body_representation[i] for i in range(len(all_dataset))]
    cfg.DATASET_NAME = all_dataset

    for training_dataset_index in range(len(all_dataset)):
        estimator=all_estimator[training_dataset_index]
        body_representation=all_body_representation[training_dataset_index]
        dataset=all_dataset[training_dataset_index]

        dataset_class = find_dataset_using_name(dataset)

        print("Loading dataset ("+str(training_dataset_index)+")......")
        # if training_dataset_index > 0:
        train_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='train',
                                    down_dim = cfg.DOWN_DIM))

        test_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='test',
                                    down_dim = cfg.DOWN_DIM))
        
    for id, dataset in enumerate(train_datasets):
        dataset.add_datasets(train_datasets)
        dataset.id = id
    
    for id, dataset in enumerate(test_datasets):
        dataset.add_datasets(train_datasets)
        dataset.id = id

    train_loader=[]
    test_loader=[]
    data_search = DataSearch(dataset=test_datasets[0], dataset_names=dataset_names)


    for test_dataset in test_datasets:
        test_loader.append(DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                collate_fn=data_search.collate_fn_test,
                                worker_init_fn=worker_init_fn))
       
    # train_loader = RandomIter(train_loader, crop=True)

    # eval and train with single dataset
    idd = 0
    test_loader = [test_loader[idd]]

    # # ========= Compile Loss ========= #
    print(cfg.EXP_NAME)

    # # ========= Initialize networks ========= #
    dim = 4 
    if cfg.BODY_REPRESENTATION == "2D": dim = 2
    elif cfg.BODY_REPRESENTATION == "3D": dim = 3

    md = cfg.MODEL
    model = Transformer(d_word_vec=md.d_word_vec, n_position=2*md.d_model, d_model=md.d_model, d_inner=md.d_inner,
            n_layers=md.n_layers, n_head=md.n_head, d_k=md.d_k, d_v=md.d_v, dim=dim,
            coord=test_dataset.input_dimension, persons=cfg.MODEL.persons, device=cfg.DEVICE).to(cfg.DEVICE)

    print("loading model...")
    checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
    model.load_state_dict(checkpoint["state_dict"])
    
    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg, check=-1)
    evaluator.flg = 2
    # evaluator.run()
    for i in [0, 1, 2]:
        evaluator.test_dataloader[0].collate_fn.__self__.index = i
        performance = evaluator.run()
        print(performance)
    
if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg.comment = 'eval'
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)











