from mmaction.apis import init_recognizer, inference_recognizer
import os
import time
import torch
from tqdm import tqdm
print(torch.__version__, torch.cuda.get_device_name())

def test_k400():
    config = 'configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py'
    checkpoint = 'data/kinetics400/models/swin_base_patch244_window877_kinetics400_22k.pth'
    model = init_recognizer(config, checkpoint, device=torch.device('cuda:0'))  # or 'cpu'
    label = 'demo/label_map_k400.txt'

    path = 'data/xd-violence/test'
    tests = os.listdir(path)
    for video in tests[:3]:
        if not video.endswith('mp4'):
            continue
        start = time.time()
        results = inference_recognizer(model, os.path.join(path, video), label)
        print(video, format(time.time()-start))
        for result in results:
            if result[1] < 0.1:
                continue
            print(f'{result[0]}: {result[1]}')

def set_cfg(cfg_file, chk_file):
    from mmcv import Config
    cfg = Config.fromfile(cfg_file)

    from mmcv.runner import set_random_seed

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)
    # Modify num classes of the model in cls_head
    cfg.model.cls_head.num_classes = 7
    # We can use the pre-trained TSN model
    cfg.load_from = chk_file

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.data.videos_per_gpu = cfg.data.videos_per_gpu # // 16
    cfg.optimizer.lr = cfg.optimizer.lr / 8 / 16
    cfg.total_epochs = 20

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 10
    # We can set the log print interval to reduce the the times of printing log
    cfg.log_config.interval = 200

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg
    

def train_xdv(arch, dsize):
    cfg = set_cfg(f'configs/recognition/swin/swin_{arch}_patch244_window877_xdviolence_k400_{dsize}.py', 
                 f'data/kinetics400/models/swin_{arch}_patch244_window877_kinetics400_{dsize}.pth')

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    import os.path as osp
    from mmaction.datasets import build_dataset
    from mmaction.models import build_model
    from mmaction.apis import train_model
    import mmcv

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_model(model, datasets, cfg, distributed=False, validate=True)
    

def test_xdv(arch, dsize, epoch=None):
    config_file = f'configs/recognition/swin/swin_{arch}_patch244_window877_kinetics400_{dsize}.py'
    path = 'data/xd-violence/test10'
    if epoch:
        checkpoint_file = f'work_dirs/xdviolence_swin_{arch}_patch244_window877.py/epoch_{epoch}.pth'
        label = 'demo/label_map_xdv.txt'
    else:
        checkpoint_file = f'data/kinetics400/models/swin_{arch}_patch244_window877_kinetics400_{dsize}.pth'
        label = 'demo/label_map_k400.txt'
        # path = 'data/kinetics400/val'
    model = init_recognizer(config_file, checkpoint_file, device=torch.device('cuda:0'))  # or 'cpu'

    tests = os.listdir(path)
    for video in tests:
        if not video.endswith('mp4'):
            continue
        start = time.time()
        results = inference_recognizer(model, os.path.join(path, video), label)
        print(f'({float(format(time.time()-start)):.1f}sec) {video.ljust(50)}', results)
            
def val_xdv(arch, dsize, epoch=None):
    cfg_file = f'configs/recognition/swin/swin_{arch}_patch244_window877_xdviolence_k400_{dsize}.py'
    checkpoint_file = f'work_dirs/xdviolence_swin_{arch}_patch244_window877_lr_8.py/epoch_{epoch}.pth'
    cfg = set_cfg(cfg_file, checkpoint_file)
    print('load_from: ', cfg.load_from)
    # model = init_recognizer(cfg_file, checkpoint_file, device=torch.device('cuda:0'))  # or 'cpu'
    # Build the recognizer
    from mmaction.models import build_model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    from mmcv.runner import load_checkpoint
    load_checkpoint(model, checkpoint_file)
        
    from mmaction.apis import single_gpu_test
    from mmaction.datasets import build_dataloader, build_dataset
    from mmcv.parallel import MMDataParallel

    # Build a test dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
            dataset,
            videos_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)
    print('\n', outputs)

    eval_config = cfg.evaluation
    eval_config.pop('interval')
    eval_res = dataset.evaluate(outputs, **eval_config)
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')            


# train_xdv('small', '1k')
# test_xdv('small', '1k')  # test 44 k400 test seq using k400 finetuned model
for e in range(10, 31, 10)[0:0]:
    test_xdv('small', '1k', e)
for e in range(10, 21, 10):
    val_xdv('small', '1k', e)
