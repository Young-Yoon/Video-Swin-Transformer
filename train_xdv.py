from mmaction.apis import init_recognizer, inference_recognizer
import os
import time
import torch
from tqdm import tqdm
print(torch.__version__, torch.cuda.get_device_name())


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
        print(f'{result[0]}: ', result[1])

from mmcv import Config
cfg = Config.fromfile('configs/recognition/swin/swin_small_patch244_window877_xdviolence_k400_1k.py')

from mmcv.runner import set_random_seed

# The flag is used to determine whether it is omnisource training
cfg.setdefault('omnisource', False)
# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 7
# We can use the pre-trained TSN model
cfg.load_from = 'data/kinetics400/models/swin_small_patch244_window877_kinetics400_1k.pth'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.data.videos_per_gpu = cfg.data.videos_per_gpu # // 16
cfg.optimizer.lr = cfg.optimizer.lr / 8 # / 16
cfg.total_epochs = 30

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10
# We can set the log print interval to reduce the the times of printing log
cfg.log_config.interval = 200

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


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

