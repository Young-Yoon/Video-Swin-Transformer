from mmaction.apis import init_recognizer, inference_recognizer
import os
import time

config_file = '../configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py'
checkpoint_file = '../checkpoints/swin_small_patch244_window877_kinetics400_1k.pth'
model = init_recognizer(config_file, checkpoint_file, device='cpu')
label = 'label_map_k400.txt'

path = '../data'
tests = os.listdir('../data')
for video in tests:
    if not video.endswith('mp4'):
        continue
    start = time.time()
    results = inference_recognizer(model, os.path.join(path, video), label)
    print(video, format(time.time()-start))
    for result in results:
        if result[1] < 0.1:
            continue
        print(f'{result[0]}: ', result[1])