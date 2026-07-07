import os
import numpy as np
import cv2
from paddle.inference import Config, create_predictor

MODEL_DIR = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/deploy/models/PPLCNet_x1_0_traffic_sign_infer'
IMG_PATH = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/deploy/images/PULC/traffic_sign/99603_17806.jpg'
LABEL_FILE = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/ppcls/utils/PULC_label_list/traffic_sign_label_list.txt'

# load labels
id2name = {}
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        id2name[int(parts[0])] = parts[1]

# build predictor with file,file signature (works on win paddle 2.6.1)
model_file = os.path.join(MODEL_DIR, 'inference.pdmodel')
params_file = os.path.join(MODEL_DIR, 'inference.pdiparams')
config = Config(model_file, params_file)
config.enable_use_gpu(8000, 0)
config.disable_glog_info()
config.switch_ir_optim(True)
config.enable_memory_optim()
predictor = create_predictor(config)

input_names = predictor.get_input_names()
print('input_names:', input_names)
input_handle = predictor.get_input_handle(input_names[0])
output_names = predictor.get_output_names()
print('output_names:', output_names)
output_handle = predictor.get_output_handle(output_names[0])

# preprocess: resize_short=256, center crop 224, normalize, CHW
img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]
scale = 256.0 / min(h, w)
img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
h, w = img.shape[:2]
top = (h - 224) // 2
left = (w - 224) // 2
img = img[top:top + 224, left:left + 224]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype('float32') / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype='float32')
std = np.array([0.229, 0.224, 0.225], dtype='float32')
img = (img - mean) / std
img = img.transpose(2, 0, 1)[None, ...]

input_handle.reshape([1, 3, 224, 224])
input_handle.copy_from_cpu(img)
predictor.run()
out = output_handle.copy_to_cpu()
print('raw out shape:', out.shape, 'min/max:', out.min(), out.max())
probs = out[0]

topk = 5
idx = np.argsort(-probs)[:topk]
print('Image:', os.path.basename(IMG_PATH))
print('Top-5 predictions:')
for i in idx:
    name = id2name.get(i, '?')
    print(f'  class_id={i:3d}  label={name:<6}  score={probs[i]:.4f}')