"""Compute per-epoch mAP curve for PPLCNet_x1_0 100-epoch run (traffic sign classification).

- Builds PPLCNet_x1_0(class_num=232), no pretrained loading.
- For each epoch_N loads epoch_N.pdparams (raw) and epoch_N.pdema (EMA), runs
  inference on a fixed 20000-image subset of the test set, then computes
  macro-Average-Precision (mAP) using softmax confidence per class.
- Writes map_results.json incrementally and final map_curve.png.
"""
import os
import re
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import paddle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, accuracy_score
from PIL import Image

from ppcls.arch.backbone.legendary_models.pp_lcnet import PPLCNet_x1_0

# ---------------- config ----------------
OUT_DIR    = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/output/PPLCNet_x1_0_100ep'
DATA_ROOT  = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/dataset'
LABEL_FILE = os.path.join(DATA_ROOT, 'traffic_sign', 'label_list_test.txt')
IMAGE_ROOT = DATA_ROOT  # label paths are relative to ./dataset/
CLASS_NUM  = 232
NUM_SAMPLES = 20000
BATCH_SIZE  = 256
RESIZE_SHORT = 256
CROP_SIZE    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEED = 2026
NUM_SAMPLES = int(os.environ.get('MAP_NUM_SAMPLES', NUM_SAMPLES))
EPOCHS = list(range(1, 101))
if os.environ.get('MAP_EPOCHS'):
    EPOCHS = [int(e) for e in os.environ['MAP_EPOCHS'].split(',') if e.strip()]
if os.environ.get('MAP_MAX_EPOCH'):
    EPOCHS = list(range(1, int(os.environ['MAP_MAX_EPOCH']) + 1))

CACHE_FILE = os.path.join(OUT_DIR, f'map_inputs_{NUM_SAMPLES}_seed{SEED}.npz')
RESULTS_FILE = os.path.join(OUT_DIR, 'map_results.json')
PLOT_FILE    = os.path.join(OUT_DIR, 'map_curve.png')

paddle.disable_signal_handler()


def parse_label_file():
    paths, labels = [], []
    with open(LABEL_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            paths.append(parts[0])
            labels.append(int(parts[1]))
    return paths, np.array(labels, dtype=np.int64)


def resize_short_center_crop(img):
    w, h = img.size
    scale = RESIZE_SHORT / float(min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resample = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR',
                             getattr(Image, 'BILINEAR', 2))
    img = img.resize((nw, nh), resample)
    cx = (nw - CROP_SIZE) // 2
    cy = (nh - CROP_SIZE) // 2
    img = img.crop((cx, cy, cx + CROP_SIZE, cy + CROP_SIZE))
    return img


def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = resize_short_center_crop(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr


def build_inputs():
    if os.path.exists(CACHE_FILE):
        print(f'[cache] loading preprocessed inputs from {CACHE_FILE}')
        d = np.load(CACHE_FILE)
        return d['X'], d['y']
    print('[prep] parsing label file and sampling subset ...')
    paths, labels = parse_label_file()
    n = len(paths)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(n, size=NUM_SAMPLES, replace=False)
    idx.sort()
    print(f'[prep] total test images={n}, sampled={NUM_SAMPLES}')
    X = np.zeros((NUM_SAMPLES, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    y = np.zeros((NUM_SAMPLES,), dtype=np.int64)
    t0 = time.time()
    for i, gi in enumerate(idx):
        p = paths[gi]
        full = os.path.join(IMAGE_ROOT, p)
        try:
            X[i] = preprocess_image(full)
        except Exception as e:
            print(f'  [warn] fail to read {full}: {e}; filling zeros')
        y[i] = labels[gi]
        if (i + 1) % 200 == 0:
            print(f'  preprocessed {i+1}/{NUM_SAMPLES}  ({time.time()-t0:.1f}s)')
    print(f'[prep] done in {time.time()-t0:.1f}s. class coverage in subset: '
          f'{len(np.unique(y))} / {CLASS_NUM}')
    np.savez(CACHE_FILE, X=X, y=y)
    print(f'[cache] saved -> {CACHE_FILE}')
    return X, y


def forward_logits(model, X):
    preds = []
    n = X.shape[0]
    with paddle.no_grad():
        for s in range(0, n, BATCH_SIZE):
            batch = paddle.to_tensor(X[s:s + BATCH_SIZE], dtype='float32')
            out = model(batch)
            if isinstance(out, (list, tuple)):
                out = out[0]
            pred = paddle.nn.functional.softmax(out, axis=1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def compute_map(y, P):
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)  # one-hot over present classes
    ap_arr = average_precision_score(y_bin, P[:, classes], average=None)
    ap_arr = np.asarray(ap_arr, dtype=np.float64).reshape(-1)
    valid = np.asarray(y_bin.sum(axis=0) > 0).reshape(-1)
    mAP = float(np.mean(ap_arr[valid])) if valid.any() else float('nan')
    top1 = float(accuracy_score(y, np.argmax(P, axis=1)))
    return mAP, top1


def main():
    if not paddle.device.is_compiled_with_cuda():
        print('[warn] paddle not built with CUDA; using CPU (slow).')
        place = paddle.CPUPlace()
    else:
        place = paddle.CUDAPlace(0)
    paddle.set_device('gpu' if paddle.device.is_compiled_with_cuda() else 'cpu')

    X, y = build_inputs()

    model = PPLCNet_x1_0(pretrained=False, use_ssld=False, class_num=CLASS_NUM)
    model.eval()

    results = {ep: {} for ep in EPOCHS}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            try:
                prev = json.load(f)
                for ep, d in prev.items():
                    if int(ep) in results:
                        results[int(ep)].update(d)
            except Exception:
                pass

    overall_t0 = time.time()
    for ep in EPOCHS:
        if results[ep] and 'map_raw' in results[ep] and 'map_ema' in results[ep]:
            print(f'[ep {ep:>3}] already done (raw_mAP={results[ep]["map_raw"]:.4f}, '
                  f'ema_mAP={results[ep]["map_ema"]:.4f}) -- skip')
            continue
        t_ep = time.time()
        # ---- raw ----
        raw_path = os.path.join(OUT_DIR, f'epoch_{ep}.pdparams')
        if not os.path.exists(raw_path):
            print(f'[ep {ep:>3}] missing {raw_path}, skip')
            continue
        sd = paddle.load(raw_path)
        missing, unexpected = model.set_state_dict(sd)
        if missing:
            print(f'  [warn] missing keys (raw): {missing[:3]} ... total {len(missing)}')
        P_raw = forward_logits(model, X)
        map_raw, t1_raw = compute_map(y, P_raw)

        # ---- ema ----
        ema_path = os.path.join(OUT_DIR, f'epoch_{ep}.pdema')
        map_ema = None
        t1_ema = None
        if os.path.exists(ema_path):
            sd2 = paddle.load(ema_path)
            m2, u2 = model.set_state_dict(sd2)
            if m2:
                print(f'  [warn] missing keys (ema): {m2[:3]} ... total {len(m2)}')
            P_ema = forward_logits(model, X)
            map_ema, t1_ema = compute_map(y, P_ema)
        else:
            print(f'  [note] no EMA file {ema_path}')

        results[ep] = {
            'map_raw': map_raw, 'top1_raw': t1_raw,
            'map_ema': map_ema, 'top1_ema': t1_ema,
            'elapsed_s': round(time.time() - t_ep, 1),
        }
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=1)
        print(f'[ep {ep:>3}] raw mAP={map_raw:.4f} (top1={t1_raw:.4f})  '
              f'ema mAP={map_ema:.4f} (top1={t1_ema:.4f})  '
              f'[{time.time()-t_ep:.1f}s, total {time.time()-overall_t0:.0f}s]')

    plot(results)


def plot(results):
    eps = sorted([int(e) for e in results.keys()
                  if results[e].get('map_raw') is not None or
                     results[e].get('map_ema') is not None])
    raw_map = [results[e].get('map_raw') for e in eps]
    ema_map = [results[e].get('map_ema') for e in eps]
    raw_t1  = [results[e].get('top1_raw') for e in eps]
    ema_t1  = [results[e].get('top1_ema') for e in eps]

    raw_eps = [e for e, v in zip(eps, raw_map) if v is not None]
    raw_map = [v for v in raw_map if v is not None]
    ema_eps = [e for e, v in zip(eps, ema_map) if v is not None]
    ema_map = [v for v in ema_map if v is not None]
    raw_t1  = [v for v in raw_t1 if v is not None]
    ema_t1  = [v for v in ema_t1 if v is not None]

    if not eps:
        print('[plot] no results, skip plotting.')
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    if raw_eps:
        ax.plot(raw_eps, [v * 100 for v in raw_map], 'o-', color='C2', ms=4,
                label='mAP (raw model)')
    if ema_eps:
        ax.plot(ema_eps, [v * 100 for v in ema_map], 's--', color='C4', ms=4,
                label='mAP (EMA model)')
    if raw_t1 and raw_eps:
        ax.plot(raw_eps, [v * 100 for v in raw_t1], '.:', color='C3', alpha=0.5,
                label='Top-1 (raw)')
    if ema_t1 and ema_eps:
        ax.plot(ema_eps, [v * 100 for v in ema_t1], '.:', color='C0', alpha=0.5,
                label='Top-1 (EMA)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mAP / Top-1 Accuracy (%)')
    fig.suptitle(f'PPLCNet_x1_0 交通标志识别 逐 epoch mAP 曲线 (100 epoch, {NUM_SAMPLES} 张测试子集)',
                 fontsize=14, fontweight='bold')
    ax.set_title('Traffic Sign Classification - per-epoch mAP curve '
                 '(macro Average Precision, softmax-based)', fontsize=10,
                 style='italic', color='dimgray')
    all_v = [v * 100 for v in (raw_map + ema_map + raw_t1 + ema_t1)]
    if all_v:
        lo = min(all_v)
        ax.set_ylim(max(50, lo - 1), 100.2)
    ax.set_xlim(0, max(eps) + 2)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    last_ep = max(eps)
    if ema_map:
        ax.annotate(f'{ema_map[-1]*100:.2f}%', xy=(last_ep, ema_map[-1]*100),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=9, color='C4', fontweight='bold')
    if raw_map:
        ax.annotate(f'{raw_map[-1]*100:.2f}%', xy=(last_ep, raw_map[-1]*100),
                    xytext=(6, -10), textcoords='offset points',
                    fontsize=9, color='C2', fontweight='bold')
    # best markers
    if ema_map:
        bi = int(np.argmax(ema_map))
        ax.annotate(f'EMA best: {ema_map[bi]*100:.2f}% @ ep {ema_eps[bi]}',
                    xy=(ema_eps[bi], ema_map[bi]*100), fontsize=9, color='C4',
                    xytext=(-12, 18), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='C4', alpha=0.6))
    if raw_map:
        bi = int(np.argmax(raw_map))
        ax.annotate(f'raw best: {raw_map[bi]*100:.2f}% @ ep {raw_eps[bi]}',
                    xy=(raw_eps[bi], raw_map[bi]*100), fontsize=9, color='C2',
                    xytext=(-12, -22), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='C2', alpha=0.6))
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=140)
    plt.close(fig)
    print(f'[plot] saved -> {PLOT_FILE}')

    # also dump a summary csv
    csv_path = os.path.join(OUT_DIR, 'map_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,map_raw,top1_raw,map_ema,top1_ema\n')
        def fmt(v):
            return '' if v is None else f'{v:.6f}'
        for e in eps:
            d = results[e]
            f.write(f'{e},{fmt(d.get("map_raw"))},{fmt(d.get("top1_raw"))},'
                    f'{fmt(d.get("map_ema"))},{fmt(d.get("top1_ema"))}\n')
    print(f'[csv]  saved -> {csv_path}')


if __name__ == '__main__':
    main()