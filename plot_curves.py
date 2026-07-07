"""Parse PaddleClas train.log + VisualDL log to plot Loss & Acc curves (PNG)."""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/output/PPLCNet_x1_0_20ep'
LOG_TXT = os.path.join(OUT_DIR, 'train.log')

train_epoch_re = re.compile(r'\[Train\]\[Epoch (\d+)/\d+\]\[Avg\]top1: ([\d.]+), top5: ([\d.]+), CELoss: ([\d.]+)')
eval_avg_re    = re.compile(r'\[Eval\]\[Epoch (\d+)\]\[Avg\]CELoss: ([\d.]+),.*top1: ([\d.]+), top5: ([\d.]+)')
iter_loss_re   = re.compile(r'\[Train\]\[Epoch (\d+)/\d+\]\[Iter: (\d+)/\d+\]lr\(.*?\): ([\d.eE+-]+), top1: ([\d.]+), top5: ([\d.]+), CELoss: ([\d.]+)')

train_avg_top1, train_avg_loss = [], []
eval_raw_ep, eval_raw_top1 = [], []
eval_ema_ep, eval_ema_top1 = [], []
iter_epoch, iter_loss, iter_lr = [], [], []

with open(LOG_TXT, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

i = 0
n = len(lines)
while i < n:
    line = lines[i]
    m = train_epoch_re.search(line)
    if m:
        train_avg_top1.append(float(m.group(2)))
        train_avg_loss.append(float(m.group(4)))
        i += 1
        continue
    m = eval_avg_re.search(line)
    if m:
        ep = int(m.group(1)); top1 = float(m.group(3))
        j = i + 1
        nxt = ''
        while j < n and not nxt.strip():
            nxt = lines[j]; j += 1
        if 'ema' in nxt:
            eval_ema_ep.append(ep); eval_ema_top1.append(top1)
        else:
            eval_raw_ep.append(ep); eval_raw_top1.append(top1)
        i += 1
        continue
    m = iter_loss_re.search(line)
    if m:
        iter_epoch.append(int(m.group(1)))
        iter_loss.append(float(m.group(6)))
        iter_lr.append(float(m.group(3)))
        i += 1
        continue
    i += 1

print(f'parsed: train_avg epochs={len(train_avg_top1)}, '
      f'raw eval={len(eval_raw_top1)}, ema eval={len(eval_ema_top1)}, '
      f'iter rows={len(iter_loss)}')

# ---- plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('PULC traffic_sign — PPLCNet_x1_0 (20 epoch, AMP-O1, EMA=0.9999, batch=512, lr=0.04)',
             fontsize=12, fontweight='bold')

# 1. train loss per iteration (smoothed)
if iter_loss:
    arr = np.array(iter_loss)
    win = 50
    if len(arr) > win:
        kernel = np.ones(win) / win
        sm = np.convolve(arr, kernel, mode='valid')
        xsm = np.arange(win - 1, len(arr))
    else:
        sm, xsm = arr, np.arange(len(arr))
    axes[0, 0].plot(np.arange(len(arr)), arr, alpha=0.25, color='C0', label='raw')
    axes[0, 0].plot(xsm, sm, color='C1', linewidth=2, label=f'smooth(w={win})')
    axes[0, 0].set_title('Train Loss per iteration')
    axes[0, 0].set_xlabel('iteration (print_batch_step=10)'); axes[0, 0].set_ylabel('CELoss')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# 2. per-epoch train avg loss
if train_avg_loss:
    ep = np.arange(1, len(train_avg_loss) + 1)
    axes[0, 1].plot(ep, train_avg_loss, 'o-', color='C3')
    axes[0, 1].set_title('Train Avg CELoss per epoch')
    axes[0, 1].set_xlabel('epoch'); axes[0, 1].set_ylabel('loss')
    axes[0, 1].grid(alpha=0.3)

# 3. eval top1 per epoch (raw + ema)
if eval_raw_top1:
    lo = min(min(eval_raw_top1), min(eval_ema_top1 or [1])) * 100
    ymin = max(50, lo - 1)
    axes[1, 0].plot(eval_raw_ep, [v * 100 for v in eval_raw_top1], 'o-', color='C2', label='eval top1 (raw model)')
    if eval_ema_top1:
        axes[1, 0].plot(eval_ema_ep, [v * 100 for v in eval_ema_top1], 's--', color='C4', label='eval top1 (EMA model)')
    axes[1, 0].set_title('Eval Top-1 Accuracy per epoch')
    axes[1, 0].set_xlabel('epoch'); axes[1, 0].set_ylabel('accuracy (%)')
    axes[1, 0].set_ylim(ymin, 100.2)
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

# 4. learning rate curve per iter
if iter_lr:
    axes[1, 1].plot(np.arange(len(iter_lr)), iter_lr, color='C5')
    axes[1, 1].set_title('Learning Rate schedule (Cosine + 2-epoch warmup)')
    axes[1, 1].set_xlabel('iteration'); axes[1, 1].set_ylabel('lr')
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'curves.png')
plt.savefig(out_png, dpi=130)
print('saved:', out_png)

# ---- combined figure: loss + acc only (as user requested) ----
fig2, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()
ep1 = np.arange(1, len(train_avg_loss) + 1)
ax.plot(ep1, train_avg_loss, 'o-', color='C0', label='Train CELoss')
ax2.plot(eval_raw_ep, [v * 100 for v in eval_raw_top1], 's-', color='C2', label='Eval Top-1 Accuracy')
ax2.plot(eval_ema_ep, [v * 100 for v in eval_ema_top1], '^--', color='C4', label='Eval Top-1 Accuracy (EMA)')
ax.set_xlabel('epoch')
ax.set_ylabel('Train CELoss', color='C0')
ax2.set_ylabel('Eval Top-1 Accuracy (%)', color='C2')
ax.set_title('PULC traffic_sign — Loss vs Accuracy over 20 epochs\n(train CELoss left axis; eval acc right axis)')
ax.grid(alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.tight_layout()
out_png2 = os.path.join(OUT_DIR, 'loss_acc_curve.png')
plt.savefig(out_png2, dpi=130)
print('saved:', out_png2)