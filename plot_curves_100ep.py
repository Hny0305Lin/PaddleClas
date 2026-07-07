"""Parse PaddleClas train.log (100 epoch run) and plot acc_curve.png + loss_curve.png.

Handles training restarts by deduplicating per epoch (last occurrence wins).
Distinguishes raw vs EMA eval via the 'best metric ema' marker line.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r'C:/Users/xindao/Desktop/paddletest/PaddleClas/output/PPLCNet_x1_0_100ep'
LOG_TXT = os.path.join(OUT_DIR, 'train.log')

train_epoch_re = re.compile(r'\[Train\]\[Epoch (\d+)/\d+\]\[Avg\]top1: ([\d.]+), top5: ([\d.]+), CELoss: ([\d.]+)')
eval_avg_re    = re.compile(r'\[Eval\]\[Epoch (\d+)\]\[Avg\]CELoss: ([\d.]+),.*top1: ([\d.]+), top5: ([\d.]+)')

train_top1, train_loss = {}, {}
eval_raw_top1, eval_raw_loss = {}, {}
eval_ema_top1, eval_ema_loss = {}, {}

with open(LOG_TXT, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

n = len(lines)
i = 0
while i < n:
    line = lines[i]
    m = train_epoch_re.search(line)
    if m:
        ep = int(m.group(1))
        train_top1[ep] = float(m.group(2))
        train_loss[ep] = float(m.group(4))
        i += 1
        continue
    m = eval_avg_re.search(line)
    if m:
        ep = int(m.group(1)); celoss = float(m.group(2)); top1 = float(m.group(3))
        j = i + 1
        nxt = ''
        while j < n and not nxt.strip():
            nxt = lines[j]; j += 1
        if 'ema' in nxt:
            eval_ema_top1[ep] = top1
            eval_ema_loss[ep] = celoss
        else:
            eval_raw_top1[ep] = top1
            eval_raw_loss[ep] = celoss
        i += 1
        continue
    i += 1

def sorted_items(d):
    eps = sorted(d.keys())
    return eps, [d[e] for e in eps]

tr_eps, tr_top1 = sorted_items(train_top1)
_, tr_loss = sorted_items(train_loss)
raw_eps, raw_top1 = sorted_items(eval_raw_top1)
_, raw_loss = sorted_items(eval_raw_loss)
ema_eps, ema_top1 = sorted_items(eval_ema_top1)
_, ema_loss = sorted_items(eval_ema_loss)

print(f'parsed (deduped): train epochs={len(tr_eps)}, '
      f'raw eval={len(raw_eps)}, ema eval={len(ema_eps)}')
print(f'  train top1 range: {min(tr_top1):.4f} -> {max(tr_top1):.4f}')
if ema_top1:
    print(f'  ema  top1 range: {min(ema_top1):.4f} -> {max(ema_top1):.4f} '
          f'(best @ epoch {ema_eps[int(np.argmax(ema_top1))]})')
if raw_top1:
    print(f'  raw  top1 range: {min(raw_top1):.4f} -> {max(raw_top1):.4f} '
          f'(best @ epoch {raw_eps[int(np.argmax(raw_top1))]})')

TITLE = 'PULC traffic_sign - PPLCNet_x1_0 (100 epoch, AMP-O1, EMA=0.9999, batch=512, lr=0.04)'

# ---- acc_curve.png ----
fig, ax = plt.subplots(figsize=(10, 6))
if raw_top1:
    ax.plot(raw_eps, [v * 100 for v in raw_top1], 'o-', color='C2', ms=4,
            label='Eval Top-1 (raw model)')
if ema_top1:
    ax.plot(ema_eps, [v * 100 for v in ema_top1], 's--', color='C4', ms=4,
            label='Eval Top-1 (EMA model)')
if tr_top1:
    ax.plot(tr_eps, [v * 100 for v in tr_top1], '.:', color='C0', alpha=0.6,
            label='Train Top-1')
ax.set_xlabel('epoch')
ax.set_ylabel('Top-1 Accuracy (%)')
fig.suptitle('PPLCNet_x1_0 交通标志识别 验证准确率曲线', fontsize=14, fontweight='bold')
ax.set_title('Traffic Sign Classification - Accuracy Curve', fontsize=11, style='italic', color='dimgray')
all_acc = [v * 100 for v in (raw_top1 + ema_top1 + tr_top1)]
if all_acc:
    lo = min(all_acc)
    ax.set_ylim(max(50, lo - 1), 100.2)
if tr_eps:
    ax.set_xlim(min(tr_eps) - 1, max(tr_eps) + 2)
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
last_ep = max(tr_eps or raw_eps or ema_eps)
if ema_top1:
    ax.annotate(f'{ema_top1[-1]*100:.2f}%', xy=(last_ep, ema_top1[-1]*100),
                xytext=(6, 6), textcoords='offset points', fontsize=9, color='C4', fontweight='bold')
if raw_top1:
    ax.annotate(f'{raw_top1[-1]*100:.2f}%', xy=(last_ep, raw_top1[-1]*100),
                xytext=(6, -10), textcoords='offset points', fontsize=9, color='C2', fontweight='bold')
if tr_top1:
    ax.annotate(f'{tr_top1[-1]*100:.2f}%', xy=(last_ep, tr_top1[-1]*100),
                xytext=(6, -10), textcoords='offset points', fontsize=9, color='C0')
plt.tight_layout()
out_acc = os.path.join(OUT_DIR, 'acc_curve.png')
plt.savefig(out_acc, dpi=130)
plt.close(fig)
print('saved:', out_acc)

# ---- loss_curve.png ----
fig, ax = plt.subplots(figsize=(10, 6))
if tr_loss:
    ax.plot(tr_eps, tr_loss, 'o-', color='C0', ms=4, label='Train Loss (avg/epoch)')
if raw_loss:
    ax.plot(raw_eps, raw_loss, 's-', color='C2', ms=4, label='Eval Loss (raw model)')
if ema_loss:
    ax.plot(ema_eps, ema_loss, '^--', color='C4', ms=4, label='Eval Loss (EMA model)')
ax.set_xlabel('epoch')
ax.set_ylabel('Loss')
fig.suptitle('PPLCNet_x1_0 交通标志识别 训练和验证损失曲线', fontsize=14, fontweight='bold')
ax.set_title('Traffic Sign Classification - Loss Curve', fontsize=11, style='italic', color='dimgray')
if tr_eps:
    ax.set_xlim(min(tr_eps) - 1, max(tr_eps) + 2)
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
last_ep = max(tr_eps or raw_eps or ema_eps)
if tr_loss:
    ax.annotate(f'{tr_loss[-1]:.4f}', xy=(last_ep, tr_loss[-1]),
                xytext=(6, 6), textcoords='offset points', fontsize=9, color='C0')
if raw_loss:
    ax.annotate(f'{raw_loss[-1]:.4f}', xy=(last_ep, raw_loss[-1]),
                xytext=(6, -12), textcoords='offset points', fontsize=9, color='C2', fontweight='bold')
if ema_loss:
    ax.annotate(f'{ema_loss[-1]:.4f}', xy=(last_ep, ema_loss[-1]),
                xytext=(6, -24), textcoords='offset points', fontsize=9, color='C4', fontweight='bold')
plt.tight_layout()
out_loss = os.path.join(OUT_DIR, 'loss_curve.png')
plt.savefig(out_loss, dpi=130)
plt.close(fig)
print('saved:', out_loss)
