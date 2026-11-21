import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

import os
for d in (os.path.expanduser('~/.local/share/fonts'), '/usr/share/fonts'):
    for f in glob.glob(os.path.join(d, '**', 'IBMPlexSans-*.ttf'), recursive=True):
        fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': ['IBM Plex Sans', 'DejaVu Sans'], 'font.size': 11})

BG, FG, DIM, GRID = '#0b0c0f', '#e8eaed', '#8b929c', '#1d2026'
AZ, DEEP, LIGHT, GREY = '#18a0ff', '#0c6ce0', '#9ad5ff', '#4a505a'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'

def frame(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=DIM, length=0)

# 1: the steps
steps = [
    ('llama.cpp + the Inkling 2 PRs, as is', 25.0),
    ('expert gate/up halves kept on the GPU', 66.5),
    ('loop and depth-mix ops, fused router', 84.0),
    ('more fused ops, faster long-context attention', 98.7),
    ('own NVFP4 shader, expert scale in the matmul', 108.9),
]
cols = [GREY, '#0a4f9e', DEEP, '#1288f0', AZ]
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
fig.patch.set_facecolor(BG)
frame(ax)
y = list(range(len(steps)))[::-1]
ax.barh(y, [v for _, v in steps], height=0.62, color=cols, zorder=3)
for yi, (label, v), c in zip(y, steps, cols):
    ax.text(v + 1.5, yi, f'{v:g}', va='center', ha='left', color=FG if c != AZ else AZ,
            fontsize=13, fontweight='semibold', zorder=4)
ax.set_yticks(y, [s for s, _ in steps], color=FG, fontsize=11.5)
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100, 125], ['0', '25', '50', '75', '100', '125'], color=DIM, fontsize=10)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_xlabel('tokens per second', color=DIM, fontsize=10.5, labelpad=8)
fig.text(0.015, 0.95, 'Inkling 2 Mini on an RX 7900 XT, Vulkan', color=FG, fontsize=17, fontweight='semibold', ha='left', va='top')
fig.text(0.015, 0.875, 'generation speed after each step: 25 -> 108.9 tok/s, 4.4x  |  llama-bench tg64, UD-Q4_K_M, batch 1', color=DIM, fontsize=11, ha='left', va='top')
fig.subplots_adjust(left=0.375, right=0.96, top=0.80, bottom=0.13)
fig.savefig(OUT + 'inkling-steps.png', facecolor=BG)

# 2: short vs 32k context, the last three builds (same settings)
builds = ['first custom ops', 'more fused ops', 'NVFP4 shader (final)']
short = [83.9, 98.7, 108.9]
long32 = [71.3, 84.6, 92.5]
bcols = [DEEP, '#1288f0', AZ]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
fig.patch.set_facecolor(BG)
frame(ax)
w = 0.26
for i, (b, c) in enumerate(zip(builds, bcols)):
    xs = [0 + (i - 1) * w, 1 + (i - 1) * w]
    vals = [short[i], long32[i]]
    ax.bar(xs, vals, width=w * 0.9, color=c, zorder=3, label=b)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.8, f'{v:g}', ha='center', va='bottom', color=AZ if c == AZ else FG,
                fontsize=12.5, fontweight='semibold', zorder=4)
ax.set_xticks([0, 1], ['short context', '32k tokens of context'], color=FG, fontsize=12)
ax.set_ylim(0, 125)
ax.set_yticks([0, 25, 50, 75, 100, 125], ['0', '25', '50', '75', '100', '125'], color=DIM, fontsize=10)
ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_ylabel('tokens per second', color=DIM, fontsize=10.5, labelpad=8)
leg = fig.legend(loc='upper left', bbox_to_anchor=(0.085, 0.805), ncol=3, frameon=False, fontsize=11, labelcolor=FG, handlelength=1.2, columnspacing=2.2)
fig.text(0.015, 0.95, 'Short vs long context', color=FG, fontsize=17, fontweight='semibold', ha='left', va='top')
fig.text(0.015, 0.875, 'llama-bench tg64 at depth 0 and 32768  |  K q8_0 / V q4_0 cache, flash attention', color=DIM, fontsize=11, ha='left', va='top')
fig.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.11)
fig.savefig(OUT + 'inkling-context.png', facecolor=BG)
print('ok')
