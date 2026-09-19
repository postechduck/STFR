#!/usr/bin/env python3
"""Block-configuration figure (Figure 4) from results/block_configurations.csv.

  python analysis/plot_block_configs.py [--results results] [--out figures] [--k 20]

The manuscript's figure uses @20, the cutoff written by analysis/block_config_table.py.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--out', default='figures')
    p.add_argument('--k', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = os.path.join(a.results, 'block_configurations.csv')
    if not os.path.exists(src):
        raise SystemExit('%s not found: run python analysis/block_config_table.py first' % src)
    d = pd.read_csv(src)
    have = sorted(d.k.unique().tolist())
    d = d[(d.k == a.k) & (d.method == 'STFR')]
    if d.empty or not {'recall_gain_pct', 'nalrp_diff'} <= set(d.columns) or d['recall_gain_pct'].isna().all():
        raise SystemExit('%s has no STFR-vs-base rows at @%d (cutoffs in the file: %s); '
                         'both base and STFR finals are needed per configuration' % (src, a.k, have))
    Ls = sorted(d.L_days.unique())
    bbs = [b for b in ('MF', 'LightGCN', 'SimGCL') if b in d.backbone.values]
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.25))
    fig.subplots_adjust(left=.078, right=.99, bottom=.19, top=.77, wspace=.30)
    xs = np.arange(len(Ls))
    width = 0.8 / max(len(bbs), 1)
    colors = {'MF': '#2469a0', 'LightGCN': '#d58727', 'SimGCL': '#3c8b6d'}
    for j, bb in enumerate(bbs):
        g = d[d.backbone == bb].set_index('L_days')
        gains = [g.loc[L, 'recall_gain_pct'] if L in g.index else np.nan for L in Ls]
        diffs = [g.loc[L, 'nalrp_diff'] if L in g.index else np.nan for L in Ls]
        off = (j - (len(bbs) - 1) / 2) * width
        axes[0].bar(xs + off, gains, width, color=colors.get(bb), label=bb, zorder=3)
        axes[1].bar(xs + off, diffs, width, color=colors.get(bb), label=bb, zorder=3)
    for ax in axes:
        ax.set_xticks(xs, [str(L) for L in Ls])
        ax.set_xlabel('Block length (days)')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e2e5e9', linewidth=.6)
        ax.axhline(0, color='#555555', linewidth=.8)
    axes[0].set_title('(a) Recall@%d gain over base' % a.k, loc='left', pad=12)
    axes[1].set_title('(b) nALRP@%d: STFR minus base' % a.k, loc='left', pad=12)
    axes[0].set_ylabel('Relative gain (%)')
    axes[1].set_ylabel('Difference')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, 1.005), ncol=len(bbs), frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'block_configurations.%s' % ext), dpi=170, bbox_inches='tight', pad_inches=.04)
    print('[figure] -> %s/block_configurations.{pdf,png}' % a.out)


if __name__ == '__main__':
    main()
