#!/usr/bin/env python3
"""Recall@20 vs nALRP@20 per setting (Figure 3) from results/summary.csv.

  python analysis/plot_pareto.py [--results results] [--out figures]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import MAIN_ARMS, CELL_ORDER

COL = {'base': '#888888', 'IPS': '#8c564b', 'DICE': '#9467bd', 'DDC': '#17becf', 'PDA': '#1f77b4',
       'TIDE': '#ff7f0e', 'CausalEPP': '#2ca02c', 'STFR': '#d62728'}
DISP = {'CausalEPP': 'CEPP', 'STFR': 'STFR (ours)', 'TIDE': 'TIDE(full)'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--out', default='figures')
    p.add_argument('--k', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv(os.path.join(a.results, 'summary.csv'))
    s = s[(s.k == a.k) & s.arm.isin(MAIN_ARMS)]
    cells = [c for c in CELL_ORDER if ((s.data == c[0]) & (s.backbone == c[1])).any()]
    n = len(cells)
    ncol = 4 if n > 4 else max(n, 1)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.75 * ncol, 4.25 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax, (ds, bb) in zip(axes.ravel(), cells):
        g = s[(s.data == ds) & (s.backbone == bb)].set_index('arm')
        for meth in MAIN_ARMS:
            if meth not in g.index:
                continue
            x, y = g.loc[meth, 'recall'], g.loc[meth, 'nalrp']
            if meth == 'STFR':
                ax.scatter(x, y, marker='*', s=430, c=COL[meth], edgecolors='black', lw=1.1, zorder=5, label=DISP.get(meth, meth))
            else:
                ax.scatter(x, y, s=90, c=COL[meth], edgecolors='black', lw=0.6, zorder=4, label=DISP.get(meth, meth))
            ax.annotate(DISP.get(meth, meth).split(' ')[0], (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8)
        ax.invert_yaxis()
        ax.set_title('%s / %s' % (ds.replace('Amazon-', '').replace('-movie', ''), bb))
        ax.set_xlabel('Recall@%d' % a.k)
        ax.set_ylabel('nALRP@%d (lower is up)' % a.k)
        ax.grid(alpha=0.3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(8, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'pareto_recall_nalrp.%s' % ext), dpi=160, bbox_inches='tight')
    print('[figure] -> %s/pareto_recall_nalrp.{pdf,png}' % a.out)


if __name__ == '__main__':
    main()
