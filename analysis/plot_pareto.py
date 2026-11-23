#!/usr/bin/env python3
"""Recall@20 vs nALRP@20 per setting from results/summary.csv.

  python analysis/plot_pareto.py [--results results] [--out figures]

Three-seed means with one-standard-deviation bars (population convention, as in
summary.csv) for the nine methods and STFR (forecast) in the nine settings; the nALRP axis
is inverted, so points toward the upper right have higher accuracy and lower mean stale
popularity.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import MAIN_ARMS, VARIANT_ARMS, REMOVAL, INJECTION, CELL_ORDER, label

COL = {'base': '#888888', 'STFR': '#d62728', 'STFR_forecast': '#ff9896'}
COL.update({m: '#9467bd' for m in REMOVAL})
COL.update({m: '#1f77b4' for m in INJECTION})
SHORT = {'base': 'backbone'}
DATASET = {'Douban-movie': 'Douban-Movie'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--out', default='figures')
    p.add_argument('--k', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv(os.path.join(a.results, 'summary.csv'))
    s = s[(s.k == a.k) & s.arm.isin(MAIN_ARMS + VARIANT_ARMS)]
    cells = [c for c in CELL_ORDER if ((s.data == c[0]) & (s.backbone == c[1])).any()]
    n = len(cells)
    ncol = 3 if n > 4 else max(n, 1)                 # nine settings: datasets x backbones
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.75 * ncol, 4.25 * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax, (ds, bb) in zip(axes.ravel(), cells):
        g = s[(s.data == ds) & (s.backbone == bb)].set_index('arm')
        for meth in MAIN_ARMS + VARIANT_ARMS:
            if meth not in g.index:
                continue
            x, y = g.loc[meth, 'recall'], g.loc[meth, 'nalrp']
            xe, ye = g.loc[meth, 'recall_sd'], g.loc[meth, 'nalrp_sd']
            star = meth in ('STFR',) + tuple(VARIANT_ARMS)
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt='none', ecolor=COL[meth], elinewidth=0.8, zorder=3)
            ax.scatter(x, y, marker='*' if star else 'o', s=330 if star else 70, c=COL[meth], edgecolors='black',
                       lw=0.9 if star else 0.6, zorder=5 if star else 4, label=label(meth))
            if meth not in VARIANT_ARMS:
                ax.annotate(SHORT.get(meth, meth), (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8)
        ax.invert_yaxis()
        ax.set_title('%s, %s' % (DATASET.get(ds, ds), bb))
        ax.set_xlabel('Recall@%d' % a.k)
        ax.set_ylabel('nALRP@%d (axis inverted)' % a.k)
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', mfc=COL['base'], mec='black', label='Unmodified backbone'),
               Line2D([], [], marker='o', ls='', mfc=COL['IPS'], mec='black', label='Removal-oriented'),
               Line2D([], [], marker='o', ls='', mfc=COL['PDA'], mec='black', label='Injection-oriented'),
               Line2D([], [], marker='*', ls='', ms=13, mfc=COL['STFR'], mec='black', label='STFR'),
               Line2D([], [], marker='*', ls='', ms=13, mfc=COL['STFR_forecast'], mec='black', label='STFR (forecast)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'pareto_recall_nalrp.%s' % ext), dpi=160, bbox_inches='tight')
    print('[figure] -> %s/pareto_recall_nalrp.{pdf,png}' % a.out)


if __name__ == '__main__':
    main()
