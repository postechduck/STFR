#!/usr/bin/env python3
"""Interactions per 60-day block and the two evaluation halves (Figure 2) from results/split_blocks.csv.

  python analysis/plot_split.py [--results results] [--out figures]
"""
import argparse
import collections
import datetime as dt
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from matplotlib.patches import Patch

C_TRAIN, C_VAL, C_TEST = '#7f7f7f', '#2b6cb0', '#c0392b'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--out', default='figures')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = pd.read_csv(os.path.join(a.results, 'split_blocks.csv')).to_dict('records')
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r['dataset'], []).append(r)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, len(by), figsize=(3.9 * len(by), 3.2), squeeze=False)
    axes = axes[0]

    def d(s):
        return dt.datetime.strptime(s, '%Y-%m-%d')

    for ax, (ds, rs) in zip(axes, by.items()):
        for r in rs:
            s, e = d(r['block_start_date']), d(r['block_end_date'])
            width = (e - s).days + 1
            if r['role'] == 'train':
                y, c, z = int(r['n_interactions']), C_TRAIN, 2
            elif r['role'] == 'val':
                y, c, z = int(r['n_rows_prefilter']), C_VAL, 3
            else:
                y, c, z = int(r['n_rows_prefilter']), C_TEST, 3
            ax.bar(s, y, width=width, align='edge', color=c, linewidth=0, zorder=z)
        val = [r for r in rs if r['role'] == 'val']
        if val:
            ax.axvline(d(val[0]['block_start_date']), color=C_TEST, linestyle='--', linewidth=0.6, zorder=1)
        ax.set_title(ds)
        span_years = (d(rs[-1]['block_end_date']) - d(rs[0]['block_start_date'])).days / 365.25
        ax.xaxis.set_major_locator(mdates.YearLocator(4 if span_years > 12 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    axes[0].set_ylabel('interactions per block')
    handles = [Patch(color=C_TRAIN, label='training span (blocks)'), Patch(color=C_VAL, label='validation window'),
               Patch(color=C_TEST, label='test window')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, 'split_blocks.%s' % ext), dpi=160, bbox_inches='tight')
    print('[figure] -> %s/split_blocks.{pdf,png}' % a.out)


if __name__ == '__main__':
    main()
