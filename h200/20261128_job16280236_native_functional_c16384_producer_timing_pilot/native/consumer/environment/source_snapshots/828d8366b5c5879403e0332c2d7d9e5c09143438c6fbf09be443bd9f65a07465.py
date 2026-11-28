#!/usr/bin/env python3
"""Plot measured H100 control ceilings with clearly analytical decode proxies."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    xs = np.logspace(-1, 3, 400)
    envelope = np.minimum(data['compute_ceiling_tflops'],
                          data['useful_copy_ceiling_gb_s'] * xs / 1000)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fig.subplots_adjust(left=.15, right=.97, top=.89, bottom=.22)
    ax.loglog(xs, envelope, color='#213c68', linewidth=2,
              label='GEMM / useful-copy control envelope')
    color = {'timing': '#bd4b22', 'energy': '#577b2f'}
    for point in data['points']:
        x = point['analytical_intensity_flops_per_read_byte']
        y = point['achieved_tflops_from_proxy_opcount']
        ax.scatter(x, y, s=68, color=color.get(point['software_mode'], '#bd4b22'), zorder=3)
        ax.annotate(f"B{point['batch_size']}/C{point['context_length']}",
                    (x, y), xytext=(6, 5), textcoords='offset points', fontsize=9)
    ax.set_xlim(.3, 1000)
    ax.set_ylim(.08, 1500)
    ax.set_xlabel('Analytical FLOPs per assumed read byte (not measured HBM intensity)')
    ax.set_ylabel('Throughput (TFLOP/s)')
    ax.set_title('H100 BF16 controls and bridge decode proxies')
    ax.grid(True, which='both', alpha=.2)
    ax.legend(loc='upper left')
    fig.text(.02, .025, 'Copy ceiling counts useful bytes. Decode FLOPs and HBM traffic are analytical / unmeasured.', fontsize=8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    print(a.output)


if __name__ == '__main__':
    main()
