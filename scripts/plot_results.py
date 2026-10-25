#!/usr/bin/env python3
"""Plot every included arm, with accuracy and recorded deployment-token cost."""
from pathlib import Path
from fractions import Fraction
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from maze.results import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/development-results.svg')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none',
        'svg.hashsalt':'maze-development-results-v1', 'font.size':10})
    report = reconstruct(ROOT)
    cohorts = ['gpqa99_v206', 'gpqa32_v166', 'mmlu32_v160']
    names = {'Maze_SC3':'Maze SC3', 'Maze_one_call':'Maze one call',
        'Maze_selected_SC3':'Maze SC3 (v121)', 'learned_best_SC3':'Maze SC3',
        'Raw_one_call':'Random one call', 'uniform_Raw_SC3':'Random-prompt SC3',
        'uniform_fixed_Raw_SC3':'Random-prompt SC3', 'legacy_v108_SC3':'Legacy planner SC3',
        'retrieval_one_call':'Retrieval one call', 'retrieval_SC3':'Retrieval SC3'}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={'height_ratios':[10,4,3]})
    fig.subplots_adjust(left=.23, right=.97, top=.89, bottom=.08, hspace=.55)
    fig.suptitle('Maze: accuracy and deployment cost', x=.035, y=.98, ha='left',
                 fontsize=21, fontweight='bold', color='#10233f')
    fig.text(.035,.94,'Saved development outcomes · Qwen3.5-4B · four complete exemplars', color='#475569', fontsize=12)
    for ax, cohort in zip(axes, cohorts):
        items = list(report['cohorts'][cohort].items())
        for i,(arm,value) in enumerate(items):
            primary = arm.startswith('Maze') or arm=='learned_best_SC3'
            accuracy = 100*float(Fraction(value['accuracy_exact']))
            tokens = float(Fraction(value['processed_tokens_exact']))
            ax.barh(i, accuracy, height=.66, color='#2463c4' if primary else '#b9c5d5')
            ax.text(accuracy+1, i, f"{value['correct']}/{value['questions']}", va='center', fontsize=9, color='#10233f')
            ax.text(104, i, f"{value['calls_exact']} calls   {tokens:,.0f} tokens", va='center', fontsize=9, color='#475569')
        ax.set_yticks(range(len(items)),[names.get(a,a.replace('Raw_','Random-path ')) for a,_ in items])
        ax.invert_yaxis();ax.set_xlim(0,143)
        ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
        ax.set_title(report['cohort_displays'][cohort],loc='left',fontweight='bold',color='#10233f',pad=9)
        ax.xaxis.grid(True,color='#e2e8f0');ax.set_axisbelow(True)
        ax.tick_params(axis='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.035,.035,'All 17 arms shown. Selected cohorts partly overlap; they are not independent replications.\nTokens = target input + output + recorded query encoding; offline costs and latency are separate.',
             color='#475569', fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'Date':None,'Creator':'Maze plot_results.py'})
    plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
