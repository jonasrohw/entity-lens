"""Descriptive reporting for explicitly post-selection follow-up experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from jlens.mechanistic import save_json
from jlens.paired_mechanistic_report import paired, RELATIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args=parser.parse_args();out=args.out
    main=json.loads((out/'analysis/summary.json').read_text())
    correct={r['subject'] for r in main['cases'] if r['mode']=='controlled' and r['both_model_correct']}
    result=[]
    for relation in RELATIONS:
        d=json.loads((out/f'mediation_{relation}.json').read_text());assert d['complete']
        for source in RELATIONS:
            rows={}
            for r in d['rows']:
                if r['source']==source:rows.setdefault(r['subject'],{})[r['head']]=r
            subjects=sorted(rows);mask=np.array([s in correct for s in subjects])
            for metric in ['logp','margin','hit']:
                damage=np.array([rows[s][18]['clean'][metric]-rows[s][18]['ablated'][metric] for s in subjects])
                restored=np.array([rows[s][18]['restored'][metric]-rows[s][18]['ablated'][metric] for s in subjects])
                random=np.array([np.mean([rows[s][h]['restored'][metric]-rows[s][h]['ablated'][metric] for h in d['random_heads']]) for s in subjects])
                result.append(dict(relation=relation,source=source,metric=metric,damage=paired(damage),
                                   restored=paired(restored),random_restored=paired(random),
                                   excess_restoration=paired(restored-random),
                                   both_correct_excess=paired((restored-random)[mask]),
                                   ratio_of_mean_restoration_to_mean_damage=float(restored.mean()/damage.mean()) if abs(damage.mean())>1e-8 else None))
    supplementary=dict(complete=True,exploratory=True,independent_confirmation=False,
                       mediation=result,readout=json.loads((out/'intermediate_readout.json').read_text())['aggregated'],
                       lexical=json.loads((out/'lexical_diagnostic.json').read_text())['summary'],
                       caveat='Head selected using these subjects. Intervals and permutation p-values are descriptive and do not account for selection; no new confirmatory significance claim.')
    save_json(out/'analysis/supplementary.json',supplementary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,relation in zip(axes,RELATIONS):
        rows=[r for r in result if r['relation']==relation and r['metric']=='logp']
        for offset,field,label in [(-.25,'damage','Damage from removal'),(0,'restored','Recovered by head 22.18'),(.25,'random_restored','Recovered by random heads')]:
            ax.bar(np.arange(2)+offset,[r[field]['mean'] for r in rows],width=.25,label=label)
        ax.set(title='Capital' if relation==RELATIONS[0] else 'Currency',xticks=[0,1],xticklabels=['Capital subspace','Currency subspace'],xlabel='Removed subject directions')
    axes[0].set_ylabel('Mean log-probability change (nats)');axes[1].legend(fontsize=7)
    fig.suptitle('Exploratory head restoration after subject-subspace removal')
    fig.tight_layout();fig.savefig(out/'analysis/head_restoration.png',dpi=180);fig.savefig(out/'analysis/head_restoration.pdf');plt.close(fig)
    print(json.dumps([r for r in result if r['metric']=='logp'],indent=2))


if __name__=='__main__':main()
