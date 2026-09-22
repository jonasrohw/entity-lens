"""Exploratory output-wording and intermediate-readout diagnostics.

Added after observing country-related first tokens in currency LRE errors.
This analysis does not select the confirmatory interventions or their sites.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jlens.mechanistic import save_json
from jlens.paired_mechanistic import RELATIONS, load_baseline, load_runner, provenance
from jlens.splits import first_object_token


def lexical_diagnostic(runner,args,countries,pairs):
    rows=[]
    for mode in ['original','controlled','unit']:
        for relation in RELATIONS:
            meta=json.loads((args.out/f'baseline_{mode}_{relation}.json').read_text())
            layer=meta['layers']['17'];score=layer['grid'][str(layer['beta'])]
            for i,subject in enumerate(meta['subjects']):
                if i<meta['nval']:continue
                country=countries[pairs[subject]['iso']]
                adjectives=sorted(set(country.get('demonyms',{}).get('eng',{}).values()))
                adjective_ids={first_object_token(s,runner.tokenizer) for s in adjectives}
                subject_ids={first_object_token(s,runner.tokenizer) for s in [subject,country['name']['common']]}
                top=score['top'][i]
                rows.append(dict(mode=mode,relation=relation,subject=subject,adjectives=adjectives,
                                 lre=runner.tokenizer.decode([top]),model=meta['target_text'][i],
                                 faithful=bool(score['hit'][i]),model_correct=bool(meta['model_correct'][i]),
                                 demonym_first_token=top in adjective_ids,subject_first_token=top in subject_ids,
                                 model_demonym_first_token=meta['targets'][i] in adjective_ids,
                                 model_subject_first_token=meta['targets'][i] in subject_ids))
    summary=[]
    for mode in ['original','controlled','unit']:
        for relation in RELATIONS:
            rs=[r for r in rows if r['mode']==mode and r['relation']==relation]
            errors=[r for r in rs if not r['faithful']]
            summary.append(dict(mode=mode,relation=relation,n=len(rs),errors=len(errors),
                                identity_errors=sum(r['demonym_first_token'] or r['subject_first_token'] for r in errors),
                                full_model_identity_tokens=sum(r['model_demonym_first_token'] or r['model_subject_first_token'] for r in rs)))
    save_json(args.out/'lexical_diagnostic.json',dict(exploratory=True,
        note='First-token matches to country labels/demonyms do not determine full-answer semantics.',rows=rows,summary=summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"generated/paired_relation_feasibility_20260920")
    args=parser.parse_args()
    runner=load_runner()
    countries={x['cca2']:x for x in json.loads((args.data/'sources/countries.json').read_text())}
    pairs={x['subject']:x for x in json.loads((args.data/'country_pairs.json').read_text())}
    lexical_diagnostic(runner,args,countries,pairs)
    rows=[]
    runner.layers=list(range(1,len(runner.blocks)+1))
    for relation in RELATIONS:
        meta,pack=load_baseline(args,relation)
        for index,subject in enumerate(meta['subjects']):
            if index<meta['nval']:continue
            c=pack['cache'][subject]
            country=countries[pairs[subject]['iso']]
            adjectives=set(country.get('demonyms',{}).get('eng',{}).values())
            words=adjectives|{subject,country['name']['common']}
            identity_ids=sorted({first_object_token(s,runner.tokenizer) for s in words})
            layer=meta['layers']['17'];lre_top=layer['grid'][str(layer['beta'])]['top'][index]
            with torch.no_grad():
                final,states,_=runner.forward(c['ids'],c['ids'].shape[1]-1,capture=True)
                assert float((final-c['final']).abs().max())<1e-4
                for block,state in states.items():
                    logits=runner.logits(state)
                    logp=logits.log_softmax(-1)
                    top=int(logits.argmax())
                    rows.append(dict(relation=relation,subject=subject,block=block,top=top,
                                     top_text=runner.tokenizer.decode([top]),identity_top=top in identity_ids,
                                     target_top=top==c['top'],gold_top=top==c['gold'],lre_top=top==lre_top,
                                     target_logp=float(logp[c['top']]),lre_token_logp=float(logp[lre_top]),
                                     identity_logp=float(torch.logsumexp(logp[identity_ids],dim=0)),
                                     lre_faithful=lre_top==c['top']))
            print('readout',relation,subject,flush=True)
    aggregated=[]
    for relation in RELATIONS:
        for block in range(1,len(runner.blocks)+1):
            rs=[r for r in rows if r['relation']==relation and r['block']==block]
            errors=[r for r in rs if not r['lre_faithful']]
            aggregated.append(dict(relation=relation,block=block,n=len(rs),
                                   identity_top=float(np.mean([r['identity_top'] for r in rs])),
                                   target_top=float(np.mean([r['target_top'] for r in rs])),
                                   lre_error_token_top=float(np.mean([r['lre_top'] for r in errors])),
                                   target_logp=float(np.mean([r['target_logp'] for r in rs]))))
    save_json(args.out/'intermediate_readout.json',dict(complete=True,exploratory=True,provenance=provenance(),rows=rows,aggregated=aggregated,
        caution='Final-normalization logit-lens readouts are descriptive, not calibrated intermediate beliefs or causal evidence. A first token does not determine a complete answer.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for relation,label in zip(RELATIONS,['Capital','Currency']):
        data=[r for r in aggregated if r['relation']==relation]
        for ax,metric,title in zip(axes,['target_top','identity_top','lre_error_token_top'],
                                   ['Matches final model token','Country/demonym first token','Matches LRE error token (error cases)']):
            ax.plot([r['block'] for r in data],[r[metric] for r in data],label=label)
            ax.set(title=title,xlabel='Transformer block',ylim=(0,1))
            if (args.out/'head_selection.json').exists():
                for site in json.loads((args.out/'head_selection.json').read_text())['selected']:
                    ax.axvline(site['block'],color='gray',alpha=.3,ls='--')
    axes[0].set_ylabel('Fraction of subjects');axes[-1].legend()
    fig.suptitle('Exploratory logit-lens readout at the prediction position')
    fig.tight_layout()
    folder=args.out/'analysis';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'intermediate_readout.pdf');fig.savefig(folder/'intermediate_readout.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
