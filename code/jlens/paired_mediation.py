"""Exploratory clean-head restoration after subject-subspace removal.

Head 22.18 was selected from the completed confirmation scan. This follow-up is
post-selection exploration, not a new independent confirmatory experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jlens.mechanistic import save_json
from jlens.paired_mechanistic import RELATIONS, load_runner, load_baseline, capture_components, patched_component, score_one, provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--relation',choices=RELATIONS,required=True)
    args=parser.parse_args()
    runner=load_runner()
    meta,pack=load_baseline(args,args.relation)
    bases=torch.load(args.out/'channels_basis.pt',map_location=runner.device,weights_only=False)
    info=json.loads((args.out/'channels_basis.json').read_text())
    # These controls are picked by seed, without looking at their head effects.
    rng=np.random.default_rng(2010)
    controls=sorted(int(h) for h in rng.choice([h for h in range(32) if h!=18],size=3,replace=False))
    prior=json.loads((args.out/f'channels_{args.relation}_17.json').read_text())['rows']
    prior={(r['subject'],r['source']):r for r in prior if r['control']==-1 and r['strength']==1}
    rows=[];checks=[]
    cache=pack['cache'];mean=bases['means'][17]
    for i,subject in enumerate(meta['subjects']):
        if i<meta['nval']:continue
        c=cache[subject];foil=cache[info['donors'][subject]]['top']
        clean_heads=capture_components(runner,c,heads=True,blocks=[22])[(22,'heads','answer')]
        clean_score=score_one(runner,c['final'],c['top'],foil)
        for source in RELATIONS:
            q=bases['bases'][17][source]
            state=c['states'][17]
            altered=state-((state-mean)@q)@q.T
            with torch.no_grad():
                ablated=runner.forward(c['ids'],c['pos'],state_patch=(17,altered))[0]
            ablated_score=score_one(runner,ablated,c['top'],foil)
            difference=abs(ablated_score['logp']-prior[(subject,source)]['full']['logp'])
            assert difference<1e-4
            checks.append(dict(subject=subject,source=source,previous_ablation_logp_error=difference))
            def state_hook(_m,_i,out):
                result=out.clone();result[0,c['pos']]=altered.to(out.device);return result
            handle=runner.blocks[16].register_forward_hook(state_hook)
            try:
                for head in [18,*controls]:
                    restored,_=patched_component(runner,c,22,'heads','answer',clean_heads,head=head)
                    rows.append(dict(subject=subject,relation=args.relation,source=source,head=head,
                                     clean=clean_score,ablated=ablated_score,
                                     restored=score_one(runner,restored,c['top'],foil)))
            finally:handle.remove()
        print('mediation',args.relation,subject,flush=True)
    save_json(args.out/f'mediation_{args.relation}.json',dict(complete=True,exploratory=True,
        selected_after_confirmation=True,provenance=provenance(),random_heads=controls,rows=rows,checks=checks,
        note='Restoration tests the path from the erased subject subspace through one head. Post-selection intervals are descriptive.'))


if __name__=='__main__':main()
