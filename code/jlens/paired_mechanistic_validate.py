"""Audit complete subject coverage, intervention identities and saved provenance."""

import argparse
import hashlib
import json
from pathlib import Path

from jlens.mechanistic import save_json
from jlens.paired_mechanistic import RELATIONS, LAYERS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    out=args.out
    manifest=json.loads((out/'manifest.json').read_text())
    parts=manifest['relations'][RELATIONS[0]]['parts']
    subjects={k:{x['subject'] for x in v} for k,v in parts.items()}
    assert sum(map(len,subjects.values()))==len(set.union(*subjects.values()))==109
    for relation in RELATIONS:
        for k,v in manifest['relations'][relation]['parts'].items():
            assert {x['subject'] for x in v}==subjects[k]
    counts={}
    identity_errors=[]
    finite_differences=[]
    for relation in RELATIONS:
        for mode in ['original','controlled','unit']:
            d=json.loads((out/f'baseline_{mode}_{relation}.json').read_text())
            assert d['complete'] and set(d['subjects'][d['nval']:])==subjects['test']
            assert set(d['subjects'][:d['nval']])==subjects['validation']
        for stage in ['channels','nonlinear']:
            for layer in LAYERS:
                d=json.loads((out/f'{stage}_{relation}_{layer}.json').read_text());assert d['complete']
                rows=d['rows'];assert {r['subject'] for r in rows}==subjects['test']
                if stage=='channels':
                    keys={(r['subject'],r['source'],r['control'],r['strength']) for r in rows}
                    assert len(keys)==len(rows)==53*2*4*2
                    norms={}
                    for r in rows:
                        norms.setdefault((r['subject'],r['source'],r['strength']),[]).append(r['perturbation_norm'])
                    assert max(max(v)-min(v) for v in norms.values())<1e-4
                else:
                    keys={(r['subject'],r['t']) for r in rows};assert len(keys)==len(rows)==53*4
                    identity_errors += [c['identity_max_error'] for c in d['checks']]
                    finite_differences += [c['finite_difference_relative_error'] for c in d['checks']]
                counts[stage]=counts.get(stage,0)+len(rows)
        for stage in ['routes','heads']:
            allrows=[];allskipped=[]
            for shard in range(4):
                d=json.loads((out/f'{stage}_{relation}_{shard}.json').read_text());assert d['complete']
                allrows+=d['rows'];allskipped+=d['skipped'];identity_errors += [c['identity_max_error'] for c in d['checks']]
            keys={(r['subject'],r['block'],r['kind'],r['group'],r['head']) for r in allrows}
            assert len(keys)==len(allrows)
            expected_subjects=subjects['test'] if stage=='heads' else subjects['test']|subjects['validation']
            assert {r['subject'] for r in allrows+allskipped}==expected_subjects
            for r in allrows:
                expected='validation' if r['subject'] in subjects['validation'] else 'test'
                assert r['partition']==expected
            if stage=='routes':
                core=[r for r in allrows if r['group'] in ['subject_last','suffix','answer']]
                assert len(core)==69*32*2*3
            else:
                assert len(allrows)+len(allskipped)==53*2*32
            counts[stage]=counts.get(stage,0)+len(allrows)
    basis=json.loads((out/'channels_basis.json').read_text())
    assert max(basis['identical_state_max_error'].values())==0
    assert all(d in subjects['train'] for d in basis['donors'].values())
    assert max(identity_errors)<1e-4
    supplementary=json.loads((out/'intermediate_readout.json').read_text())
    assert supplementary['complete']
    assert len(supplementary['rows'])==53*2*32
    counts['intermediate_readouts']=len(supplementary['rows'])
    mediation_errors=[]
    for relation in RELATIONS:
        d=json.loads((out/f'mediation_{relation}.json').read_text())
        assert d['complete'] and d['exploratory'] and d['selected_after_confirmation']
        keys={(r['subject'],r['source'],r['head']) for r in d['rows']}
        assert len(keys)==len(d['rows'])==53*2*4
        assert {r['subject'] for r in d['rows']}==subjects['test']
        mediation_errors += [r['previous_ablation_logp_error'] for r in d['checks']]
        counts['exploratory_head_restoration']=counts.get('exploratory_head_restoration',0)+len(d['rows'])
    assert max(mediation_errors)<1e-4
    source_root=Path(__file__).resolve().parents[1]
    sources=[source_root/'jlens'/name for name in ['paired_mechanistic.py','paired_mechanistic_report.py','paired_readout.py','paired_mediation.py','paired_supplement_report.py','paired_mechanistic_validate.py','mechanistic.py','model.py','splits.py','paired_relation_audit.py']]
    sources += [source_root/'tests'/name for name in ['test_mechanistic.py','test_paired_mechanistic.py','test_paired_mechanistic_report.py']]
    hashes={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    inputs=source_root/'generated/paired_relation_feasibility_20260920'
    input_hashes={name:hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in ['country_pairs.json','paired_manifest.json','sources/countries.json','sources/countryInfo.txt']}
    artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*'))
               if p.is_file() and p.suffix in ['.json','.pt','.md','.pdf','.png'] and p.name!='PROVENANCE.json' and not p.name.endswith('.partial.json')}
    audit=dict(complete=True,counts=counts,confirmation_subjects=53,
               identical_state_max_error=max(basis['identical_state_max_error'].values()),
               identity_patch_max_error=max(identity_errors),finite_difference_max_relative_error=max(finite_differences),
               repeated_ablation_logp_max_error=max(mediation_errors),
               source_sha256=hashes,input_sha256=input_hashes,artifact_sha256=artifacts,
               checks='Paired partitions, held-out coverage, unique intervention rows, donor-training isolation, matched perturbation norms, unchanged-prefix states, identity patches, finite differences.')
    save_json(out/'PROVENANCE.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','artifact_sha256']},indent=2))


if __name__=='__main__':main()
