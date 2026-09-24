"""Release gate: complete full-row evidence and no historical numerical inputs."""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from report_gptj_pythia_diagnosis import scalar_rho

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parent/"paper"
OUT=ROOT/"results/paper_full_evaluation"
BASE=ROOT/"results/all_models_gender_original_continents"
KINDS=["diff","lre","raw_jacobian","logit_lens","differential_logit_lens"]

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dependencies(path,seen=None):
    seen=set() if seen is None else seen
    path=path.resolve()
    if path in seen:return seen
    seen.add(path)
    text=re.sub(r"(?<!\\)%[^\n]*","",path.read_text())
    for child in re.findall(r"\\(?:input|include)\{([^}]+)\}",text):
        p=PAPER/child
        if not p.suffix:p=p.with_suffix('.tex')
        dependencies(p,seen)
    return seen

def main():
    stats=read(OUT/'paper_statistics.json')
    assert stats['neural_baselines_complete'] and stats['attribute_baselines_complete']
    attrs=read(OUT/'attribute_rates.json')
    baselines=read(OUT/'baseline_correlations.json')
    retrieval_path=OUT/'wikipedia/retrieval.json'
    retrieval=read(retrieval_path)
    sources={};checks=[];n_rows=0
    for name,model in stats['models'].items():
        manifest=read(BASE/name/'manifest.json')
        paths=list((OUT/name/'measurements').glob('*.json'))
        assert {p.stem for p in paths}==set(manifest['dataset']) and len(paths)==28
        scores={k:[] for k in KINDS};outcomes=[];attr_scores=[]
        model_rows=0
        assert attrs[name]['retrieval_sha256']==sha(retrieval_path)
        for p in paths:
            d=read(p);old=read(ROOT/d['lre_source']);approved=read(ROOT/d['approved_lens_source'])['lens']
            assert sha(ROOT/d['lre_source'])==d['lre_sha256']
            assert sha(ROOT/d['approved_lens_source'])==d['approved_lens_sha256']
            assert sha(ROOT/'data/relations'/p.name)==d['data_sha256']
            worker=read(OUT/name/'workers'/(d['worker']+'.json'))
            assert worker['model']==manifest['model'] and worker['revision']==manifest['revision']
            for f,digest in worker['source_sha256'].items():assert sha(OUT/name/'provenance'/(digest+'_'+Path(f).name))==digest
            expected_prompt=('\n' if model['newline'] else '')+'{} is known for'
            assert d['lens']['neutral_prompt']==approved['neutral_prompt']==expected_prompt
            samples=read(ROOT/'data/relations'/p.name)['samples']
            actual=d['lens']['rows']
            assert collections.Counter((r['subject'],r['object']) for r in actual)==collections.Counter((r['subject'],r['object']) for r in samples)
            assert len(actual)==d['n_original']==len(samples)
            model_rows+=len(actual)
            for a,b in zip(actual,approved['rows'],strict=True):
                for key in ['subject','object','model_top1_id','model_top1','n_references']:assert a[key]==b[key]
                assert a['subject'] in retrieval['subjects']
                if p.stem=='name_gender':
                    assert a['gender_target_terms']==b['gender_target_terms']
                    for kind in KINDS:assert a['rank_'+kind]==min([a['rank_'+kind+'_literal']]+list(a['gender_ranks_'+kind].values()))
            for kind in KINDS:
                ranks=np.array([r['rank_'+kind] for r in actual])
                aggregate=float(np.median(np.log10(ranks+1)))
                assert abs(aggregate-d['lens']['agg'][kind]['median_lograng'])<1e-12
            ar=attrs[name]['relations'][p.stem]
            assert attrs[name]['readout_sha256'][p.name]==sha(p)
            assert ar['n']==len(actual)==len(ar['rows'])
            for a,b in zip(ar['rows'],actual,strict=True):
                assert (a['subject'],a['object'])==(b['subject'],b['object'])
                assert a['rate_zero_extended']==(a['rate'] if a['rate'] is not None else 0)
                assert 0<=a['rate_zero_extended']<=100
            assert abs(ar['mean_zero_extended']-np.mean([r['rate_zero_extended'] for r in ar['rows']]))<1e-12
            assert ar['undefined_original']==sum(r['rate'] is None for r in ar['rows'])
            if not old['date_audit']['excluded']:
                outcomes.append(old['faithfulness']['best_faith'])
                for kind in KINDS:scores[kind].append(-((approved if kind=='diff' else d['lens'])['agg'][kind]['median_lograng']))
                attr_scores.append(ar['mean_zero_extended'])
            if 'year' in p.stem:
                for r in old['date_audit']['rows']:
                    assert r['ignored']==(r['first_token']=='1')==(r['joint_token']=='1')
                assert old['date_audit']['excluded']==any(r['ignored'] for r in old['date_audit']['rows'])
            sources[str(p.relative_to(ROOT))]=sha(p)
        assert model_rows==9747
        n_rows+=model_rows
        assert len(outcomes)==model['n_relations']
        for kind,x in {**scores,'attribute_rate':attr_scores}.items():
            val=float(spearmanr(x,outcomes).statistic)
            assert abs(val-scalar_rho(x,outcomes))<1e-12
            expected=model['primary']['rho'] if kind=='diff' else baselines[name][kind]['rho']
            assert abs(val-expected)<1e-12
            checks.append(dict(model=name,readout=kind,rho=val,n=len(x)))
    assert len(sources)==196 and n_rows==68229
    inputs=dependencies(PAPER/'lens.tex')|dependencies(PAPER/'shared_baselines.tex')
    for p in inputs:
        rel=str(p.relative_to(PAPER))
        assert not rel.startswith('archive/') and not rel.startswith('generated/shared_baselines/'),rel
        assert p.name not in ['rawmaterial.tex','table_prominence.tex','table_refsets.tex','table_subject_level.tex'],rel
        assert 'Pending full-row' not in p.read_text(),rel
    pdfs={}
    for stem in ['lens','shared_baselines']:
        log=(PAPER/(stem+'.log')).read_text()
        assert not re.search(r'Undefined control sequence|undefined references|Citation .+ undefined|Reference .+ undefined|Overfull \\hbox|! LaTeX Error',log)
        fls=(PAPER/(stem+'.fls')).read_text()
        assert 'INPUT ./generated/shared_baselines/' not in fls and 'INPUT ./archive/' not in fls
        assert 'fig_iclr_agreement.pdf' not in fls
        text=subprocess.check_output(['pdftotext','-layout',str(PAPER/(stem+'.pdf')),'-'],text=True).replace('\u2212','-')
        for name,model in stats['models'].items():assert f"{model['primary']['rho']:.4f}" in text,(stem,name)
        for name,values in baselines.items():
            for kind,value in values.items():assert f"{value['rho']:.4f}" in text,(stem,name,kind)
        for feature,value in stats['corpus_correlations'].items():assert f"{value['rho']:.4f}" in text,(stem,feature)
        assert '??' not in text
        pdfs[stem]=dict(sha256=sha(PAPER/(stem+'.pdf')),bytes=(PAPER/(stem+'.pdf')).stat().st_size)
    result=dict(complete=True,model_relation_records=196,evaluated_rows=n_rows,independently_checked_correlations=checks,active_tex_inputs={str(p.relative_to(PAPER)):sha(p) for p in sorted(inputs)},historical_numerical_inputs_in_compiled_papers=[],pdfs=pdfs,measurement_sha256=sources)
    (OUT/'paper_release_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 196 complete model-relation records, 68,229 rows, 42 independent correlation checks; no historical numerical inputs; both PDFs validated.')

if __name__=='__main__':main()
