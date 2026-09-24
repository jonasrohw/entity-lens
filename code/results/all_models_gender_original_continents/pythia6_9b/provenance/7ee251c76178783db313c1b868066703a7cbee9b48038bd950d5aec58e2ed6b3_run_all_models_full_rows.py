"""All seven retained models: gender alternatives, original continent references.

Full-row evaluation for the original 28 relations only; eight demonstrations and
references per method; duplicates preserved. The user explicitly requested no
continent balancing. Retain per-model year exclusions and Qwen leading newline.
Reuse the completed OLMo/Qwen8 measurements; other models need new full-row runs.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from jlens.all_row_protocol import all_row_split,row_multiset

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/all_models_gender_original_continents'
DATA=ROOT/'data/relations'
NAME=None
OUT=None
BETAS=[0.5,1.0,1.5,2.0,3.0,5.0]
EXECUTED_SOURCE=Path(__file__).read_bytes()
LABELS={'olmo3_7b':'OLMo 3 7B','gptj':'GPT-J 6B','pythia6_9b':'Pythia 6.9B','mistral7b':'Mistral 7B','gemma2_9b':'Gemma 2 9B','qwen3_8b_base':'Qwen3 8B Base','qwen3_8_27b':'Qwen3.8 27B'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def specs():
    models={m['name']:m for m in read(ROOT/'results/five_model_sweep/models.json')}
    revisions=read(ROOT/'results/five_model_sweep/revisions.json')
    for name in ['gptj','olmo3_7b']:
        d=read(ROOT/f'results/cross_model_audit/raw/targets/{name}/metadata.json')
        models[name]=dict(name=name,model_id=d['model'],lens_layer=15 if name=='gptj' else 17,layers=[4,10,12,15,18,20] if name=='gptj' else [5,11,14,17,20,23])
        revisions[d['model']]=d['revision']
    for m in models.values():m['revision']=revisions[m['model_id']]
    return models


def old_faith_path(rel):
    if NAME in ['gptj','olmo3_7b']:
        folder='olmo' if NAME=='olmo3_7b' else NAME
        suffix='_ts' if rel.startswith('presidents_') else ''
        return ROOT/f'results/{folder}/results/main/faithfulness_{rel}{suffix}.json'
    return ROOT/f'results/five_model_sweep/{NAME}/main/{rel}.json'


def old_faith(rel):
    d=read(old_faith_path(rel))
    return d if NAME in ['gptj','olmo3_7b'] else d['faithfulness']


def prepare():
    global NAME,OUT
    paths=sorted(DATA.glob('*.json'));assert len(paths)==28
    models=specs()
    for name in LABELS:
        NAME=name;OUT=DEST/name;spec=models[name];dataset={}
        for path in paths:
            samples=read(path)['samples'];d={}
            for method in ['lre','lens']:
                split=all_row_split(samples,lens=method=='lens',balanced_landmarks=False)
                d[method+'_split']=dict(zip(['demonstrations','references','evaluation'],split,strict=True))
            original=old_faith(path.stem)
            dataset[path.stem]=dict(**d,n_original=len(samples),data_sha256=sha(path),original_parameters={k:original[k] for k in ['best_layer','best_beta']},original_faith_source=str(old_faith_path(path.stem).relative_to(ROOT)),original_faith_sha256=sha(old_faith_path(path.stem)))
        manifest=dict(name=name,model=spec['model_id'],revision=spec['revision'],layers=spec['layers'],betas=BETAS,lens_layer=spec['lens_layer'],prompt='\n{} is known for' if name.startswith('qwen') else '{} is known for',dtype='float32',attention='eager',gender_alternatives=True,diverse_continent_references=False,n_relations=28,n_rows=sum(d['n_original'] for d in dataset.values()),dataset=dataset,policy='All source rows evaluated, including demonstrations and references; original seeded references; gender alternatives; relation-level standalone-1 year exclusions for primary correlation.')
        p=OUT/'manifest.json'
        if p.exists():assert read(p)==manifest
        else:write(p,manifest)
        if name not in ['olmo3_7b','qwen3_8b_base']:continue
        base=ROOT/('results/olmo_all_rows_28' if name=='olmo3_7b' else 'results/qwen_all_rows_28')
        landmark=ROOT/('results/full_eval_gender_continent_ablation/olmo_original_references' if name=='olmo3_7b' else 'results/qwen_all_rows_no_corrections')
        for rel,plan in dataset.items():
            dest=OUT/'measurements'/f'{rel}.json'
            if dest.exists():continue
            source=base/'measurements'/f'{rel}.json';d=read(source);provenance={str(source.relative_to(ROOT)):sha(source)}
            if rel=='landmark_on_continent':
                for part,key,splitkey in [('faith','faithfulness','lre_split'),('lens','lens','lens_split')]:
                    p=landmark/f'landmark_{part}.json';value=read(p);d[key]=value['measurement'];d[splitkey]=value['split'];provenance[str(p.relative_to(ROOT))]=sha(p)
                d['lre_reference_category_counts']=dict(collections.Counter(x['object'] for x in d['lre_split']['references']))
                d['lens_reference_category_counts']=dict(collections.Counter(x['object'] for x in d['lens_split']['references']))
                d['faithfulness']['original_reference_changed']=False
            for key in ['lre_split','lens_split']:assert d[key]==plan[key]
            assert d['data_sha256']==plan['data_sha256']
            d.pop('worker',None);d['reused_sources']=provenance;d['model']=name;d['manifest_sha256']=sha(OUT/'manifest.json')
            write(dest,d)
    print('Prepared seven models; reused all 56 OLMo/Qwen8 relation measurements.',flush=True)


def date_audit(tokenizer,rel,data,split,trailing):
    if 'year' not in rel:return dict(is_year=False,ignored_rows=[],excluded=False)
    from jlens.splits import first_object_token
    context=''.join(data['prompt_templates'][0].format(x['subject'])+' '+x['object']+'\n' for x in split[0])
    rows=[]
    for i,x in enumerate(split[2]):
        prefix=context+data['prompt_templates'][0].format(x['subject'])+(' ' if trailing else '')
        enc=tokenizer(prefix+('' if trailing else ' ')+x['object'],return_offsets_mapping=True,add_special_tokens=False)
        ids=[tid for tid,(a,b) in zip(enc.input_ids,enc.offset_mapping) if b>len(prefix) and tokenizer.decode([tid]).strip()]
        tid=first_object_token(x['object'],tokenizer,trailing)
        token=tokenizer.decode([tid]).strip();joint=tokenizer.decode([ids[0]]).strip()
        assert (token=='1')==(joint=='1')
        rows.append(dict(index=i,**x,first_token=token,token_id=tid,joint_token=joint,ignored=token=='1'))
    ignored=[r['index'] for r in rows if r['ignored']]
    return dict(is_year=True,ignored_rows=ignored,excluded=bool(ignored),rows=rows,reason='Exclude the affected relation per the user’s explicit relation-level year rule.')


def worker(a):
    import numpy as np
    import torch
    from functools import partial
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from jlens import model_sweep as sweep
    from run_targeted_corrections import TERMS,expanded_ranks
    manifest=read(OUT/'manifest.json');LAYERS=manifest['layers']
    requested=set(a.relations.split(',')) if a.relations else set(manifest['dataset'])
    assert requested<=set(manifest['dataset'])
    pending=[r for r in requested if not (OUT/'measurements'/f'{r}.json').exists()]
    # Largest jobs first; separate smoke run validates the implementation first.
    pending.sort(key=lambda r:(0 if r in {'name_gender','country_capital_city','presidents_birth_year','presidents_election_year'} else 1,-manifest['dataset'][r]['n_original'],r))
    if not pending:return
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['revision'],local_files_only=True)
    print('LOAD',a.worker,'batch',a.batch_size,flush=True)
    kwargs=dict(local_files_only=True,dtype=torch.float32,attn_implementation='eager',device_map={'':'cuda:0'})
    if torch.cuda.device_count()>1:
        kwargs.update(device_map='balanced',max_memory={i:f'{int(torch.cuda.get_device_properties(i).total_memory/2**30)-11}GiB' for i in range(torch.cuda.device_count())})
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],**kwargs).eval().requires_grad_(False)
    assert all(p.device.type=='cuda' for p in model.parameters())
    model.config.use_cache=model.config.get_text_config().use_cache=False
    model.forward=partial(model.forward,use_cache=False,logits_to_keep=1)
    convolution_check=sweep.enable_fast_qwen_convolution(model,tok,manifest['lens_layer'])
    verification=sweep.verify(model,tok,manifest['lens_layer'],a.batch_size)
    prefix_cache_check=sweep.enable_prefix_cache(model,tok,manifest['lens_layer'],a.batch_size)
    source_bytes={str(Path(__file__).relative_to(ROOT)):EXECUTED_SOURCE}
    for p in [Path(sweep.__file__),ROOT/'jlens/model.py',ROOT/'jlens/splits.py',ROOT/'jlens/all_row_protocol.py',ROOT/'jlens/prefix_intervention.py',ROOT/'scripts/run_targeted_corrections.py']:
        source_bytes[str(p.relative_to(ROOT))]=p.read_bytes()
    source_hashes={}
    for name,content in source_bytes.items():
        digest=hashlib.sha256(content).hexdigest();source_hashes[name]=digest
        p=OUT/'provenance'/f'{digest}_{Path(name).name}';p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(content)
    metadata=dict(worker=a.worker,pid=os.getpid(),batch_size=a.batch_size,manifest_sha256=sha(OUT/'manifest.json'),verification=verification,prefix_cache_check=prefix_cache_check,convolution_check=convolution_check,device_map=getattr(model,'hf_device_map',None),source_sha256=source_hashes,torch=torch.__version__,cuda_device=torch.cuda.get_device_name(0))
    write(OUT/'workers'/f'{a.worker}.json',metadata)
    print('VERIFIED',verification,flush=True)
    for rel in pending:
        output=OUT/'measurements'/f'{rel}.json'
        with sweep.measurement_claim(output) as acquired:
            if not acquired or output.exists():continue
            start=time.time();plan=manifest['dataset'][rel];data_path=DATA/f'{rel}.json';data=read(data_path)
            assert sha(data_path)==plan['data_sha256']
            samples=data['samples'];n=len(samples)
            lre=tuple(plan['lre_split'][k] for k in ['demonstrations','references','evaluation'])
            lens_split=tuple(plan['lens_split'][k] for k in ['demonstrations','references','evaluation'])
            assert row_multiset(lre[2])==row_multiset(lens_split[2])==row_multiset(samples)
            trailing=sweep.trailing_space_needed(tok,rel,samples)
            if NAME=='gptj' and rel.startswith('presidents_'):trailing=True
            audit=date_audit(tok,rel,data,lre,trailing)
            print('START',rel,n,flush=True)
            faith_path=OUT/'checkpoints'/f'{rel}_faith.json'
            if faith_path.exists():
                saved=read(faith_path);assert saved['manifest_sha256']==metadata['manifest_sha256'];faith=saved['measurement']
            else:
                predictions=[];original_decode=sweep.decode
                def capture(*args,**kwargs):
                    logits=original_decode(*args,**kwargs);predictions.append(logits.argmax(-1).cpu().tolist());return logits
                sweep.decode=capture
                try:
                    measured=sweep.measure_faith(model,tok,samples,data['prompt_templates'][0],'main',LAYERS,trailing,a.batch_size,None,split_override=lre)
                finally:sweep.decode=original_decode
                targets=[r['model_top1_id'] for r in measured['rows']]
                cells={};idx=0
                for layer in LAYERS:
                    for beta in BETAS:
                        pred=predictions[idx];idx+=1
                        hits=[int(p==t) for p,t in zip(pred,targets,strict=True)]
                        score=float(np.mean(hits));assert score==measured['grid'][str(layer)][str(beta)][0]
                        cells[f'{layer}/{beta}']=dict(layer=layer,beta=beta,faithfulness=score,predictions=pred,hits=hits)
                assert len(predictions)==idx==36
                best=f"{measured['best_layer']}/{float(measured['best_beta'])}"
                original=plan['original_parameters']
                fixed=f"{original['best_layer']}/{float(original['best_beta'])}" if original else None
                checks=[]
                if old_faith_path(rel).exists():
                    old=old_faith(rel)
                    for key,c in cells.items():
                        old_score=old['grid'][str(c['layer'])][str(c['beta'])][0]
                        replay=float(np.mean(c['hits'][16:116]))
                        if replay!=old_score:checks.append(dict(cell=key,old=old_score,replayed=replay))
                faith=dict(n=n,best_key=best,best_faith=cells[best]['faithfulness'],fixed_key=fixed,fixed_faith=cells[fixed]['faithfulness'] if fixed else None,cells=cells,rows=measured['rows'],original_grid_differences=checks,original_reference_changed=False)
                write(faith_path,dict(manifest_sha256=metadata['manifest_sha256'],measurement=faith,worker=a.worker))
            print('FAITH_DONE',rel,faith['best_faith'],round(time.time()-start,1),flush=True)
            term_ids={};term_ranks={};original_decode=sweep.decode
            if rel=='name_gender':
                for word in sum(TERMS.values(),[]):
                    ids=tok.encode(' '+word,add_special_tokens=False)
                    assert len(ids)==1,(word,ids)
                    term_ids[word]=ids[0]
                def capture_terms(*args,**kwargs):
                    logits=original_decode(*args,**kwargs)
                    if not term_ranks:
                        term_ranks.update({word:(logits>logits[:,tid:tid+1]).sum(-1).cpu().tolist() for word,tid in term_ids.items()})
                    return logits
                sweep.decode=capture_terms
            try:
                lens=sweep.measure_lens(model,tok,samples,data['prompt_templates'][0],'main',manifest['lens_layer'],trailing,a.batch_size,None,neutral_prompt=manifest['prompt'],split_override=lens_split)
            finally:sweep.decode=original_decode
            if rel=='name_gender':expanded_ranks(lens,term_ids,term_ranks)
            lens_audit=date_audit(tok,rel,data,lens_split,trailing)
            assert lens_audit['excluded']==audit['excluded']
            audit['lens_rows']=lens_audit.get('rows',[])
            assert len(lens['rows'])==lens['n']==len(faith['rows'])==n
            for row,sample in zip(lens['rows'],lens_split[2],strict=True):
                assert row['subject']==sample['subject'];row['object']=sample['object']
            assert row_multiset([dict(subject=r['subject'],object=r['object']) for r in faith['rows']])==row_multiset(samples)
            record=dict(relation=rel,worker=a.worker,data_sha256=plan['data_sha256'],n_original=n,lre_split=plan['lre_split'],lens_split=plan['lens_split'],trailing_space=trailing,faithfulness=faith,lens=lens,date_audit=audit,elapsed_seconds=time.time()-start,lre_reference_category_counts=dict(collections.Counter(x['object'] for x in lre[1])),lens_reference_category_counts=dict(collections.Counter(x['object'] for x in lens_split[1])))
            write(output,record)
            print('DONE',rel,n,round(time.time()-start,1),flush=True)
    print('WORKER_COMPLETE',a.worker,flush=True)


def report():
    import numpy as np
    from scipy.stats import spearmanr
    global NAME,OUT
    correlations=[];progress={};all_rows=[];grid_changes={}
    for name,label in LABELS.items():
        NAME=name;OUT=DEST/name;manifest=read(OUT/'manifest.json');rows=[];missing=[];excluded=[]
        for rel,plan in manifest['dataset'].items():
            path=OUT/'measurements'/f'{rel}.json'
            if not path.exists():missing.append(rel);continue
            d=read(path);samples=read(DATA/f'{rel}.json')['samples'];n=len(samples)
            assert d['data_sha256']==sha(DATA/f'{rel}.json')==plan['data_sha256']
            assert sha(ROOT/plan['original_faith_source'])==plan['original_faith_sha256']
            assert d['n_original']==n==plan['n_original']
            f,l=d['faithfulness'],d['lens']
            for payload,key in [(f,'lre_split'),(l,'lens_split')]:
                assert d[key]==plan[key]
                assert payload['n']==len(payload['rows'])==n
                assert row_multiset([{k:r[k] for k in ['subject','object']} for r in payload['rows']])==row_multiset(samples)
                assert [r['subject'] for r in payload['rows']]==[r['subject'] for r in plan[key]['evaluation']]
                assert len(plan[key]['references'])==8
            assert len(f['cells'])==36
            for cell in f['cells'].values():
                assert len(cell['predictions'])==len(cell['hits'])==n
                assert cell['hits']==[int(p==r['model_top1_id']) for p,r in zip(cell['predictions'],f['rows'],strict=True)]
                assert cell['faithfulness']==float(np.mean(cell['hits']))
            assert f['best_key']==max(f['cells'],key=lambda k:f['cells'][k]['faithfulness'])
            assert f['best_faith']==f['cells'][f['best_key']]['faithfulness']
            assert [r['hit'] for r in f['rows']]==f['cells'][f['best_key']]['hits']
            assert f['fixed_faith']==f['cells'][f['fixed_key']]['faithfulness']
            original=plan['original_parameters'];assert f['fixed_key']==f"{original['best_layer']}/{float(original['best_beta'])}"
            score=-float(np.median(np.log10([r['rank_diff']+1 for r in l['rows']])))
            assert abs(score+l['agg']['diff']['median_lograng'])<1e-12
            if rel=='name_gender':
                for r in l['rows']:
                    assert r['rank_diff']==min([r['rank_diff_original']]+[x['rank'] for x in r['gender_term_ranks'].values()])
            for r in l['rows']:
                assert r['n_references']==sum(x['subject']!=r['subject'] for x in d['lens_split']['references'])
            audit=d['date_audit']
            if audit['is_year']:
                assert len(audit['rows'])==n
                assert audit['excluded']==any(r['ignored'] for r in audit['rows'])
                assert all(r['ignored']==(r['first_token']==r['joint_token']=='1') for r in audit['rows'])
            if audit['excluded']:excluded.append(rel)
            if 'reused_sources' in d:
                for source,digest in d['reused_sources'].items():assert sha(ROOT/source)==digest
                assert d['manifest_sha256']==sha(OUT/'manifest.json')
            else:
                metadata=read(OUT/'workers'/f"{d['worker']}.json")
                assert metadata['manifest_sha256']==sha(OUT/'manifest.json')
                for source,digest in metadata['source_sha256'].items():assert sha(OUT/'provenance'/f'{digest}_{Path(source).name}')==digest
            rows.append(dict(model=name,relation=rel,n=n,excluded=audit['excluded'],lens_score=score,
                             faithfulness_reselected=f['best_faith'],faithfulness_original=f['fixed_faith'],
                             selected_key=f['best_key'],original_key=f['fixed_key'],source_sha256=sha(path)))
            changes=f.get('original_grid_differences',[])
            if changes:grid_changes.setdefault(name,{})[rel]=changes
        progress[name]=dict(completed=len(rows),expected=28,evaluated_rows=sum(r['n'] for r in rows),expected_rows=9747,missing=missing,excluded_relations=excluded)
        all_rows+=rows
        if missing:continue
        assert sum(r['n'] for r in rows)==9747
        for cohort,scored in [('primary',[r for r in rows if not r['excluded']]),('all_28_years_included',rows)]:
            result=dict(model=name,label=label,cohort=cohort,n_relations=len(scored),n_rows=sum(r['n'] for r in scored),excluded_relations=excluded if cohort=='primary' else [])
            for setting in ['reselected','original']:
                s=spearmanr([r['lens_score'] for r in scored],[r[f'faithfulness_{setting}'] for r in scored])
                result[f'rho_{setting}']=float(s.statistic);result[f'p_{setting}']=float(s.pvalue)
            correlations.append(result)
        write(OUT/'summary.json',dict(complete=True,correlations=[r for r in correlations if r['model']==name],progress=progress[name]))
    summary=dict(complete=all(p['completed']==28 for p in progress.values()),progress=progress,correlations=correlations,original_subset_grid_differences=grid_changes,
                 protocol='Gender alternatives yes; diverse continent references no; all original rows evaluated, eight references per method, unchanged demonstrations; Qwen newline retained; original year policy for primary result.')
    write(DEST/'summary.json',summary)
    for filename,rows in [('correlations.csv',correlations),('relation_scores.csv',all_rows)]:
        if rows:
            with (DEST/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# All models: gender alternatives, original continent references','',f"Status: {'complete' if summary['complete'] else 'in progress'}. Only complete model cohorts receive correlations.",'',
           '| Model | Completed relations | Validated rows |','| --- | ---: | ---: |']
    for name,p in progress.items():lines.append(f"| {LABELS[name]} | {p['completed']}/28 | {p['evaluated_rows']}/9747 |")
    lines+=['','| Model | Cohort | Relations | Scored rows | Rho reselected | Rho original settings |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in correlations:lines.append(f"| {r['label']} | {r['cohort']} | {r['n_relations']} | {r['n_rows']} | {r['rho_reselected']:.8f} | {r['rho_original']:.8f} |")
    lines+=['','All seven retained paper models use the original 28 relation datasets. Every original row is evaluated by both methods, including eight demonstration rows and eight reference rows per method. Duplicates retain their source weight. No row cap, correctness filter or object-token collision filter is applied. Subject-self differential-lens reference comparisons are excluded without excluding queries.',
            '', 'Gender lens scoring accepts the established gender-matched single-token alternatives. Exact model-top-1 LRE faithfulness is unchanged. Both LRE and lens use their original seeded landmark-continent reference sets, per the user’s explicit request to disable diverse-continent selection. Qwen models retain exactly one leading newline in the neutral prompt; other models use the original neutral prompt.',
            '', 'Primary correlations retain the preceding relation-level year policy: exclude the two affected year relations for models whose tokenizer emits standalone 1. Every year row is nevertheless evaluated and retained with a tokenizer audit. The separate all-28 sensitivity includes all year results and requires no additional inference.',
            '', 'The differential lens score is minus median(log10(zero-based target rank + 1)). Reselected LRE maximizes faithfulness over each model’s original six-layer/six-beta grid on the full evaluation set; original settings freeze the previously selected layer/beta. These are overlapping-data, in-sample evaluations, not independent held-out estimates.',
            '', 'OLMo and Qwen 8B are entirely reused from the completed full-row experiments, including their original-reference landmark measurements. The other five models’ older capped/differently split results do not satisfy this evaluation protocol and require new measurements. Model revisions, source hashes, full split rows, numerical checks, grid predictions and per-row lens ranks are retained.',
            '', 'Rebuild: `.venv/bin/python scripts/run_all_models_full_rows.py --report`.', '']
    (DEST/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(complete=summary['complete'],progress=progress,correlations=correlations,models_with_grid_differences=list(grid_changes)),indent=2),flush=True)


def main():
    global NAME,OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--name',choices=LABELS);p.add_argument('--worker');p.add_argument('--relations',default='')
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    if a.prepare:prepare()
    if a.worker:
        assert a.name;NAME=a.name;OUT=DEST/NAME;worker(a)
    if a.report:report()


if __name__=='__main__':main()
