"""Full-row attribute overlap, with an explicit zero extension for empty filters."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from jlens import model_sweep as sweep
from jlens.geva_baseline import attribute_rate, attribute_tokens

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/paper_full_evaluation"
BASE=ROOT/"results/all_models_gender_original_continents"

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);a=p.parse_args()
    manifest=read(BASE/a.name/'manifest.json')
    paths=list((OUT/a.name/'measurements').glob('*.json'))
    assert {p.stem for p in paths}==set(manifest['dataset'])
    records={p.stem:read(p) for p in paths}
    subjects=sorted({r['subject'] for d in records.values() for r in d['lens']['rows']})
    retrieval_path=OUT/'wikipedia/retrieval.json';retrieval=read(retrieval_path)
    assert set(subjects)<=set(retrieval['subjects'])
    tokenizer=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['revision'],local_files_only=True)
    stop=set(stopwords.words('english'))
    vocab_size=max(len(tokenizer),1+max(t for d in records.values() for row in d['lens']['rows'] for t in row['logit_lens_top50_ids']))
    decoded=tokenizer.batch_decode([[i] for i in range(vocab_size)])
    values={};tops={}
    for d in records.values():
        for row in d['lens']['rows']:
            subject=row['subject'];ids=row['logit_lens_top50_ids']
            # Same subject-layer neutral readout can have numerical top-50
            # ordering changes; evaluate each saved row's exact predictions.
            tops.setdefault(subject,set()).update(ids)
    for i,subject in enumerate(subjects):
        text=''.join(' '+h['text'] for h in retrieval['subjects'][subject] if h['keep'])
        ids=set(tokenizer.encode(text,add_special_tokens=False))
        attrs=attribute_tokens((decoded[t] for t in ids),stop)
        values[subject]=attrs
        if (i+1)%1000==0:print('ATTRIBUTES',a.name,i+1,len(subjects),flush=True)
    result={}
    for rel,d in records.items():
        rows=[]
        for row in d['lens']['rows']:
            ids=row['logit_lens_top50_ids']
            rate,overlap,denom=attribute_rate([decoded[t] for t in ids],values[row['subject']],stop)
            rows.append(dict(subject=row['subject'],object=row['object'],rate=rate,rate_zero_extended=rate if rate is not None else 0.0,overlap=overlap,filtered_prediction_count=denom))
        assert len(rows)==manifest['dataset'][rel]['n_original']
        result[rel]=dict(n=len(rows),mean_zero_extended=float(np.mean([r['rate_zero_extended'] for r in rows])),undefined_original=sum(r['rate'] is None for r in rows),rows=rows)
    assert sum(d['n'] for d in result.values())==9747
    sweep.write(OUT/a.name/'attribute_rates.json',dict(model=a.name,revision=manifest['revision'],n_subjects=len(subjects),relations=result,retrieval_sha256=sha(retrieval_path),readout_sha256={p.name:sha(p) for p in paths},policy='Full evaluation; filtered top-50 overlap; empty filtered prediction set assigned zero explicitly, never dropped.',source_sha256=sha(Path(__file__))))
    print('COMPLETE',a.name,flush=True)

if __name__=='__main__':main()
