"""Recalculate reported correlations using standard-library tied ranks."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]

def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    result=[0.0]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        rank=(start+1+end)/2
        for i in order[start:end]:result[i]=rank
        start=end
    return result

def spearman(x,y):
    a,b=ranks(x),ranks(y);ma=statistics.mean(a);mb=statistics.mean(b)
    numerator=math.fsum((u-ma)*(v-mb) for u,v in zip(a,b))
    denominator=math.sqrt(math.fsum((u-ma)**2 for u in a)*math.fsum((v-mb)**2 for v in b))
    return numerator/denominator

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    path=OUT/'independent_correlation_check.json'
    checked=json.loads(path.read_text()) if path.exists() else {}
    names=list(dict.fromkeys(r['model'] for r in summary['correlations']))
    for name in names:
        files=sorted((OUT/name/'measurements').glob('*.json'));assert len(files)==28
        hashes={p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if name in checked and checked[name]['measurement_sha256']==hashes:
            print('UNCHANGED_VERIFIED',name);continue
        rows=[]
        for p in files:
            d=json.loads(p.read_text());f=d['faithfulness'];lens=d['lens']
            assert f['n']==lens['n']==len(lens['rows'])==d['n_original']
            row_ranks=[]
            for r in lens['rows']:
                rank=r['rank_diff']
                if p.stem=='name_gender':
                    assert rank==min([r['rank_diff_original']]+[v['rank'] for v in r['gender_term_ranks'].values()])
                row_ranks.append(rank)
            independent_score=-statistics.median(math.log10(v+1) for v in row_ranks)
            score=-lens['agg']['diff']['median_lograng']
            assert abs(independent_score-score)<1e-12
            means={key:statistics.mean(cell['hits']) for key,cell in f['cells'].items()}
            best=max(means,key=means.get);assert best==f['best_key']
            rows.append(dict(relation=p.stem,n=d['n_original'],excluded=d['date_audit']['excluded'],lens=score,reselected=means[best],original=means[f['fixed_key']]))
        assert sum(r['n'] for r in rows)==9747
        results=[]
        for expected in summary['correlations']:
            if expected['model']!=name:continue
            selected=[r for r in rows if expected['cohort']=='all_28_years_included' or not r['excluded']]
            assert len(selected)==expected['n_relations']
            result=dict(cohort=expected['cohort'],n=len(selected))
            for setting in ['reselected','original']:
                value=spearman([r['lens'] for r in selected],[r[setting] for r in selected])
                assert abs(value-expected['rho_'+setting])<1e-12,(name,setting,value,expected)
                result['rho_'+setting]=value
            results.append(result)
        checked[name]=dict(measurement_sha256=hashes,method='Standard-library Pearson correlation of average tied ranks; row-score and gender-rank checks; no scipy.',correlations=results)
        print('INDEPENDENTLY_VERIFIED',name,results)
    path.write_text(json.dumps(checked,indent=2)+'\n')

if __name__=='__main__':main()
