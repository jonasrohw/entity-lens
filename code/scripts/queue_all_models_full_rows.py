"""Use free local GPUs for the already authorized seven-model full-row run."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/all_models_gender_original_continents'
PENDING=[('gemma2_9b','main',1,32),('pythia6_9b','main',1,32),
         ('pythia6_9b','scalar_resume',1,8),
         ('pythia6_9b','finish_parallel',1,8),
         ('qwen3_8_27b','full_a',2,64),('qwen3_8_27b','full_b',2,64)]
started=[]


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    pending=list(PENDING)
    launches=json.loads((OUT/'launches.json').read_text())
    pending=[j for j in pending if not any(x['model']==j[0] and x['worker']==j[1] for x in launches)]
    while pending:
        launches=json.loads((OUT/'launches.json').read_text())
        smoke=OUT/'logs/qwen3_8_27b_smoke.log'
        smoke_done=smoke.exists() and smoke.read_text().splitlines()[-1:] == ['WORKER_COMPLETE smoke']
        if smoke_done:
            reserved=set()
            for job in launches:
                proc=Path(f"/proc/{job['pid']}/cmdline")
                try:active=b'run_all_models_full_rows.py' in proc.read_bytes()
                except FileNotFoundError:active=False
                if active:reserved.update(job['gpus'])
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            free=[int(line.split(',')[0]) for line in raw.splitlines() if int(line.split(',')[1])<1000 and int(line.split(',')[0]) not in reserved]
            for name,worker,count,batch in list(pending):
                if len(free)<count:continue
                gpus=free[:count];free=free[count:]
                cmd=[sys.executable,'scripts/run_all_models_full_rows.py','--name',name,'--worker',worker,'--batch-size',str(batch)]
                if name=='qwen3_8_27b':cmd+=['--prompt-prefix-cache']
                logpath=OUT/'logs'/f'{name}_{worker}.log'
                with logpath.open('w') as log:
                    process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus))),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                rec=dict(model=name,worker=worker,gpus=gpus,pid=process.pid,command=cmd,log=str(logpath))
                launches.append(rec);started.append(rec);pending.remove((name,worker,count,batch))
                write(OUT/'launches.json',launches);print('LAUNCHED',name,worker,gpus,process.pid,flush=True)
        write(OUT/'queue.json',dict(pid=os.getpid(),pending=pending,started=started,state='waiting' if pending else 'all_jobs_launched'))
        if pending:time.sleep(5)
    print('ALL_JOBS_LAUNCHED',flush=True)


if __name__=='__main__':main()
