import time,json,subprocess,os
from pathlib import Path
root=Path('results/paper_full_evaluation');small=['olmo3_7b','gptj','mistral7b','pythia6_9b','gemma2_9b','qwen3_8b_base']
while True:
 launches=json.loads((root/'launches.json').read_text())
 exits=all(any(e['name']==n and e.get('exit_code')==0 for e in launches) for n in ['mistral7b','gemma2_9b'])
 if exits and all(len(list((root/n/'measurements').glob('*.json')))==28 for n in small):break
 time.sleep(10)
if len(list((root/'qwen3_8_27b/measurements').glob('*.json')))<28:
 cmd=['.venv/bin/python','scripts/run_paper_full_baselines.py','--name','qwen3_8_27b','--worker','qwen27_extra','--batch-size','64']
 with (root/'logs/qwen27_extra.log').open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONUNBUFFERED':'1'},start_new_session=True)
 (root/'qwen27_extra_launch.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2))
 p.wait()
 (root/'qwen27_extra_exit.json').write_text(json.dumps(dict(exit_code=p.returncode)))
