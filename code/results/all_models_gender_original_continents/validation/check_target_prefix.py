import torch
from transformers import GPTJConfig,GPTJForCausalLM,AutoTokenizer
from jlens import model_sweep as s
from jlens.batched_prompts import cache_prompts,prompt_targets
from jlens.all_row_protocol import all_row_split
from unittest.mock import patch
torch.set_num_threads(2);torch.manual_seed(19)
tok=AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b',local_files_only=True)
m=GPTJForCausalLM(GPTJConfig(n_embd=32,n_head=4,n_layer=2,rotary_dim=8,vocab_size=50400)).eval().requires_grad_(False)
m._sweep_prompt_batch_size=1;m._sweep_prompt_batch_checks=[];m._sweep_prompt_prefix=True;m._sweep_prompt_prefix_checks=[]
names=['France','Italy','China','Germany','France','United States']
jobs=[(('The capital of Germany is Berlin.\n'*8)+f'{x} is known for',x) for x in names]
actual=cache_prompts(m,tok,jobs,[1]);expected=[s.cache_prompt(m,tok,*j,[1]) for j in jobs]
assert not getattr(m,'_sweep_prompt_prefix_disabled',False)
for a,b in zip(actual,expected):
 assert a['position']==b['position'] and a['target']==b['target']
 torch.testing.assert_close(a['ids'],b['ids'])
 for k in b['states']:torch.testing.assert_close(a['states'][k],b['states'][k])
 torch.testing.assert_close(a['logits'],b['logits'],atol=1e-6,rtol=1e-5)
assert prompt_targets(m,tok,[j[0] for j in jobs])==[x['target'] for x in expected]
with patch('torch.testing.assert_close',side_effect=AssertionError('intentional gate failure')):
 fallback=cache_prompts(m,tok,jobs,[1])
assert m._sweep_prompt_prefix_disabled
for a,b in zip(fallback,expected):assert torch.equal(a['logits'],b['logits'])
assert len(actual)==6
torch.testing.assert_close(actual[0]['logits'],actual[4]['logits'])
print('PASS: scalar/prefix-cached states, logits, targets, row order, duplicates and fallback',flush=True)
m._sweep_prompt_prefix_disabled=False
samples=[dict(subject=f'Person {i}',object='Spain') for i in range(18)]+[dict(subject='Person 0',object='Spain')]
split=all_row_split(samples,lens=False,balanced_landmarks=False)
results=[]
for enabled in [False,True]:
 m._sweep_prompt_prefix=enabled
 f=s.measure_faith(m,tok,samples,'{} lives in','main',[1],False,4,None,split_override=split)
 l=s.measure_lens(m,tok,samples,'{} lives in','main',1,False,4,None,split_override=split)
 results.append((f,l))
a,b=results
assert a[0]['grid']==b[0]['grid']
assert a[0]['rows']==b[0]['rows']
assert a[1]['rows']==b[1]['rows']
assert a[1]['agg']==b[1]['agg']
assert not m._sweep_prompt_prefix_disabled
print('PASS: complete synthetic LRE grids, predictions, ranks and lens aggregates',flush=True)

m._sweep_prompt_prefix_targets_only=True
m._sweep_prompt_prefix_disabled=False
before=len(m._sweep_prompt_prefix_checks)
cached=cache_prompts(m,tok,jobs,[1])
assert len(m._sweep_prompt_prefix_checks)==before
for a,b in zip(cached,expected):assert torch.equal(a["logits"],b["logits"])
assert prompt_targets(m,tok,[j[0] for j in jobs])==[x["target"] for x in expected]
assert len(m._sweep_prompt_prefix_checks)==before+1
print("PASS: target-only prefix cache leaves state forwards scalar",flush=True)
