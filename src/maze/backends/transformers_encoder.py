"""Explicit full-token mean pooling from a separately loaded frozen encoder."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from maze.data import load_bank, load_queries
from maze.features import FeatureStore
from maze.io import sha256


def encode(bank_file:str, queries_file:str, system_file:str, output:str, *, model_name:str,
           revision:str, layers:list[int], start_mode:str='encoded_system_full_qa',
           device:str='cpu', max_tokens:int=8192, dtype:str='float32') -> None:
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    if not revision or not layers or len(layers)!=len(set(layers)) or min(layers)<0:
        raise ValueError('An explicit model revision and distinct hidden-state indices are required')
    if dtype not in {'float32','float16','bfloat16'} or max_tokens<1:
        raise ValueError('Unsupported dtype or context cap')
    bank,queries=load_bank(bank_file),load_queries(queries_file)
    if set(bank)&set(queries) or '__system__' in set(bank)|set(queries):
        raise ValueError('Bank and query IDs must be distinct; __system__ is reserved')
    system=Path(system_file).read_text(encoding='utf-8').strip()
    if not system:
        raise ValueError('Empty system instruction')
    tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision,trust_remote_code=False)
    model=AutoModel.from_pretrained(model_name,revision=revision,trust_remote_code=False,
                                   torch_dtype=getattr(torch,dtype)).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    nodes=[{'id':'__system__','kind':'system','question_key':None}]
    texts=[system]
    for item in bank.values():
        nodes.append({'id':item.id,'kind':'exemplar','question_key':item.key})
        texts.append(item.text() if start_mode=='encoded_system_full_qa' else item.query().text())
    for item in queries.values():
        nodes.append({'id':item.id,'kind':'query','question_key':item.key});texts.append(item.text())
    if start_mode not in {'encoded_system_full_qa','zero_start_question_nodes'}:
        raise ValueError('Unknown start mode')
    arrays=[];tokens={}
    # One complete node at a time makes padding semantics explicit and bounds memory.
    with torch.inference_mode():
        for node,text in zip(nodes,texts):
            inputs=tokenizer(text,return_tensors='pt',truncation=False,add_special_tokens=True)
            length=int(inputs['input_ids'].shape[-1])
            if length>max_tokens:
                raise ValueError(f"Node {node['id']} exceeds {max_tokens} tokens; no silent truncation")
            inputs={k:v.to(device) for k,v in inputs.items()}
            hidden=model(**inputs,output_hidden_states=True,return_dict=True).hidden_states
            if hidden is None or max(layers)>=len(hidden):
                raise ValueError('Requested hidden-state indices are unavailable')
            arr=torch.stack([hidden[j][0].float().mean(dim=0) for j in layers]).cpu().numpy()
            if node['kind']=='system' and start_mode=='zero_start_question_nodes': arr[:]=0
            arrays.append(arr);tokens[node['id']]=length
    recipe={'model':model_name,'revision':revision,'layers':layers,'pooling':'full-token-mean-float32',
            'storage_dtype':'float32','transformers_version':transformers.__version__,
            'start_mode':start_mode,'node_encoding':'standalone_full_qa' if start_mode=='encoded_system_full_qa' else 'standalone_question',
            'system_sha256':sha256(system_file),'bank_sha256':sha256(bank_file),
            'queries_sha256':sha256(queries_file),'node_token_counts':tokens}
    FeatureStore(np.stack(arrays),nodes,recipe).save(output)
