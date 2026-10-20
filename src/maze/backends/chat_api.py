"""A small chat-completions HTTP adapter with no implicit retries."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('API redirects are disabled to avoid forwarding credentials')


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int|None
    output_tokens: int|None
    finish_reason: str
    response_model: str|None = None

    def __post_init__(self):
        if not isinstance(self.text,str) or not isinstance(self.finish_reason,str):
            raise ValueError('Generation needs text and a finish reason')
        for n in (self.input_tokens,self.output_tokens):
            if n is not None and (type(n) is not int or n<0):
                raise ValueError('Token counters must be nonnegative integers or unknown')


class ChatAPI:
    """Compatible with a configured vLLM-style /chat/completions endpoint.

    No provider or key is hardcoded. A timeout is an unresolved physical request,
    not permission to issue the request again. The caller retains that event.
    """
    def __init__(self, base_url:str, *, key_env:str='MAZE_API_KEY', timeout:float=180.):
        parsed=urlparse(base_url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Provide a clean API base URL; credentials belong in an environment variable')
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in {'localhost','127.0.0.1','::1'}):
            raise ValueError('Use HTTPS, except for a loopback inference server')
        if timeout<=0:
            raise ValueError('Positive request timeout required')
        self.endpoint=base_url.rstrip('/')+'/chat/completions'
        self.key=os.environ.get(key_env,'')
        self.timeout=timeout
        self.opener=build_opener(NoRedirect())

    def generate(self, request:dict) -> Generation:
        payload={'model':request['model'],'messages':request['messages'],'n':1,'stream':False,
                 'seed':request['seed'],**request['decoding']}
        allowed={'model','messages','n','stream','seed','temperature','top_p','max_tokens','stop','chat_template_kwargs'}
        if not set(payload)<=allowed:
            raise ValueError('Unrecognized generation setting')
        headers={'Content-Type':'application/json'}
        if self.key:
            headers['Authorization']='Bearer '+self.key
        req=Request(self.endpoint,data=json.dumps(payload,allow_nan=False).encode(),headers=headers,method='POST')
        with self.opener.open(req,timeout=self.timeout) as response:
            data=response.read(8*1024*1024+1)
        if len(data)>8*1024*1024:
            raise ValueError('API response exceeded the safety cap')
        value=json.loads(data)
        if len(value.get('choices',[]))!=1:
            raise ValueError('Exactly one response is required for each physical request')
        choice=value['choices'][0]
        text=choice.get('message',{}).get('content')
        if text is None: text=''
        usage=value.get('usage') or {}
        return Generation(text,usage.get('prompt_tokens'),usage.get('completion_tokens'),
                          str(choice.get('finish_reason') or 'unknown'),value.get('model'))
