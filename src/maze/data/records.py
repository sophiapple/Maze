"""Separate query inputs from exemplar answers and evaluation labels."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from maze.io import read_jsonl


def question_key(question: str) -> str:
    if not isinstance(question,str) or not question.strip():
        raise ValueError('A nonempty question is required')
    normalized=re.sub(r'\s+',' ',unicodedata.normalize('NFKC',question)).strip().casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    choices: tuple[str,...] = ()

    def __post_init__(self):
        if not isinstance(self.id,str) or not self.id.strip():
            raise ValueError('Nonempty query ID required')
        question_key(self.question)
        if not isinstance(self.choices,tuple) or len(self.choices)>26 or any(not isinstance(c,str) or not c.strip() for c in self.choices):
            raise ValueError('choices must be a tuple of nonempty strings (at most 26)')

    @property
    def key(self) -> str:
        return question_key(self.question)

    @classmethod
    def from_dict(cls,value:dict) -> 'Query':
        if not {'id','question'} <= set(value) or not set(value) <= {'id','question','choices'}:
            raise ValueError('Query inputs accept only id, question and choices; keep answers separate')
        return cls(value['id'],value['question'],tuple(value.get('choices',())))

    def text(self) -> str:
        tail='\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(self.choices))
        return self.question + ('\n'+tail if tail else '')


@dataclass(frozen=True)
class Exemplar:
    id: str
    question: str
    answer: str
    rationale: str = ''
    choices: tuple[str,...] = ()

    def __post_init__(self):
        Query(self.id,self.question,self.choices)
        if not isinstance(self.answer,str) or not self.answer.strip() or not isinstance(self.rationale,str):
            raise ValueError('Exemplars need an answer string and optional rationale string')

    @property
    def key(self) -> str:
        return question_key(self.question)

    @classmethod
    def from_dict(cls,value:dict) -> 'Exemplar':
        if not {'id','question','answer'}<=set(value) or not set(value)<={'id','question','answer','rationale','choices'}:
            raise ValueError('Unexpected exemplar fields')
        return cls(value['id'],value['question'],value['answer'],value.get('rationale',''),tuple(value.get('choices',())))

    def query(self) -> Query:
        return Query(self.id,self.question,self.choices)

    def response(self) -> str:
        return (self.rationale.strip()+'\n' if self.rationale.strip() else '')+'Final answer: '+self.answer

    def text(self) -> str:
        return 'Question: '+self.query().text()+'\nAnswer: '+self.response()


def _unique(records):
    if not records or len({x.id for x in records})!=len(records) or len({x.key for x in records})!=len(records):
        raise ValueError('Records must be nonempty with unique IDs and normalized questions')
    return {x.id:x for x in records}


def load_queries(path:str|Path) -> dict[str,Query]:
    return _unique([Query.from_dict(row) for row in read_jsonl(path)])


def load_bank(path:str|Path) -> dict[str,Exemplar]:
    return _unique([Exemplar.from_dict(row) for row in read_jsonl(path)])


def render_messages(system:str, bank:dict[str,Exemplar], path:tuple[str,...], query:Query) -> list[dict]:
    if not isinstance(system,str) or not system.strip() or not path or len(path)!=len(set(path)):
        raise ValueError('Complete system instruction and distinct ordered exemplars are required')
    exemplars=[bank[x] for x in path]
    keys=[e.key for e in exemplars]
    if len(set(keys))!=len(keys) or query.key in keys:
        raise ValueError('Self-exemplar or duplicate-question leakage')
    content='\n\n'.join([e.text() for e in exemplars]+['Question: '+query.text()+'\nAnswer:'])
    return [{'role':'system','content':system},{'role':'user','content':content}]
