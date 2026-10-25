#!/usr/bin/env python3
"""Audit source integrity, preserved implementations, and local documentation links."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','venv','__pycache__','.pytest_cache','runs','build','dist','.DS_Store'}
ROOT_FILES={'README.md','LICENSE','CITATION.cff','THIRD_PARTY_NOTICES.md','CONTRIBUTING.md','SECURITY.md',
            'CHANGELOG.md','MANIFEST.in','pyproject.toml','requirements-cpu.txt','.gitignore','.gitattributes'}
ROOT_DIRS={'src','configs','docs','scripts','tests','examples','evidence','results','provenance','validation','reproduction','.github'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    result=[]
    for path in ROOT.rglob('*'):
        rel=path.relative_to(ROOT)
        if rel.parts[0] in {'data','checkpoints'} or rel.parts[:2]==('configs','local'):continue
        if any(p in EXCLUDED or p.endswith('.egg-info') for p in rel.parts):continue
        if not path.is_file() or path.suffix in {'.pyc','.pyo'}:continue
        # Setuptools adds these at the sdist root; they are packaging metadata,
        # not source files covered by MANIFEST.json. Do not ignore nested names.
        if rel.as_posix() in {'PKG-INFO','setup.cfg'}:continue
        if rel.as_posix()=='MANIFEST.json':continue
        if rel.parts[0] not in ROOT_FILES|ROOT_DIRS:raise ValueError(f'Unexpected root artifact: {rel}')
        result.append(path)
    return sorted(result)


def check_links():
    checked=0
    for path in files():
        if path.suffix != '.md':continue
        text=path.read_text()
        targets=re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)',text)+re.findall(r'(?:src|href)="([^"]+)"',text)
        for target in targets:
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
            local=target.split('#')[0].split('?')[0]
            if local and not (path.parent/local).exists():raise ValueError(f'Broken local link in {path.relative_to(ROOT)}: {target}')
            checked+=1
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-manifest',action='store_true',help='Explicitly refresh after reviewing intentional source changes')
    a=p.parse_args();items=files()
    for path in items:
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        if path.name in {'.env','id_rsa'} or path.suffix in {'.key','.pem','.pkl','.pickle','.pt','.pth'}:
            raise ValueError('Private or executable asset in public source tree')
    preserved=json.loads((ROOT/'provenance/preserved_sources.json').read_text())
    for item in preserved:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Preserved-source mismatch: '+item['path'])
    links=check_links()
    current={path.relative_to(ROOT).as_posix():sha(path) for path in items}
    manifest=ROOT/'MANIFEST.json'
    if a.write_manifest:
        manifest.write_text(json.dumps({'schema':1,'files':current},indent=2)+'\n')
    else:
        expected=json.loads(manifest.read_text())['files']
        if current!=expected:
            changed=sorted(k for k in set(current)|set(expected) if current.get(k)!=expected.get(k))
            raise ValueError('Source manifest mismatch: '+', '.join(changed[:12]))
    print(json.dumps({'state':'passed','files':len(items),'local_documentation_links':links,
                      'preserved_source_files':len(preserved),'mode':'write' if a.write_manifest else 'verify'},indent=2))


if __name__=='__main__':main()
