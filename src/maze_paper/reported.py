"""Source-bound transcription and arithmetic for author-reported paper tables.

This module never substitutes paper values for measured development outcomes.
It requires no model, benchmark text, NumPy, or network connection.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


def clean_tex(value: str) -> str:
    value = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", value)
    value = re.sub(r"\\(?:cmidrule|cline)(?:\([^)]*\))?\{[^}]*\}", "", value)
    value = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:textbf|emph|underline|textit)\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\{\\bf\s+([^}]*)\}", r"\1", value)
    value = value.replace(r"\pm", " ± ").replace(r"\times", "×").replace(r"\%", "%")
    return " ".join(value.replace("$", "").strip().split())


def extract_tables(source: str) -> list[dict]:
    tables = []
    for match in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source, re.S):
        raw = match.group()
        label = re.search(r"\\label\{([^}]+)\}", raw)
        if not label or r"\begin{tabular}" not in raw:
            continue
        if label[1] in {t['label'] for t in tables}:
            raise ValueError("Duplicate paper table label")
        body = raw.split(r"\toprule", 1)[-1].split(r"\end{tabular}", 1)[0]
        rows = [[clean_tex(x) for x in row.split('&')]
                for row in re.split(r"\\\\", body) if '&' in row]
        tables.append({'label': label[1], 'rows': rows,
                       'source_excerpt_sha256': hashlib.sha256(raw.encode()).hexdigest()})
    return tables


def reconstruct(root: str | Path, paper: str | Path | None = None) -> dict:
    root = Path(root)
    record = json.loads((root/'evidence/paper/tables.json').read_text())
    excerpt = root/'evidence/paper/table_excerpts.tex'
    if hashlib.sha256(excerpt.read_bytes()).hexdigest() != record['excerpts_sha256']:
        raise ValueError('Paper excerpt digest mismatch')
    tables = extract_tables(excerpt.read_text())
    if tables != record['tables']:
        raise ValueError('Paper transcription differs from source excerpts')
    if paper is not None:
        raw = Path(paper).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['main_tex_sha256']:
            raise ValueError('Expected the bound October 20 camera-ready source')
        if extract_tables(raw.decode()) != tables:
            raise ValueError('Paper tables differ from supplied full source')
    indexed = {t['label']: t for t in tables}
    rows, contrasts = [], []
    tasks = ['MMLU-Pro', 'GPQA', 'GSM8K', 'MATH500', 'AIME']
    for label, method, controls, calls in (
        ('tab:single_call_results', 'Maze (Ours)', ['Original', 'pFLAT', 'EASE'], [1, 1]),
        ('tab:multi_call_results', 'MultiMaze', ['Majority vote'], [16, 32]),
    ):
        groups = {}
        model = ''
        for row in indexed[label]['rows'][1:]:
            if row[0]:
                model = row[0]
            if not model or len(row) not in (7, 8):
                raise ValueError('Malformed main-table row')
            values = [Decimal(x) for x in row[2:7]]
            if any(x < 0 or x > 100 for x in values):
                raise ValueError('Invalid reported accuracy')
            if row[1] in groups.setdefault(model, {}):
                raise ValueError('Duplicate model/method')
            groups[model][row[1]] = values
            for task, value in zip(tasks, values):
                rows.append({'table': label, 'model': model, 'method': row[1],
                             'benchmark': task, 'accuracy_percent_reported': str(value)})
        for model, methods in groups.items():
            for i, task in enumerate(tasks):
                for control in controls:
                    a, b = methods[method][i], methods[control][i]
                    contrasts.append({'table': label, 'model': model, 'benchmark': task,
                        'method': method, 'control': control,
                        'method_accuracy_pct': str(a), 'control_accuracy_pct': str(b),
                        'gain_pp': str(a-b), 'target_calls': calls[0], 'control_calls': calls[1]})
    if len(rows) != 105 or len(contrasts) != 60:
        raise ValueError('Main-table coverage changed; review the transcription contract')
    summaries = []
    for label, control in [('tab:single_call_results', 'Original'),
                           ('tab:single_call_results', 'pFLAT'),
                           ('tab:single_call_results', 'EASE'),
                           ('tab:multi_call_results', 'Majority vote')]:
        gains = [Decimal(x['gain_pp']) for x in contrasts if x['table']==label and x['control']==control]
        summaries.append({'table': label, 'control': control, 'cells': len(gains),
            'positive_cells': sum(x>0 for x in gains), 'gain_min_pp': str(min(gains)),
            'gain_max_pp': str(max(gains)), 'gain_mean_pp': str(sum(gains)/len(gains))})
    return {'schema': 1, 'evidence_role': 'author_reported_camera_ready',
        'source_main_tex_sha256': record['main_tex_sha256'], 'table_count': len(tables),
        'fresh_generation': False, 'independent_regrading': False,
        'full_benchmark_replication': False, 'main_table_values': rows,
        'main_table_contrasts': contrasts, 'summaries': summaries}


def table(report: dict) -> str:
    lines = ['# Camera-ready results: source-verified transcription', '',
        'These are author-reported paper values. This command verifies transcription and arithmetic; it does not rerun the benchmarks.', '',
        '| Paper comparison | Positive model–task cells | Gain range (pp) | Target calls |',
        '|---|---:|---:|---:|']
    for r in report['summaries']:
        multi = r['table']=='tab:multi_call_results'
        label = ('MultiMaze' if multi else 'Maze')+' vs '+r['control']
        lines.append(f"| {label} | {r['positive_cells']}/{r['cells']} | +{r['gain_min_pp']} to +{r['gain_max_pp']} | {'16 / 32' if multi else '1 / 1'} |")
    lines += ['', 'The 15 cells in each row cover three targets × five benchmarks. Comparisons share cells and are not independent replications.',
        'All 105 main-table values, including random-proposal oracle values, are in `main_results.csv`. Oracle pick is diagnostic and sees a different proposal set from the main MultiMaze policy.',
        'The complete 15 source tables retain matched-pool controls, ranker ablations, variability, latency, transfer, diagnostics, and split accounting in `evidence/paper/`.', '']
    return '\n'.join(lines)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'summary.md').write_text(table(report))
    for name, rows in [('main_results.csv', report['main_table_values']),
                       ('main_contrasts.csv', report['main_table_contrasts'])]:
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
