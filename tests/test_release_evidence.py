import copy
import json
from pathlib import Path
import tempfile
import unittest
from maze_paper.evidence import verify_vote_records
from maze_paper.reported import reconstruct, extract_tables, write_report

ROOT = Path(__file__).resolve().parents[1]


class VoteEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls=[json.loads(x) for x in (ROOT/'evidence/development/gpqa32_vote_calls.jsonl').read_text().splitlines()]
        cls.rows=[json.loads(x) for x in (ROOT/'evidence/development/outcomes.jsonl').read_text().splitlines()]
        cls.meta=json.loads((ROOT/'evidence/development/gpqa32_vote_reference.json').read_text())

    def test_all_votes_and_reused_request_accounting(self):
        result=verify_vote_records(self.calls,self.rows,self.meta)
        self.assertEqual(result['votes_checked'],128)
        self.assertEqual(result['call_observations'],544)
        self.assertEqual(result['distinct_cached_requests'],487)
        self.assertEqual(result['reused_observations'],57)

    def test_missing_call_or_duplicate_cannot_complete_vote(self):
        for calls in (self.calls[1:],self.calls+[self.calls[0]]):
            with self.assertRaises(ValueError):verify_vote_records(calls,self.rows,self.meta)

    def test_cost_or_grade_corruption_rejected(self):
        for key in ('input_tokens','correct'):
            calls=copy.deepcopy(self.calls)
            calls[0][key] = 1-calls[0][key] if key=='correct' else calls[0][key]+1
            with self.assertRaises(ValueError):verify_vote_records(calls,self.rows,self.meta)

    def test_conflicting_shared_request_rejected(self):
        calls=copy.deepcopy(self.calls);seen=set()
        for r in calls:
            if r['call_id'] in seen:
                r['output_tokens']+=1
                break
            seen.add(r['call_id'])
        else:self.fail('fixture must contain cached request reuse')
        with self.assertRaisesRegex(ValueError,'Conflicting reuse'):
            verify_vote_records(calls,self.rows,self.meta)

    def test_earliest_valid_tie_with_null_and_capped_answer(self):
        # Hand-worked fixture: null abstains; a0 precedes a1, so a0 wins 1:1.
        calls=[dict(cohort='gpqa32_v166',question='q001',arm='Maze_selected_SC3',
            draw=i,call_id=f'c{i}',answer_class=a,correct=c,input_tokens=10,
            output_tokens=20,truncated=t) for i,(a,c,t) in enumerate([(None,0,0),('a0',1,1),('a1',0,0)])]
        row=dict(cohort='gpqa32_v166',question='q001',arm='Maze_selected_SC3',correct=1,
            calls=3,input_tokens=30,output_tokens=60,query_encoder_tokens=5,
            processed_tokens=95,invalid_calls=1,truncated_calls=1)
        meta=dict(arm_calls={'Maze_selected_SC3':3},query_encoder_tokens={'q001':5},expected={
            'call_observations':3,'distinct_cached_requests':3,'reused_observations':0,'votes_checked':1,
            'distinct_request_input_tokens':30,'distinct_request_output_tokens':60,
            'new_generation_calls':0,'independent_regrading':False})
        verify_vote_records(calls,[row],meta)
        calls[1]['draw'],calls[2]['draw']=2,1
        with self.assertRaisesRegex(ValueError,'Vote or cost'):verify_vote_records(calls,[row],meta)


class PaperEvidenceTests(unittest.TestCase):
    def test_complete_source_and_main_table_coverage(self):
        result=reconstruct(ROOT)
        self.assertEqual(result['table_count'],15)
        self.assertEqual(len(result['main_table_values']),105)
        self.assertEqual(len(result['main_table_contrasts']),60)
        self.assertEqual([r['positive_cells'] for r in result['summaries']],[15]*4)
        self.assertFalse(result['full_benchmark_replication'])
        self.assertEqual(result,json.loads((ROOT/'results/paper/summary.json').read_text()))

    def test_paper_outputs_are_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'paper';write_report(reconstruct(ROOT),dest)
            for p in dest.iterdir():self.assertEqual(p.read_bytes(),(ROOT/'results/paper'/p.name).read_bytes())

    def test_wrong_full_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'main.tex';source.write_text('wrong source')
            with self.assertRaisesRegex(ValueError,'bound October 20'):reconstruct(ROOT,source)

    def test_duplicate_table_label_rejected(self):
        s=r'\begin{table}\label{a}\begin{tabular}{cc}\toprule X & 1\\\end{tabular}\end{table}'
        with self.assertRaises(ValueError):extract_tables(s+s)

    def test_changed_cell_rejected_against_excerpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'evidence/paper';p.mkdir(parents=True)
            (p/'table_excerpts.tex').write_bytes((ROOT/'evidence/paper/table_excerpts.tex').read_bytes())
            record=json.loads((ROOT/'evidence/paper/tables.json').read_text())
            record['tables'][0]['rows'][1][2]='100.00'
            (p/'tables.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'transcription'):reconstruct(root)


if __name__=='__main__':unittest.main()
