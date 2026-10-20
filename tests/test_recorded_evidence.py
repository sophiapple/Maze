import copy
import json
from pathlib import Path
import unittest
from maze_paper.evidence import aggregate,verify_reference
from maze.results import reconstruct
ROOT=Path(__file__).resolve().parents[1]


class RecordedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(x) for x in (ROOT/'evidence/development/outcomes.jsonl').read_text().splitlines()]
        cls.ref=json.loads((ROOT/'evidence/development/reference.json').read_text())

    def test_all_included_arms_match_reference(self):
        self.assertEqual(len(self.rows),789)
        v=aggregate(self.rows)
        self.assertEqual(sum(map(len,v.values())),10)
        self.assertTrue(verify_reference(v,self.ref))

    def test_retains_every_question_in_each_included_arm(self):
        values=aggregate(self.rows)
        self.assertTrue(all(x['questions']==99 for x in values['gpqa99_v206'].values()))
        self.assertTrue(all(x['questions']==32 for x in values['mmlu32_v160'].values()))
        self.assertEqual(values['gpqa99_v206']['Maze_SC3']['correct'],71)
        self.assertEqual(values['mmlu32_v160']['learned_best_SC3']['correct'],25)
        self.assertTrue(any(r['correct']==0 for r in self.rows))
        self.assertTrue(any(r['invalid_calls']>0 for r in self.rows))

    def test_missing_question_and_duplicate_rejected(self):
        for rows in (self.rows[1:],self.rows+[self.rows[0]]):
            with self.assertRaises(ValueError):aggregate(rows)

    def test_outcome_schema(self):
        for field,value in (('correct',.5),('prediction','A'),('calls',0),('input_tokens',True)):
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.assertRaises(ValueError):aggregate(rows)

    def test_token_accounting_checked(self):
        rows=copy.deepcopy(self.rows);rows[0]['processed_tokens']+=1
        with self.assertRaises(ValueError):aggregate(rows)

    def test_changed_correctness_detected(self):
        rows=copy.deepcopy(self.rows);rows[0]['correct']=1-rows[0]['correct']
        with self.assertRaises(ValueError):verify_reference(aggregate(rows),self.ref)

    def test_missing_included_comparator_rejected(self):
        rows=[r for r in self.rows if r['arm']!='Raw_MV3']
        with self.assertRaises(ValueError):verify_reference(aggregate(rows),self.ref)

    def test_seven_exact_comparisons(self):
        report=reconstruct(ROOT)
        expected={'gpqa-one-vs-raw':'1/33','gpqa-one-vs-mv2':'2/99',
                  'gpqa-sc3-vs-mv3':'5/99','gpqa-sc3-vs-mv4':'2/99','gpqa-sc3-vs-mv8':'2/99',
                  'mmlu-sc3-vs-retrieval':'1/32','mmlu-sc3-vs-random':'1/8'}
        self.assertEqual({r['id']:r['gain_exact'] for r in report['comparisons']},expected)
        self.assertFalse(report['fresh_generation']);self.assertFalse(report['independent_regrading'])
        published=json.loads((ROOT/'results/reproduced/summary.json').read_text())
        self.assertEqual(report,published)
