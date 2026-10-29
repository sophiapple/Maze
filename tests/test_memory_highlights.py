import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('memory_highlights', ROOT/'scripts/check_memory_highlights.py')
HIGHLIGHTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HIGHLIGHTS)


class MemoryHighlightTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'evidence/highlights/compact_memory.json').read_text())

    def test_public_tables_and_reproduction_scope(self):
        report = HIGHLIGHTS.verify(ROOT)
        self.assertEqual(report['selected_positive_contrasts'], 6)
        self.assertFalse(report['question_level_reconstruction'])
        self.assertFalse(report['intervals_recomputed'])

    def test_duplicate_and_invalid_counts_are_rejected(self):
        duplicate = copy.deepcopy(self.data)
        duplicate['rows'].append(duplicate['rows'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            HIGHLIGHTS.table(duplicate)
        for counts in ([65], [True], []):
            invalid = copy.deepcopy(self.data)
            invalid['rows'][0]['correct_by_conditional_replicate'] = counts
            with self.assertRaises(ValueError):
                HIGHLIGHTS.table(invalid)

    def test_changed_cost_cannot_leave_a_stale_public_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'evidence/highlights').mkdir(parents=True)
            (root/'docs').mkdir()
            changed = copy.deepcopy(self.data)
            changed['rows'][0]['comparator_mean_tokens'] *= 2
            (root/'evidence/highlights/compact_memory.json').write_text(json.dumps(changed))
            for rel in ('README.md', 'docs/COMPACT_MEMORY.md'):
                (root/rel).write_text((ROOT/rel).read_text())
            with self.assertRaisesRegex(ValueError, 'differs from aggregate arithmetic'):
                HIGHLIGHTS.verify(root)


if __name__ == '__main__':
    unittest.main()
