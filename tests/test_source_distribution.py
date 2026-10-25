import importlib.util
from pathlib import Path
import tempfile
import unittest


class SourceDistributionAuditTests(unittest.TestCase):
    def test_generated_metadata_is_distinct_from_source(self):
        path=Path(__file__).resolve().parents[1]/'scripts/audit_repository.py'
        spec=importlib.util.spec_from_file_location('audit_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            module.ROOT=Path(tmp)
            (module.ROOT/'README.md').write_text('source')
            for name in ('PKG-INFO','setup.cfg','.DS_Store'):(module.ROOT/name).write_text('generated')
            self.assertEqual([p.name for p in module.files()],['README.md'])
            (module.ROOT/'unexpected.py').write_text('source outside contract')
            with self.assertRaises(ValueError):module.files()


if __name__=='__main__':unittest.main()
