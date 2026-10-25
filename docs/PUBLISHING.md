# Maze release workflow

The canonical project is [sophiapple/Maze](https://github.com/sophiapple/Maze). This is one integrated source repository: library, paper references, recorded experiments, documentation, and CI are versioned together. Experiment identities remain stable across software releases.

## Validate a release

```bash
python -m pip install -e '.[train]' build
python scripts/verify_release.py
python -m unittest discover -s tests -v
python -m build
```

Review any intentional source changes before running `python scripts/audit_repository.py --write-manifest`. That command refreshes integrity metadata, not scientific outcomes. The normal verifier rejects a stale manifest.

The source distribution includes all evidence and commands needed for offline reconstruction. Extract it into a clean directory and run `python scripts/verify_release.py` there. The wheel installs the library and CLI; use the repository or source distribution for the bundled experiment records.

## Publish through review

Create a branch in the repository or a fork, commit the reviewed source, and open a pull request against `main`. Review CI and the generated tables before merging. A PR does not imply a tagged release. Keep API credentials, benchmark text, raw responses, private checkpoints, build directories, and research workspaces outside Git.

GitHub Actions uses read-only repository permissions. It verifies recorded outcomes, paper transcription, numerical parity, and the source distribution on CPU. It does not make model API calls. Optional real benchmarks require separately supplied authorized assets and a frozen experiment protocol.

Preserve the code license, model/dataset terms, source attributions, exact experiment identities, all included controls, and cost scope. Publish authorized large assets separately with hashes and configurations when available. Software versioning must not turn historical development data into new independent replication.
