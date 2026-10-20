"""Small, explicit file formats for portable research artifacts."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def identity(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def read_jsonl(path: str | Path) -> list[dict]:
    values = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            values.append(value)
    return values


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)


def write_jsonl(path: str | Path, values: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json_bytes(value).decode("utf-8"))


def safe_member(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or root not in path.parents:
        raise ValueError("Asset paths must stay inside their bundle")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def append_durable(path: str | Path, value: dict) -> None:
    """Append and fsync a request event before proceeding to another request."""
    with Path(path).open("ab") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
