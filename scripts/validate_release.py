#!/usr/bin/env python3
"""Run lightweight checks for the public reproducibility release."""

from __future__ import annotations

import ast
import hashlib
import io
import re
import sys
import tokenize
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u4e00-\u9fff]")
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|media|Users)/)")

REQUIRED = (
    "README.md",
    "environment.yml",
    "scripts/bootstrap_workspace.py",
    "pipelines/simulation/run_main_pipeline.py",
    "pipelines/analysis/run_main_analysis.py",
    "pipelines/analysis/build_manuscript_tables.py",
    "pipelines/operating_point/run_operating_point_pipeline.py",
    "pipelines/operating_point/run_operating_point_analysis.py",
    "pipelines/topology_control/run_dp_pipeline.py",
    "pipelines/topology_control/run_three_topology_analysis.py",
    "pipelines/biophysical_robustness/run_biophysical_pipeline.py",
    "pipelines/empirical/run_empirical_pipeline.py",
)


def fail(message: str, failures: list[str]):
    failures.append(message)
    print("[FAIL]", message)


def main():
    failures: list[str] = []

    for relative in REQUIRED:
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"required file missing or empty: {relative}", failures)

    morphology_root = REPOSITORY_ROOT / "data" / "microns" / "morphologies"
    expected_layers = {"L2_3": 20, "L4": 20, "L5": 20, "L6": 20}
    for layer, expected in expected_layers.items():
        observed = len(list((morphology_root / layer).glob("*.swc")))
        if observed != expected:
            fail(f"{layer}: expected {expected} SWCs, found {observed}", failures)

    fixed_inputs = (
        "data/microns/proofread_cells_metadata.csv",
        "data/microns/proofread_cells_synapses.csv",
        "data/microns/proofread_cells_celltypes_all.csv",
        "data/microns/patch/syn_ee.csv",
        "data/microns/patch/syn_ei.csv",
        "data/microns/patch/syn_ie.csv",
        "data/microns/representative_morphologies/parameter.json",
        "data/microns/synapse_segment_mapping/syn_to_segment_L2_3.csv",
    )
    for relative in fixed_inputs:
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"fixed MICrONS input missing or empty: {relative}", failures)

    checksum_manifest = REPOSITORY_ROOT / "data" / "microns" / "CHECKSUMS.sha256"
    if not checksum_manifest.is_file():
        fail("MICrONS checksum manifest is missing", failures)
    else:
        for line_number, line in enumerate(
            checksum_manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                expected_hash, relative = line.split("  ", 1)
            except ValueError:
                fail(f"invalid checksum line {line_number}", failures)
                continue
            path = checksum_manifest.parent / Path(relative)
            if not path.is_file():
                fail(f"checksummed input is missing: {relative}", failures)
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                fail(f"checksum mismatch: {relative}", failures)

    mechanisms = sorted((REPOSITORY_ROOT / "mechanisms").glob("*.mod"))
    if len(mechanisms) != 4:
        fail(f"expected four NMODL source files, found {len(mechanisms)}", failures)

    forbidden_raw = [
        path for path in (REPOSITORY_ROOT / "data").rglob("*")
        if path.is_file() and (
            path.suffix.lower() == ".edf"
            or path.name.lower() in {"participants.tsv", "channels.tsv"}
        )
    ]
    if forbidden_raw:
        fail("raw juvenile-pig EDF/BIDS files must not be redistributed", failures)

    python_files = sorted(REPOSITORY_ROOT.rglob("*.py"))
    for path in python_files:
        relative = path.relative_to(REPOSITORY_ROOT)
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            fail(f"syntax error in {relative}: {exc}", failures)
            continue

        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and HAN.search(token.string):
                fail(f"non-English comment in {relative}:{token.start[0]}", failures)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ABSOLUTE_PATH.search(node.value):
                    fail(f"hard-coded user path in {relative}:{node.lineno}", failures)

    for path in REPOSITORY_ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size >= 100_000_000:
            fail(f"file exceeds GitHub's 100 MB limit: {path.relative_to(REPOSITORY_ROOT)}", failures)

    top_readmes = list(REPOSITORY_ROOT.glob("README*"))
    nested_readmes = [p for p in REPOSITORY_ROOT.rglob("README*") if p.parent != REPOSITORY_ROOT]
    environment_files = list(REPOSITORY_ROOT.rglob("environment.yml"))
    if len(top_readmes) != 1 or nested_readmes:
        fail("the release must contain exactly one top-level README", failures)
    if environment_files != [REPOSITORY_ROOT / "environment.yml"]:
        fail("the release must contain exactly one top-level environment.yml", failures)

    if failures:
        print(f"\nRelease validation failed with {len(failures)} problem(s).")
        sys.exit(1)
    print(f"[PASS] {len(python_files)} Python files parsed successfully")
    print("[PASS] 80 morphology files found (20 per layer)")
    print("[PASS] fixed MICrONS inputs, checksums, and four NMODL sources verified")
    print("[PASS] no raw juvenile-pig EDF/BIDS files included")
    print("[PASS] no non-English comments or user-specific paths found")
    print("[PASS] no file exceeds GitHub's 100 MB limit")
    print("[PASS] exactly one top-level README and environment.yml")


if __name__ == "__main__":
    main()
