# RSF Triton/HIP evaluation artifact v1

This artifact supports the paper's three bounded claims: registered-space mapping comparisons, explicit timing semantics, and reversible attribution for the official SeisSol Aging-RSF kernel semantics. TPV101 is application context; this repository does **not** claim a production SeisSol Triton backend.

## Contents

- `results/`: task-level and aggregated structured measurements from BW1000 and Dongfang.
- `raw/`: selected stdout/stderr and failure records for the formal revision jobs.
- `scripts/`: aggregation, validation, and figure-generation code.
- `kernels/`: exact Triton, HIP/C++, FP64 reference, test-vector, and Slurm sources used by the frozen revision experiments.
- `docs/`: frozen protocols, reports, manuscript, figures, and source data.
- `figure_source_manifest.json`: the unique Figure 1–7 mapping; legacy duplicate figure numbers are excluded.
- `SHA256SUMS`: checksum manifest for every released file except itself.

## Verify

```bash
python3 verify_artifact.py
sha256sum -c SHA256SUMS
```

The verifier recomputes the headline job-level medians and checks the frozen claim values. It does not rerun GPU benchmarks. Hardware reruns require the documented DTK/Triton environments and Slurm scripts from the author workspace.

SuiteSparse matrices are not redistributed here. Obtain the named matrices from the SuiteSparse Matrix Collection under their source terms, then verify matrix identity against the dimensions and source hashes recorded in the raw ledgers.

## Evidence boundary

The five independent Slurm array tasks are the experimental units; calls within a task only form a task-level median. P5–P95 intervals are descriptive. No formal equivalence claim is made for stencil or reduction. BW1000 conclusions come from BW1000 measurements; Dongfang results are not substituted for target-platform evidence.

## License

Code is released under the MIT License (`LICENSE-CODE.txt`). Data, figures, and documentation are released under Creative Commons Attribution 4.0 International, CC BY 4.0 (`LICENSE-DATA.txt`). Third-party materials are not relicensed; SuiteSparse matrices are not redistributed.
