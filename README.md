# RSF Triton/HIP evaluation artifact v1.0.2

This artifact supports the paper's three bounded claims: registered-space mapping comparisons, explicit timing semantics, and reversible attribution for the official SeisSol Aging-RSF kernel semantics. TPV101 is application context; this repository does **not** claim a production SeisSol Triton backend.

Version `v1.0.2` adds the BW1000 RSF-C3 static loop-bound intervention. For the nine frozen cases only, shortening the fixed-replay bound from 60 to the known maximum of 22 reduced the five-task median Triton/HIP ratio from 3.714 to 2.900. The bound is an input-specific oracle, not a deployable RSF limit; the remaining gap and compiler mechanism are unresolved. The earlier `v1.0.0` and `v1.0.1` tags remain unchanged.

## Contents

- `results/`: task-level and aggregated structured measurements from BW1000 and Dongfang.
- `raw/`: selected stdout/stderr and failure records for the formal revision jobs.
- `scripts/`: aggregation, validation, and figure-generation code.
- `kernels/`: exact Triton, HIP/C++, FP64 reference, test-vector, and Slurm sources used by the frozen revision experiments.
- `docs/`: frozen protocols, reports, manuscript, figures, and source data.
- `figure_source_manifest.json`: the unique Figure 1–7 mapping; legacy duplicate figure numbers are excluded.
- `SHA256SUMS`: checksum manifest for every released file except itself.
- `results/bw1000/revision_v2/ablation/` and `raw/bw1000/revision-v2-rsf-c3-loop-bound-*/`: v3 task-level summaries, five formal jobs, passing and failed smoke records, and independent PMC.

## Verify

```bash
python3 verify_artifact.py
python3 scripts/verify_rsf_c3_loop_bound_v3.py
sha256sum -c SHA256SUMS
```

The verifiers recompute the headline job-level medians, check the frozen claim values, and audit v3 raw samples, reversibility, source hashes, and PMC. They do not rerun GPU benchmarks. Hardware reruns require the documented DTK/Triton environments and Slurm scripts in `kernels/`.

SuiteSparse matrices are not redistributed here. Obtain the named matrices from the SuiteSparse Matrix Collection under their source terms, then verify matrix identity against the dimensions and source hashes recorded in the raw ledgers.

## Evidence boundary

The five independent Slurm array tasks are the experimental units; calls within a task only form a task-level median. P5–P95 intervals are descriptive. No formal equivalence claim is made for stencil or reduction. BW1000 conclusions come from BW1000 measurements; Dongfang results are not substituted for target-platform evidence.

The v3 22-step result applies only to the nine registered RSF inputs with known actual iteration counts. The pre60→bound22→restore60 performance comparison used unprofiled device events; the PMC job was separate and its dispatch time is not used for speedup. The failed first smoke run is retained in `raw/`. No DOI or anonymous reviewer route has been assigned to this GitHub release.

## License

Code is released under the MIT License (`LICENSE-CODE.txt`). Data, figures, and documentation are released under Creative Commons Attribution 4.0 International, CC BY 4.0 (`LICENSE-DATA.txt`). Third-party materials are not relicensed; SuiteSparse matrices are not redistributed.
