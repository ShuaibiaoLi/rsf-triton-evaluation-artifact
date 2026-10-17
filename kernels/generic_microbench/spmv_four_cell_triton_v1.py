#!/usr/bin/env python3
"""Revision-v1 Triton half of the symmetric SpMV mapping experiment."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def csr_row_program(rowptr, col, val, x, y, rows: tl.constexpr,
                    KMAX: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, CHUNK)
    begin = tl.load(rowptr + row)
    end = tl.load(rowptr + row + 1)
    accumulator = tl.zeros((CHUNK,), tl.float64)
    for base in tl.static_range(0, KMAX, CHUNK):
        position = begin + base + lane
        active = position < end
        column = tl.load(col + position, mask=active, other=0)
        coefficient = tl.load(val + position, mask=active, other=0.0)
        vector = tl.load(x + column, mask=active, other=0.0)
        accumulator += coefficient * vector
    tl.store(y + row, tl.sum(accumulator, axis=0))


@triton.jit
def csr_block_rows(rowptr, col, val, x, y, rows: tl.constexpr,
                   KMAX: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    row = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    lane = tl.arange(0, KMAX)
    row_mask = row < rows
    begin = tl.load(rowptr + row, mask=row_mask, other=0)
    end = tl.load(rowptr + row + 1, mask=row_mask, other=0)
    position = begin[:, None] + lane[None, :]
    active = row_mask[:, None] & (position < end[:, None])
    column = tl.load(col + position, mask=active, other=0)
    coefficient = tl.load(val + position, mask=active, other=0.0)
    vector = tl.load(x + column, mask=active, other=0.0)
    tl.store(y + row, tl.sum(coefficient * vector, axis=1), mask=row_mask)


def parse_mtx(path):
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().lower()
        if not header.startswith("%%matrixmarket matrix coordinate"):
            raise ValueError(f"unsupported MatrixMarket header: {header}")
        symmetric = "symmetric" in header or "hermitian" in header
        for line in handle:
            if line.strip() and not line.startswith("%"):
                rows, cols, declared = map(int, line.split()[:3])
                break
        entries = []
        parsed = 0
        for line in handle:
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            row, column = int(fields[0]) - 1, int(fields[1]) - 1
            value = float(fields[2]) if len(fields) > 2 else 1.0
            entries.append((row, column, value))
            parsed += 1
            if symmetric and row != column:
                entries.append((column, row, value))
    if parsed != declared:
        raise ValueError(f"declared {declared} entries but parsed {parsed}")
    entries.sort(key=lambda item: (item[0], item[1]))
    rowptr = [0] * (rows + 1)
    columns, values = [], []
    for row, column, value in entries:
        rowptr[row + 1] += 1
        columns.append(column)
        values.append(value)
    for row in range(rows):
        rowptr[row + 1] += rowptr[row]
    return rows, cols, rowptr, columns, values


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--mapping", choices=("row_program", "block_row"), required=True)
    parser.add_argument("--block-size", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")

    rows, cols, rowptr, columns, values = parse_mtx(args.matrix)
    lengths = [rowptr[i + 1] - rowptr[i] for i in range(rows)]
    maximum = max(lengths, default=0)
    kmax = 1 if maximum == 0 else 1 << (maximum - 1).bit_length()
    if kmax > 2048:
        raise ValueError(f"max row length {maximum} exceeds registered KMAX=2048")
    requested_rows_per_program = args.block_size // 64
    rows_per_program = 1 if args.mapping == "row_program" else min(requested_rows_per_program, max(1, 2048 // kmax))
    host_x = [math.sin(0.001 * i) + math.cos(0.0003 * i) for i in range(cols)]
    reference = [sum(values[p] * host_x[columns[p]]
                     for p in range(rowptr[row], rowptr[row + 1])) for row in range(rows)]
    device_rowptr = torch.tensor(rowptr, dtype=torch.int32, device="cuda")
    device_columns = torch.tensor(columns, dtype=torch.int32, device="cuda")
    device_values = torch.tensor(values, dtype=torch.float64, device="cuda")
    device_x = torch.tensor(host_x, dtype=torch.float64, device="cuda")
    device_y = torch.empty(rows, dtype=torch.float64, device="cuda")

    def launch():
        if args.mapping == "row_program":
            csr_row_program[(rows,)](
                device_rowptr, device_columns, device_values, device_x, device_y,
                rows=rows, KMAX=kmax, CHUNK=args.block_size, num_warps=args.num_warps)
        else:
            csr_block_rows[(triton.cdiv(rows, rows_per_program),)](
                device_rowptr, device_columns, device_values, device_x, device_y,
                rows=rows, KMAX=kmax, ROWS_PER_PROGRAM=rows_per_program,
                num_warps=args.num_warps)

    launch()
    torch.cuda.synchronize()
    observed = device_y.cpu().tolist()
    max_error = max((abs(a - b) for a, b in zip(observed, reference)), default=0.0)
    scale = max(1.0, max((abs(value) for value in reference), default=0.0))
    tolerance = 1e-12 + 1e-10 * scale
    passed = math.isfinite(max_error) and max_error <= tolerance
    samples = []
    if passed:
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize()
        for _ in range(args.repeat):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); launch(); stop.record(); stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0)

    result = {
        "schema_version": "spmv-four-cell-result/1.0", "backend": "triton",
        "mapping": args.mapping, "matrix": args.matrix.stem, "matrix_path": str(args.matrix),
        "rows": rows, "cols": cols, "nnz": len(values), "kmax": kmax,
        "block_size": args.block_size, "num_warps_or_wave_groups": args.num_warps,
        "chunk_size": args.block_size if args.mapping == "row_program" else None,
        "requested_rows_per_program": requested_rows_per_program if args.mapping == "block_row" else None,
        "rows_per_program": rows_per_program, "warmup": args.warmup, "repeat": args.repeat,
        "max_abs_error": max_error, "tolerance": tolerance,
        "correctness": "PASS" if passed else "FAIL", "samples_us": samples,
        "median_us": statistics.median(samples) if samples else None,
        "p5_us": percentile(samples, 0.05) if samples else None,
        "p95_us": percentile(samples, 0.95) if samples else None,
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "triton": triton.__version__
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
