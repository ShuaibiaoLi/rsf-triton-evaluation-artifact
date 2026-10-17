"""Shared FP64 comparison policy for the SRC-03 numerical gate."""

ATOL = 1.0e-12
RTOL = 1.0e-10


def compare(actual, expected):
    max_abs = 0.0
    max_rel = 0.0
    failures = 0
    failure_indices = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        absolute = abs(got - want)
        relative = absolute / abs(want) if want != 0.0 else (0.0 if absolute == 0.0 else float("inf"))
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        if absolute > ATOL + RTOL * abs(want):
            failures += 1
            failure_indices.append(index)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "failures": failures,
        "failure_indices": failure_indices,
        "atol": ATOL,
        "rtol": RTOL,
    }
