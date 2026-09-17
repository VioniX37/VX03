# Benchmark: custom (20260917-092603)

Machine: Windows-11-10.0.26200-SP0 · 12 logical CPUs · GPU: None · time limit 60.0 s · HiGHS run single-threaded

| instance | rows | cols | nnz | int | reference | ours | HiGHS | rel. err | our time | HiGHS time | ratio | iters / nodes | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| blend | 74 | 83 | 491 | 0 | -30.81214985 | -30.81214985 | - | 5.6e-12 | 3.52 s | - | - | 98 / 0 | match (certified) |
| 25fv47 | 821 | 1571 | 10400 | 0 | 5501.845888 | 5501.845888 | - | 2.4e-12 | 8.54 s | - | - | 2841 / 0 | match (certified) |
| perold | 625 | 1376 | 6018 | 0 | -9380.758077 | -9380.755278 | - | 3.0e-07 | 21.26 s | - | - | 1783 / 0 | match (certified) |
| CVXQP1_S | 50 | 100 | 148 | 0 | - | 11590.71812 | - | - | 1.81 s | - | - | 8 / 0 | mismatch (optimal) |
| gt2 | 29 | 188 | 376 | 188 | 21166 | 21166 | - | 1.0e-14 | 4.63 s | - | - | 3670 / 160 | match (certified) |

**4 / 5 instances solved correctly.**
