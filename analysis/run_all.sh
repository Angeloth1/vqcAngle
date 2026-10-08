#!/usr/bin/env bash
# Reproduce every table, log and the manuscript from the result files. From the repository root:
#     bash paper/analysis/run_all.sh
# (verify_refs.py / final_refs.py need network access and are run separately; their output is cached in paper/references/.)
# The experiments themselves (exp1-exp6, exp3b; ~20 min for the new ones) are not re-run here: this script starts from results/.
set -euo pipefail
PY=.venv/bin/python
$PY verify_fastsim.py 2>&1 | grep -v -E "^No (interpret|gradient)" > /dev/null
$PY paper/analysis/analyze_results.py > /dev/null
$PY paper/analysis/extra_contrasts.py > paper/tables/extra_contrasts_log.txt 2>&1
$PY paper/analysis/notebook_recheck.py > /dev/null 2>&1              # stage 1: checkpoints, head-shot recomputation (~2 min)
$PY paper/analysis/notebook_recheck.py --ladder-only > /dev/null 2>&1 # stage 2: angle-arm decomposition (~1.5 min)
$PY paper/analysis/resources.py > /dev/null
$PY paper/analysis/analyze_gapfill.py > /dev/null                    # exp5 / exp6 / exp3b statistics and reproduction checks
$PY paper/analysis/novel_mechanism.py > /dev/null 2>&1               # out-of-distribution diagnostics (~1 min)
$PY paper/analysis/novel_mechanism.py --nn > /dev/null 2>&1
$PY paper/analysis/make_tables.py > /dev/null
$PY paper/analysis/build_manuscript.py
$PY paper/analysis/write_verification_report.py
$PY paper/analysis/build_supplement.py
$PY paper/analysis/audit_numbers.py
