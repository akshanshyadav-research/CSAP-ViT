# Results and provenance

## Manuscript-reported results

Source: the user-supplied **CSAP-ViT: Cascade Similarity–Attention Pruning for Accelerating ViTs on FPGA**, Table I. Values below are transcribed, not recomputed or newly evaluated.

| Model | Schedule | Top-1 (%) | GOPs | Reported GOP savings (%) |
|---|---|---:|---:|---:|
| Small | Baseline | 78.01 | 4.603 | 0 |
| Small | 15/15/15/15 | 76.32 | 3.16 | 31.34 |
| Base | Baseline | 83.32 | 17.571 | 0 |
| Base | 15/15/15/15 | 83.08 | 12.16 | 30.80 |
| Base | 20/15/15/15 | 82.72 | 11.47 | 34.73 |
| Base | 30/25/25/25 | 81.29 | 8.79 | 50.00 |
| Large | Baseline | 84.60 | 61.57 | 0 |
| Large | 15/15/15/15 | 83.85 | 40.87 | 33.55 |
| Large | 30/25/25/25 | 81.64 | 28.71 | 53.36 |

83.32% to 83.08% is a **0.24 percentage-point** accuracy loss. Reported savings are preserved as printed; rounding of GOP values can produce small recalculation differences.

The manuscript describes ZCU104 at 250 MHz with FX16, Vitis HLS 2023.2, and Vivado. It reports 4.05 W, up to 21.30 FPS, and 5.25 FPS/W. No HLS/RTL, synthesis project, bitstream, or hardware measurement artifacts were present, so this software archive cannot reproduce those hardware results.

## Historical logs

`logged_results.csv` extracts the **last matching record per file** with `Accuracy:` and `processed_image:` fields. It retains the source path and image count. This is not necessarily a completed run: periodic logging can omit final images, files may append multiple runs, and experiments can cover different subsets.

The CSV does not rank configurations or assert correspondence to the manuscript table. Other formats, summaries, duplicates, incomplete logs, and empty files remain in `experiments/` even when not represented in the CSV.

## Archive transformations

`archive_manifest.json` stores each included original source file's SHA-256 and size. Notebook source, execution counts, and textual output are retained; rich visual outputs and attachments are removed from publication copies. Python exports transform IPython cells without executing the experiments. Syntax status is recorded in the manifest.

`excluded_files.json` records the reference PDF and unreviewed raster/embedded-raster figures. This conservative selection prevents embedded dataset samples from being published. Original files remain unchanged outside the publication folder.
