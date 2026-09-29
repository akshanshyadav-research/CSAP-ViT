# Paper figure provenance

These are direct rendered extracts from the user-supplied manuscript, **CSAP-ViT: Cascade Similarity–Attention Pruning for Accelerating ViTs on FPGA**, by Akshansh Yadav and Palash Das. No diagram content was redrawn or generated.

- `csap_algorithm.png`: Algorithm 1, PDF page 3, right column.
- `csap_pipeline_fig3.png`: Figure 3 including caption, PDF page 3, left column; software inference pipeline.
- `pruning_comparison_fig2.png`: Figure 2 including caption, PDF page 1, right column; Sim-Trim versus ATS pruning examples.
- `simtrim_dataflow.png`: Figure 4, PDF page 4, left column; retained as an asset but not displayed in the main README.

Figures 2 and 3 include the illustrative images printed in the paper and were explicitly requested for the README. Dataset files and the complete manuscript are not included in this publication archive.

Extraction with Poppler, from the repository root when the reference PDF is in the parent directory:

```bash
pdftoppm -f 3 -singlefile -scale-to 2800 -x 1000 -y 210 -W 870 -H 1240 -png ../research_paper.pdf assets/csap_algorithm
pdftoppm -f 3 -singlefile -scale-to 2800 -x 115 -y 215 -W 870 -H 460 -png ../research_paper.pdf assets/csap_pipeline_fig3
pdftoppm -f 1 -singlefile -scale-to 2800 -x 1000 -y 1980 -W 875 -H 500 -png ../research_paper.pdf assets/pruning_comparison_fig2
pdftoppm -f 4 -singlefile -scale-to 2800 -x 120 -y 220 -W 860 -H 710 -png ../research_paper.pdf assets/simtrim_dataflow
```

Source PDF SHA-256: `515085414807367fbfa9b36bfa272cc647f00d98c18ae9e0bb7d9c8c7b25b4c8`.
