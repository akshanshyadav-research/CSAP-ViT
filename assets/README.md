# Paper figure provenance

These are direct rendered extracts from the user-supplied manuscript, **CSAP-ViT: Cascade Similarity–Attention Pruning for Accelerating ViTs on FPGA**, by Akshansh Yadav and Palash Das. No diagram content was redrawn or generated.

- `csap_algorithm.png`: Algorithm 1, PDF page 3, right column.
- `simtrim_dataflow.png`: Figure 4 including caption, PDF page 4, left column.

Both diagrams exclude dataset samples. The complete manuscript is not included in this publication archive.

Extraction with Poppler, from the repository root when the reference PDF is in the parent directory:

```bash
pdftoppm -f 3 -singlefile -scale-to 2800 -x 1000 -y 210 -W 870 -H 1240 -png ../research_paper.pdf assets/csap_algorithm
pdftoppm -f 4 -singlefile -scale-to 2800 -x 120 -y 220 -W 860 -H 710 -png ../research_paper.pdf assets/simtrim_dataflow
```

Source PDF SHA-256: `515085414807367fbfa9b36bfa272cc647f00d98c18ae9e0bb7d9c8c7b25b4c8`.
