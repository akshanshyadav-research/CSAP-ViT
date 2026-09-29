# Method and implementation

## Sim-Trim

For patch `t_i` and CLS `c`, compute `s_i = dot(t_i,c) / (norm(t_i)*norm(c))`. Remove patches with the highest scores and always keep CLS. `prune_tokens` uses PyTorch cosine similarity, including its epsilon handling of zero vectors. Positional embeddings are added before pruning, as in the inspected historical cosine notebook.

For `N` patches and drop fraction `r`, remove `floor(N*r)` patches. Stable sorting resolves ties, and selected patches retain their original order. This gives a fixed count across a batch and keeps at least one patch for `0 <= r < 1`.

At the input stage CLS is a learned prefix plus positional embedding: it has not yet aggregated this image through attention. Interpreting high similarity as redundancy is an empirical motivation from the manuscript, not a guarantee that input CLS already captures image-specific semantics.

The manuscript uses threshold notation with a pruning percentage as an example. A cosine threshold is not a percentage. Here configuration values are **drop fractions**, converted to selection by ranking; cosine values are never compared directly with a percentage such as 20.

## Attention-based Token Selection

At selected blocks, calculate normalized Q/K from the block's normalized input. Compute `softmax((q_cls * scale) @ K^T)`, average across heads, and rank patch scores excluding CLS. Keep high-attention patches and CLS. Apply selection after the full transformer block, including attention, MLP, and residual paths.

`CSAPViT` recomputes the CLS attention row while leaving the original timm block unchanged. This incurs extra work and is reference code, not a hardware performance benchmark. Use evaluation mode for inference.

| Model | Depth | Sim-Trim placement | ATS blocks in this implementation |
|---|---:|---|---|
| Small | 12 | Before encoder | 3, 6, 9 |
| Base | 12 | Before encoder | 3, 6, 9 |
| Large | 24 | Before encoder | 6, 12, 18 |

The pre-encoder stage is labeled `-1` in the manuscript. Python block indices are zero-based. A schedule `15/15/15/15` applies 15% removal to remaining patches at each stage; it is not a single 60% removal.

## Paper-to-code map

| Component | Portable implementation | Historical material |
|---|---|---|
| Sim-Trim | `prune_tokens` | `vit_final` and similarity-only families |
| ATS cascade | `CSAPViT`, `attention_prune` | Combined attention/similarity families |
| Similarity/dissimilarity ablations | `SimilarityPrunedViT` | Layer-wise and dissimilar-token experiments |
| Merging, grouped pruning, other distances | Historical exports only | Visualization, merging, weighted aggregation experiments |
| Fixed-point quantization | Historical notebooks only | QPyTorch `FixedPoint(16, 8)` cells |
| FPGA architecture | Manuscript only | No hardware source found in supplied folder |

## Reproducibility differences

- Historical threshold code sometimes uses sequence length including CLS when calculating the drop count. The portable code uses patch count only.
- Legacy threshold comparisons can retain ties and variable token counts; stable rank selection uses a fixed count here.
- Inspected legacy notebooks use nearest-neighbor resizing and `ToTensor` without standard model normalization. The portable CLI defaults to the pretrained model's timm evaluation transform; `legacy-nearest` offers the historical resize/scale for controlled comparisons.
- Some inspected notebooks omit `.eval()` and autograd disabling. The CLI uses both evaluation and inference modes.
- QPyTorch cells simulate quantized weights and inputs in floating-point tensors. They do not establish packed FX16 storage or quantization after every intermediate activation.
- Exact historical weight tags, dataset split, dependency versions, and a paper-table-to-log mapping were not established. Model aliases and pretrained weight configurations must be controlled for exact reproduction.
- Tested scope: ordinary single-CLS timm ViTs with token pooling. Distilled models, register tokens, alternative attention modules, and training are outside this implementation's tested scope.

Nine synthetic tests check selection identity/order, CLS preservation, ties, invalid rates, zero-pruning equivalence, and cascade token counts. They do not verify ImageNet accuracy, quantization, GOP counts, power, or throughput.
