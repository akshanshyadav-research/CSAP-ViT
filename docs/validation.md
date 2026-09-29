# Validation performed during preparation

- Python 3.10; PyTorch 2.3.1; torchvision 0.18.1; timm 1.0.28.
- Nine unit/integration tests passed on CPU, including a repeatable end-to-end evaluation on temporary synthetic images with explicit non-contiguous class mapping.
- All 42 saved configurations passed schema validation.
- All 1,508 historical Python exports passed syntax parsing; historical experiments were not executed.
- The new quickstart notebook's code cells executed successfully.
- Evaluation and suite CLI help commands completed successfully.
- Publication copies contain no notebook rich outputs/attachments and no file at or above 100 MiB.
- A targeted scan found no GitHub-token, AWS-access-key, or private-key-header patterns. This is a limited pattern scan, not a comprehensive security audit.
- Extracted manuscript diagrams were visually checked for completeness.

Not performed: full ImageNet evaluation, pretrained-weight download, GPU verification, QPyTorch/FX16 reproduction, FPGA synthesis/deployment, or validation of the paper's measured accuracy/GOPs/power/throughput. GitHub publication requires authenticated access.
