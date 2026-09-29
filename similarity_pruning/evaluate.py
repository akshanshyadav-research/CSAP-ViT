"""Deterministic, configuration-driven evaluation on an external ImageFolder."""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.data import create_transform, resolve_model_data_config

from .pruning import SimilarityPrunedViT, CSAPViT

DEFAULTS = dict(model='vit_base_patch16_224.augreg2_in21k_ft_in1k', method='csap',
                embedding_drop=0.0, block_drop=0.0, blocks=[], drop_least_similar=False,
                batch_size=16, workers=0, seed=0, preprocessing='timm', hash_images=False)


def load_config(path):
    config = json.loads(Path(path).read_text()) if path else {}
    merged = dict(DEFAULTS, **config)
    validate_config(merged)
    return merged


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(config, data, class_index, output, device='cpu', checkpoint=None):
    validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing result: {output}')
    # Fail on invalid external data before downloading weights.
    data = Path(data)
    mapping = json.loads(Path(class_index).read_text())
    class_to_index = {}
    for key, values in mapping.items():
        if not isinstance(values, list) or len(values) < 1 or not isinstance(values[0], str):
            raise ValueError('Mapping must have index keys and [synset, label] values')
        if values[0] in class_to_index:
            raise ValueError('Duplicate synset in class mapping')
        class_to_index[values[0]] = int(key)
    dataset = ImageFolder(data)
    missing = set(dataset.classes) - class_to_index.keys()
    if missing:
        raise ValueError(f'Class folders absent from mapping: {sorted(missing)}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = {'pretrained': checkpoint is None}
    if checkpoint is not None:
        kwargs['checkpoint_path'] = str(checkpoint)
    base = timm.create_model(config['model'], **kwargs)
    weight_sha256 = model_digest(base)
    transform_config = resolve_model_data_config(base)
    if config['preprocessing'] == 'timm':
        dataset.transform = create_transform(**transform_config, is_training=False)
    else:
        # Legacy nearest-neighbor coordinates: floor(output_index * input_size / output_size).
        height, width = transform_config['input_size'][1:]
        dataset.transform = transforms.Compose([LegacyNearestResize((height, width)), transforms.ToTensor()])
        transform_config = dict(input_size=(3, height, width), interpolation='nearest-floor',
                                normalization='none', pixel_scale='uint8 / 255')
    labels = torch.tensor([class_to_index[name] for name in dataset.classes], device=device)
    if (labels < 0).any() or (labels >= base.num_classes).any():
        raise ValueError('Class mapping indices outside classifier range')
    samples = []
    for name, local_label in dataset.samples:
        path = Path(name)
        item = dict(path=str(path.relative_to(data)), label=class_to_index[dataset.classes[local_label]],
                    bytes=path.stat().st_size)
        if config['hash_images']:
            item['sha256'] = digest_file(path)
        samples.append(item)
    dataset_sha256 = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    wrapper = CSAPViT if config['method'] == 'csap' else SimilarityPrunedViT
    model = wrapper(base, config['embedding_drop'], config['block_drop'], config['blocks'],
                    not config['drop_least_similar']).to(device).eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['workers'],
                        shuffle=False, generator=torch.Generator().manual_seed(config['seed']))
    correct = top5_correct = total = 0
    token_counts = []
    handles = [block.register_forward_pre_hook(
        lambda module, args, i=i: token_counts.append((i, args[0].shape[1])))
        for i, block in enumerate(base.blocks)]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for images, targets in loader:
                logits = model(images.to(device))
                target_indices = labels[targets.to(device)]
                correct += (logits.argmax(dim=1) == target_indices).sum().item()
                top5_correct += (logits.topk(min(5, logits.shape[1]), dim=1).indices == target_indices[:, None]).any(dim=1).sum().item()
                total += targets.numel()
                # Token counts are fixed by configuration; capture only the first batch.
                for handle in handles:
                    handle.remove()
                handles = []
                print(f'Processed {total}/{len(dataset)} | top1 {100*correct/total:.4f}%', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    result = dict(config=config, correct=correct, total=total, top1_percent=100*correct/total,
                  top5_percent=100*top5_correct/total, token_counts_entering_blocks=token_counts,
                  preprocessing=transform_config, model_weights_sha256=weight_sha256,
                  pretrained_config=base.pretrained_cfg, dataset_manifest_sha256=dataset_sha256,
                  class_index_sha256=digest_file(class_index), elapsed_seconds=time.perf_counter()-started,
                  device=device, python=platform.python_version(), torch=torch.__version__,
                  torchvision=torchvision.__version__, timm=timm.__version__, git_revision=revision,
                  precision='float32', note='New software evaluation; not verified paper reproduction. Timing includes data loading and logging, not an FPGA benchmark.')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix('.dataset.json')
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite dataset manifest: {manifest_path}')
    manifest_path.write_text(json.dumps(samples, indent=2)+'\n')
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


class LegacyNearestResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        array = np.asarray(image)
        height, width = self.size
        rows = np.arange(height) * array.shape[0] // height
        cols = np.arange(width) * array.shape[1] // width
        return array[rows[:, None], cols[None, :]].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--class-index', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Optional local timm-compatible state dictionary; disables download')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--method', choices=['csap','similarity'])
    parser.add_argument('--embedding-drop', type=float)
    parser.add_argument('--block-drop', type=float)
    parser.add_argument('--blocks', nargs='*', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--preprocessing', choices=['timm','legacy-nearest'])
    parser.add_argument('--hash-images', action='store_true', default=None)
    parser.add_argument('--drop-least-similar', action='store_true', default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    # Reuse the same schema validation after command-line overrides.
    validate_config(config)
    result = evaluate(config, args.data, args.class_index, args.output, args.device, args.checkpoint)
    print(json.dumps({key:result[key] for key in ['top1_percent','top5_percent','total','model_weights_sha256']}, indent=2))


def validate_config(config):
    unknown = set(config) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if not isinstance(config['model'], str) or not config['model']:
        raise ValueError('model must be a nonempty model identifier')
    if config['method'] not in {'csap', 'similarity'} or config['preprocessing'] not in {'timm', 'legacy-nearest'}:
        raise ValueError('Invalid method or preprocessing')
    for key in ['embedding_drop', 'block_drop']:
        if type(config[key]) not in (int, float) or not 0 <= config[key] < 1:
            raise ValueError(f'{key} must be in [0,1)')
    for key, minimum in [('batch_size', 1), ('workers', 0), ('seed', 0)]:
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    blocks = config['blocks']
    if not isinstance(blocks, list) or any(type(i) is not int or i < 0 for i in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError('Block indices must be distinct nonnegative integers')
    for key in ['hash_images', 'drop_least_similar']:
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be boolean')


if __name__ == '__main__':
    main()
