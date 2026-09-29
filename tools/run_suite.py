"""Run a baseline and configured pruning experiments, then write a CSV and plot."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['small','base','large'], default='base')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--class-index', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--all', action='store_true', help='Baseline, Sim-Trim, and all 12 paper pruning schedules')
    parser.add_argument('--preprocessing', choices=['timm','legacy-nearest'], default='timm')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a new output directory to avoid mixing runs')
    root = Path(__file__).resolve().parents[1]
    folder = root/'configs'/args.size
    paths = ([folder/'baseline.json', folder/'simtrim_15.json'] + sorted(folder.glob('csap_*.json')) if args.all
             else [folder/'baseline.json', folder/'simtrim_15.json', folder/'csap_15_15_15_15.json'])
    args.output_dir.mkdir(parents=True)
    rows = []
    for config in paths:
        result_path = args.output_dir/(config.stem+'.json')
        cmd = [sys.executable, '-m', 'similarity_pruning.evaluate', '--config', str(config),
               '--data', str(args.data.resolve()), '--class-index', str(args.class_index.resolve()),
               '--output', str(result_path.resolve()), '--device', args.device,
               '--batch-size', str(args.batch_size), '--preprocessing', args.preprocessing]
        if args.checkpoint:
            cmd.extend(['--checkpoint', str(args.checkpoint.resolve())])
        subprocess.run(cmd, cwd=root, check=True)
        result = json.loads(result_path.read_text())
        rows.append(dict(configuration=config.stem, top1_percent=result['top1_percent'],
                         top5_percent=result['top5_percent'], images=result['total'],
                         accuracy_loss_pp=rows[0]['top1_percent']-result['top1_percent'] if rows else 0.,
                         dataset_sha256=result['dataset_manifest_sha256'], weights_sha256=result['model_weights_sha256']))
        if any(rows[0][key] != rows[-1][key] for key in ['dataset_sha256','weights_sha256']):
            raise RuntimeError('Dataset or model weights changed during suite; comparison is invalid')
        with (args.output_dir/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(max(8,len(rows)*.6),5))
    axes.bar([row['configuration'] for row in rows],[row['top1_percent'] for row in rows])
    axes.set_ylabel('Measured top-1 accuracy (%)')
    axes.set_title(f'CSAP-ViT {args.size}: new evaluation ({rows[0]["images"]} images)')
    axes.tick_params(axis='x',rotation=60)
    figure.tight_layout();figure.savefig(args.output_dir/'accuracy.png',dpi=180);plt.close(figure)
    print(f'Results: {args.output_dir}/summary.csv')


if __name__ == '__main__':
    main()
