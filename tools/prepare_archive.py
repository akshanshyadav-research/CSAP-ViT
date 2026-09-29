"""Build a dataset-free notebook/code/result archive without executing notebooks."""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    target = Path(__file__).resolve().parents[1]
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    manifest, excluded, results = [], [], []
    counts = Counter()
    for path in sorted(source.rglob('*')):
        if not path.is_file() or target in path.parents:
            continue
        rel = path.relative_to(source)
        if any(part in {'.ipynb_checkpoints', '.git', '__pycache__', 'data', 'dataset', 'datasets', 'val', 'val2', 'train'} for part in rel.parts):
            continue
        raw = path.read_bytes()
        record = dict(source=str(rel), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if path.suffix == '.ipynb':
            notebook = json.loads(raw)
            removed = 0
            for cell in notebook.get('cells', []):
                if cell.pop('attachments', None): removed += 1
                outputs = []
                for out in cell.get('outputs', []):
                    if out.get('output_type') == 'stream':
                        outputs.append(out)
                    elif out.get('output_type') in {'display_data', 'execute_result'}:
                        data = out.get('data', {})
                        removed += int(any(k != 'text/plain' for k in data))
                        if 'text/plain' in data:
                            out['data'] = {'text/plain': data['text/plain']}
                            out['metadata'] = {}
                            outputs.append(out)
                    elif out.get('output_type') == 'error':
                        outputs.append(out)
                if 'outputs' in cell: cell['outputs'] = outputs
                cell['metadata'] = {}
                text = ''.join(cell.get('source', []))
                if 'data:image/' in text:
                    raise ValueError(f'Embedded source image needs manual review: {rel}')
            notebook['metadata'] = {key: val for key, val in notebook.get('metadata', {}).items()
                                    if key in {'kernelspec', 'language_info'}}
            destination = target/'experiments'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
            chunks = ['# Exported from experiments/' + str(rel) + '\n# Historical code: review paths and side effects before executing.\n']
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    code = ''.join(cell.get('source', []))
                    chunks.append(f'\n# %% Original cell {index}\n' + shell.transform_cell(code))
            script = '\n'.join(chunks)
            export = target/'python_exports'/rel.with_suffix('.py')
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_text(script)
            try:
                ast.parse(script)
                record['python_syntax'] = 'valid'
            except SyntaxError as error:
                record['python_syntax'] = f'historical syntax error at line {error.lineno}: {error.msg}'
            record.update(kind='notebook', removed_visual_outputs=removed, archived=str(destination.relative_to(target)))
        elif path.suffix == '.txt' or (path.suffix == '' and b'\x00' not in raw):
            destination = target/'experiments'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            record.update(kind='result_text', archived=str(destination.relative_to(target)))
            text = raw.decode('utf-8', errors='replace')
            matches = list(re.finditer(r'Accuracy:\s*([0-9.]+).*?processed_image:\s*([0-9]+)', text))
            if matches:
                final = matches[-1]
                results.append(dict(path=str(destination.relative_to(target)), last_logged_accuracy_percent=final[1],
                                    processed_images=final[2], logged_records=len(matches)))
        elif path.suffix == '.svg' and b'<image' not in raw and b'data:image' not in raw:
            destination = target/'experiments'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            record.update(kind='vector_figure', archived=str(destination.relative_to(target)))
        else:
            record['reason'] = ('Reference paper kept outside public archive' if path.suffix == '.pdf'
                                else 'Raster or embedded-image figure requires review to avoid including dataset samples')
            excluded.append(record)
            continue
        counts[record['kind']] += 1
        manifest.append(record)
    docs = target/'docs'
    docs.mkdir(exist_ok=True)
    (docs/'archive_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (docs/'excluded_files.json').write_text(json.dumps(excluded, indent=2)+'\n')
    with (docs/'logged_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=['path','last_logged_accuracy_percent','processed_images','logged_records'])
        writer.writeheader(); writer.writerows(results)
    families = Counter(Path(item['source']).parts[0] for item in manifest)
    (docs/'archive_index.md').write_text('# Experiment archive\n\nOriginal folder names are preserved. See `archive_manifest.json` for file hashes and Python syntax status.\n\n| Original folder/file | Archived files |\n|---|---:|\n' + '\n'.join(f'| `{name}` | {count} |' for name,count in sorted(families.items()))+'\n')
    print(json.dumps(dict(counts=counts, excluded_files=len(excluded), parsed_logs=len(results),
                          visual_outputs_removed=sum(m.get('removed_visual_outputs',0) for m in manifest),
                          exports_with_syntax_errors=sum(m.get('python_syntax','valid')!='valid' for m in manifest)),indent=2))


if __name__ == '__main__':
    main()
