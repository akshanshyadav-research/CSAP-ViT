import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from similarity_pruning.evaluate import DEFAULTS, evaluate, load_config, validate_config, LegacyNearestResize


class EvaluationTests(unittest.TestCase):
    def test_all_saved_configs(self):
        paths = list((Path(__file__).resolve().parents[1]/'configs').rglob('*.json'))
        self.assertEqual(len(paths), 42)
        for path in paths:
            load_config(path)
        with self.assertRaises(ValueError):
            validate_config(dict(DEFAULTS, blocks=[0, 0]))

    def test_legacy_resize(self):
        values=np.arange(27,dtype=np.uint8).reshape(3,3,3)
        actual=LegacyNearestResize((2,2))(Image.fromarray(values))
        np.testing.assert_array_equal(actual, values[:2,:2])

    def test_complete_evaluation_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for index,synset in enumerate(['n00000001','n00000002']):
                directory=root/'data'/synset;directory.mkdir(parents=True)
                Image.fromarray(np.full((32,32,3),index*100,dtype=np.uint8)).save(directory/'sample.png')
            mapping=root/'mapping.json'
            # Non-contiguous labels explicitly test mapping away from ImageFolder indices.
            mapping.write_text(json.dumps({'1':['n00000001','one'],'3':['n00000002','two']}))
            def small_model(*args, **kwargs):
                model=VisionTransformer(img_size=32,patch_size=8,embed_dim=24,depth=2,num_heads=3,num_classes=5)
                model.pretrained_cfg=dict(input_size=(3,32,32),interpolation='bicubic',mean=(.5,.5,.5),std=(.5,.5,.5),crop_pct=1.)
                return model
            config=dict(DEFAULTS,embedding_drop=.5,block_drop=.5,blocks=[0],batch_size=2,hash_images=True)
            with patch('similarity_pruning.evaluate.timm.create_model',side_effect=small_model):
                first=evaluate(config,root/'data',mapping,root/'first.json')
                second=evaluate(config,root/'data',mapping,root/'second.json')
            self.assertEqual(first['total'],2)
            self.assertEqual(first['token_counts_entering_blocks'],[(0,9),(1,5)])
            for key in ['top1_percent','top5_percent','dataset_manifest_sha256','model_weights_sha256']:
                self.assertEqual(first[key],second[key])
            samples=json.loads((root/'first.dataset.json').read_text())
            self.assertEqual([sample['label'] for sample in samples],[1,3])
            self.assertTrue(all('sha256' in sample for sample in samples))
            with self.assertRaises(FileExistsError):
                evaluate(config,root/'data',mapping,root/'first.json')


if __name__ == '__main__':
    unittest.main()
