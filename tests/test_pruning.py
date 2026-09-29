import unittest

import torch
from timm.models.vision_transformer import VisionTransformer

from similarity_pruning import SimilarityPrunedViT, prune_tokens, CSAPViT
from similarity_pruning.pruning import attention_prune


class PruningTests(unittest.TestCase):
    def test_selection_and_cls_preservation(self):
        x = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]])
        self.assertTrue(torch.equal(prune_tokens(x, 1/3), x[:, [0, 2, 3]]))
        self.assertTrue(torch.equal(prune_tokens(x, 1/3, False), x[:, [0, 1, 2]]))

    def test_ties_and_batch_size(self):
        x = torch.ones(3, 9, 4)
        self.assertEqual(prune_tokens(x, .5).shape, (3, 5, 4))
        self.assertEqual(prune_tokens(x, .999).shape, (3, 2, 4))
        self.assertIs(prune_tokens(x, 0), x)

    def test_invalid_rates(self):
        for rate in [-1, 1, float('nan')]:
            with self.assertRaises(ValueError):
                prune_tokens(torch.ones(1, 3, 2), rate)

    def test_baseline_equivalence_and_pruned_forward(self):
        torch.manual_seed(0)
        base = VisionTransformer(img_size=32, patch_size=8, embed_dim=24,
                                 depth=2, num_heads=3, num_classes=5).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(base(images), SimilarityPrunedViT(base)(images))
            output = SimilarityPrunedViT(base, .5, .5, [0])(images)
        self.assertEqual(output.shape, (2, 5))
        self.assertTrue(torch.isfinite(output).all())


class CascadeTests(unittest.TestCase):
    def test_attention_selection(self):
        x = torch.arange(10.).reshape(1, 5, 2)
        scores = torch.tensor([[.1, .4, .2, .3]])
        self.assertTrue(torch.equal(attention_prune(x, scores, .5), x[:, [0, 2, 4]]))

    def test_cascade_token_counts_and_baseline(self):
        base = VisionTransformer(img_size=32, patch_size=8, embed_dim=24,
                                 depth=2, num_heads=3, num_classes=5).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(base(images), CSAPViT(base)(images))
            counts = []
            handle = base.blocks[1].register_forward_pre_hook(
                lambda module, args: counts.append(args[0].shape[1]))
            try:
                output = CSAPViT(base, .5, .5, [0])(images)
            finally:
                handle.remove()
        self.assertEqual(counts, [5])
        self.assertEqual(output.shape, (2, 5))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
