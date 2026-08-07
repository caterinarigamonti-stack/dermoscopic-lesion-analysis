import unittest

import numpy as np
import torch
from PIL import Image

from lesion_segmentation.preprocessing import make_resize_spec, resize_to_canvas
from scripts.evaluate_task1_advanced import (
    binary_overlap,
    parse_ensemble_weights,
    postprocess_mask,
    predict_probabilities,
    restore_binary,
    restore_probability,
)


class PreprocessingTests(unittest.TestCase):
    def test_ensemble_weights_are_normalized(self):
        self.assertEqual(parse_ensemble_weights(None, 2), (0.5, 0.5))
        self.assertEqual(parse_ensemble_weights("1,3", 2), (0.25, 0.75))

    def test_letterbox_preserves_aspect_ratio(self):
        spec = make_resize_spec(400, 200, 384, "letterbox")

        self.assertEqual((spec.content_width, spec.content_height), (384, 192))
        self.assertEqual((spec.content_left, spec.content_top), (0, 96))

    def test_mask_letterbox_uses_zero_padding(self):
        mask = Image.fromarray(np.ones((50, 100), dtype=np.uint8) * 255)
        spec = make_resize_spec(100, 50, 64, "letterbox")

        result = np.asarray(
            resize_to_canvas(mask, spec, Image.Resampling.NEAREST, 0)
        )

        self.assertTrue((result[16:48] == 255).all())
        self.assertTrue((result[:16] == 0).all())
        self.assertTrue((result[48:] == 0).all())

    def test_restore_probability_removes_letterbox_padding(self):
        probability = np.zeros((8, 8), dtype=np.float32)
        probability[2:6] = 0.75

        restored = restore_probability(probability, 10, 5, 0, 2, 8, 4)

        self.assertEqual(restored.shape, (5, 10))
        self.assertAlmostEqual(float(restored.mean()), 0.75, places=5)

    def test_restore_binary_removes_letterbox_padding(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6] = True

        restored = restore_binary(mask, 10, 5, 0, 2, 8, 4)

        self.assertEqual(restored.shape, (5, 10))
        self.assertTrue(restored.all())

    def test_largest_component_removes_small_island(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:10, 2:10] = True
        mask[15:17, 15:17] = True

        result = postprocess_mask(mask, "largest")

        self.assertEqual(int(result.sum()), 64)

    def test_binary_overlap(self):
        target = np.zeros((4, 4), dtype=bool)
        prediction = np.zeros((4, 4), dtype=bool)
        target[:2, :2] = True
        prediction[:2, 1:3] = True

        dice, iou = binary_overlap(prediction, target)

        self.assertAlmostEqual(dice, 0.5)
        self.assertAlmostEqual(iou, 1 / 3)

    def test_flip_tta_restores_orientation(self):
        class CoordinateModel(torch.nn.Module):
            def forward(self, value):
                logits = value[:, :1] * 8 - 4
                return logits

        image = torch.zeros((1, 3, 4, 4))
        image[:, 0, :, 2:] = 1

        prediction = predict_probabilities(CoordinateModel(), image, "flips")

        self.assertGreater(float(prediction[0, 0, 0, 3]), 0.95)
        self.assertLess(float(prediction[0, 0, 0, 0]), 0.05)


if __name__ == "__main__":
    unittest.main()
