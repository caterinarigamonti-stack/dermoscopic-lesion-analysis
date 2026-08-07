import unittest

import numpy as np

from scripts.evaluate import (
    average_precision,
    hd95,
    positive_segmentation_metrics,
    roc_auc,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_hd95_identical_masks_is_zero(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        self.assertEqual(hd95(mask, mask), 0.0)

    def test_hd95_one_pixel_translation(self):
        first = np.zeros((32, 32), dtype=bool)
        second = np.zeros((32, 32), dtype=bool)
        first[8:24, 8:24] = True
        second[8:24, 9:25] = True
        self.assertAlmostEqual(hd95(first, second), 1.0)

    def test_presence_ranking_metrics(self):
        labels = np.asarray([0, 1, 0, 1])
        perfect_scores = np.asarray([0.1, 0.8, 0.2, 0.9])
        self.assertAlmostEqual(average_precision(perfect_scores, labels), 1.0)
        self.assertAlmostEqual(roc_auc(perfect_scores, labels), 1.0)

    def test_positive_metrics_exclude_absent_cases(self):
        dice = np.asarray([[0.2, 1.0], [0.8, 0.4], [1.0, 0.6]])
        iou = dice / 2
        presence = np.asarray([[1, 0], [1, 1], [0, 1]])

        per_channel, mean_dice, mean_iou = positive_segmentation_metrics(
            dice,
            iou,
            presence,
            ("first", "second"),
        )

        self.assertAlmostEqual(per_channel["first"]["dice_positive"], 0.5)
        self.assertAlmostEqual(per_channel["second"]["dice_positive"], 0.5)
        self.assertAlmostEqual(mean_dice, 0.5)
        self.assertAlmostEqual(mean_iou, 0.25)


if __name__ == "__main__":
    unittest.main()
