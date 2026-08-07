import unittest

import torch

from lesion_segmentation.losses import dice_loss, lovasz_hinge_loss, task_loss


class LossTests(unittest.TestCase):
    def test_positive_only_dice_ignores_empty_target_channels(self):
        target = torch.zeros((1, 2, 8, 8))
        target[:, 0, 2:6, 2:6] = 1
        logits = torch.full_like(target, -8.0)
        logits[:, 0, 2:6, 2:6] = 8.0
        logits[:, 1] = 8.0

        positive_only = dice_loss(logits, target, positive_only=True)
        all_channels = dice_loss(logits, target, positive_only=False)

        self.assertLess(float(positive_only), 0.01)
        self.assertGreater(float(all_channels), 0.4)

    def test_positive_only_dice_weights_attributes_equally(self):
        target = torch.zeros((3, 2, 8, 8))
        target[:, 0, 2:6, 2:6] = 1
        target[0, 1, 2:6, 2:6] = 1
        logits = torch.full_like(target, -8.0)
        logits[:, 0, 2:6, 2:6] = 8.0

        loss = dice_loss(logits, target, positive_only=True)

        self.assertAlmostEqual(float(loss), 0.47, places=2)

    def test_lovasz_hinge_rewards_correct_segmentation(self):
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 2:6, 2:6] = 1
        correct_logits = torch.where(target.bool(), 8.0, -8.0)
        wrong_logits = -correct_logits

        correct_loss = lovasz_hinge_loss(correct_logits, target)
        wrong_loss = lovasz_hinge_loss(wrong_logits, target)

        self.assertLess(float(correct_loss), 1e-6)
        self.assertGreater(float(wrong_loss), 8.0)

    def test_task1_lovasz_dice_loss_has_components(self):
        target = torch.zeros((1, 1, 8, 8))
        logits = torch.zeros_like(target)
        loss, components = task_loss(
            "task1",
            logits,
            target,
            torch.ones_like(target),
            None,
            torch.zeros((1, 1)),
            task1_loss_name="lovasz_dice",
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("lovasz_loss", components)
        self.assertIn("dice_loss", components)


if __name__ == "__main__":
    unittest.main()
