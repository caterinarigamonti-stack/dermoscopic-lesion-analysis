import unittest

import torch

from lesion_segmentation.model import create_model


class ModelTests(unittest.TestCase):
    def test_task1_architectures_preserve_spatial_shape(self):
        image = torch.zeros((1, 3, 64, 64))
        for architecture in ("unet", "unetplusplus", "deeplabv3plus"):
            with self.subTest(architecture=architecture):
                model = create_model(
                    "task1",
                    encoder_name="resnet34",
                    encoder_weights=None,
                    architecture=architecture,
                )
                model.eval()
                with torch.no_grad():
                    output = model(image)
                self.assertEqual(output.shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
