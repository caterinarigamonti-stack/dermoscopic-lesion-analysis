import unittest

import numpy as np

from lesion_segmentation.constants import ATTRIBUTES
from lesion_segmentation.reporting import (
    attribute_status,
    build_findings_report,
    lesion_size_category,
    validate_report_consistency,
)


class ReportingTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        self.assertEqual(attribute_status(0.40), "absent")
        self.assertEqual(attribute_status(0.41), "uncertain")
        self.assertEqual(attribute_status(0.59), "uncertain")
        self.assertEqual(attribute_status(0.60), "present")

    def test_lesion_size_boundaries(self):
        self.assertEqual(lesion_size_category(0.079), "small")
        self.assertEqual(lesion_size_category(0.08), "moderate")
        self.assertEqual(lesion_size_category(0.25), "moderate")
        self.assertEqual(lesion_size_category(0.251), "large")

    def test_generated_report_is_consistent(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        probabilities = {
            attribute: probability
            for attribute, probability in zip(
                ATTRIBUTES, (0.8, 0.1, 0.5, 0.3, 0.7)
            )
        }
        payload, text = build_findings_report(
            "000001", "val", "baseline", mask, probabilities
        )
        self.assertEqual(validate_report_consistency(payload, text), [])


if __name__ == "__main__":
    unittest.main()
