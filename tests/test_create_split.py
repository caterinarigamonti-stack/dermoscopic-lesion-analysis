import unittest

from scripts.create_split import create_split


def synthetic_rows(count: int = 200):
    rows = []
    for index in range(count):
        rows.append(
            {
                "image_id": f"{index:06d}",
                "labels": (
                    int(index % 2 == 0),
                    int(index % 5 == 0),
                    int(index % 11 == 0),
                    int(index % 7 == 0),
                    int(index % 3 == 0),
                ),
                "lesion_ratio": 0.03 + (index % 40) / 100.0,
                "lesion_size": (
                    "small"
                    if index % 3 == 0
                    else "moderate"
                    if index % 3 == 1
                    else "large"
                ),
            }
        )
    return rows


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_exact(self):
        rows = synthetic_rows()
        first = create_split(rows, val_fraction=0.2, seed=17)
        second = create_split(rows, val_fraction=0.2, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("val"), 40)
        self.assertEqual(list(first.values()).count("train"), 160)
        self.assertEqual(set(first), {row["image_id"] for row in rows})


if __name__ == "__main__":
    unittest.main()
