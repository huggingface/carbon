import sys
import unittest
from pathlib import Path

TEST_FILE = Path(__file__).resolve()
EVALUATION_DIR = TEST_FILE.parents[1]
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import sequence_recovery_eval as seqrec


class CalculateAccuracyTest(unittest.TestCase):
    def test_fixed_bp_preserves_legacy_denominator(self) -> None:
        metrics = seqrec.calculate_accuracy(
            predictions=["A" * 15],
            labels=["A" * 30],
            accuracy_mode="fixed_bp",
            score_len_bp=30,
        )

        self.assertEqual(metrics, [{"accuracy": 0.5, "scored_bp": 30}])

    def test_prediction_length_scores_generated_window(self) -> None:
        metrics = seqrec.calculate_accuracy(
            predictions=["AACGTT"],
            labels=["AATGTAAC"],
            accuracy_mode="prediction_length",
        )

        self.assertEqual(metrics, [{"accuracy": 4 / 6, "scored_bp": 6}])

    def test_prediction_length_caps_to_label_length(self) -> None:
        metrics = seqrec.calculate_accuracy(
            predictions=["AACCGGTT"],
            labels=["AACC"],
            accuracy_mode="prediction_length",
        )

        self.assertEqual(metrics, [{"accuracy": 1.0, "scored_bp": 4}])

    def test_empty_prediction_scores_zero(self) -> None:
        metrics = seqrec.calculate_accuracy(
            predictions=[""],
            labels=["ACGT"],
            accuracy_mode="prediction_length",
        )

        self.assertEqual(metrics, [{"accuracy": 0.0, "scored_bp": 0}])

    def test_fixed_bp_requires_positive_score_length(self) -> None:
        with self.assertRaises(ValueError):
            seqrec.calculate_accuracy(
                predictions=["ACGT"],
                labels=["ACGT"],
                accuracy_mode="fixed_bp",
                score_len_bp=0,
            )


if __name__ == "__main__":
    unittest.main()
