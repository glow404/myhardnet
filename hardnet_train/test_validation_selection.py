from __future__ import annotations

import unittest

import torch

from hardnet_train.checkpoint_selection import checkpoint_selection_metrics
from hardnet_train.metrics import in_batch_descriptor_validation_metrics


class CheckpointSelectionTests(unittest.TestCase):
    def test_selection_uses_only_four_overall_validation_metrics(self) -> None:
        metrics = {
            "loss": 0.6,
            "pos_mean": 0.2,
            "pos_p95": 0.4,
            "fpr_at_tpr95": 0.2,
            "neg_mean": 0.7,
        }

        result = checkpoint_selection_metrics(metrics, {"descriptor_metric": "l2"})

        self.assertEqual(set(result), {"checkpoint_selection_score"})
        self.assertAlmostEqual(result["checkpoint_selection_score"], 0.69)

        improvements = (
            {"loss": 0.3},
            {"neg_mean": 0.9},
            {"fpr_at_tpr95": 0.1},
            {"pos_p95": 0.2},
        )
        for improvement in improvements:
            improved_metrics = {**metrics, **improvement}
            improved = checkpoint_selection_metrics(
                improved_metrics,
                {"descriptor_metric": "l2"},
            )
            self.assertGreater(
                improved["checkpoint_selection_score"],
                result["checkpoint_selection_score"],
            )


class ValidationResponsibilityTests(unittest.TestCase):
    def _metrics(self, selected_distance: float) -> dict[str, float]:
        return in_batch_descriptor_validation_metrics(
            positive_dist=torch.tensor([0.2, 0.2]),
            candidate_dist=torch.tensor(
                [
                    [0.1, 0.7, 0.8, 0.9],
                    [0.6, 0.7, 0.8, 0.9],
                ]
            ),
            candidate_mask=torch.ones((2, 4), dtype=torch.bool),
            selected_negative_dist=torch.full((2, 1), selected_distance),
            selected_negative_mask=torch.ones((2, 1), dtype=torch.bool),
            selected_negative_anchor_index=torch.tensor([[0], [1]]),
            margin=0.8,
            valid_anchor_count=2,
            skipped_anchor_count=0,
        )

    def test_roc_uses_candidate_pool_and_loss_uses_selected_topk(self) -> None:
        easy_topk = self._metrics(selected_distance=0.3)
        hard_topk = self._metrics(selected_distance=0.1)

        self.assertEqual(
            easy_topk["fpr_at_tpr95"],
            hard_topk["fpr_at_tpr95"],
        )
        self.assertEqual(easy_topk["neg_mean"], hard_topk["neg_mean"])
        self.assertLess(easy_topk["loss"], hard_topk["loss"])
        self.assertEqual(
            set(easy_topk),
            {
                "active_triplet_ratio",
                "eer",
                "fpr_at_tpr95",
                "loss",
                "neg_mean",
                "neg_p01",
                "neg_p05",
                "pos_mean",
                "pos_p95",
                "pos_p99",
                "roc_auc",
                "skipped_anchor_count",
                "tpr_at_fpr_1e_4",
                "valid_anchor_count",
                "valid_anchor_ratio",
            },
        )


if __name__ == "__main__":
    unittest.main()