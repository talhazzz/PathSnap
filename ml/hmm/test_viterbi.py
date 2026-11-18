"""Focused smoke tests for connected and disconnected Viterbi transitions."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ml.hmm.viterbi import decode_viterbi


def candidate(point_idx: int, arc_id: int, probability: float, point_step: float) -> dict:
    return {
        "trip_id": "synthetic",
        "point_idx": point_idx,
        "candidate_arc_id": arc_id,
        "projection_fraction": 0.5,
        "arc_length_m": 10.0,
        "distance_to_arc_start_m": 5.0,
        "distance_to_arc_end_m": 5.0,
        "gps_step_distance_m": point_step,
        "candidate_probability": probability,
    }


class ViterbiDecoderTests(unittest.TestCase):
    def test_prefers_connected_road_and_resets_disconnected_path(self) -> None:
        candidates = pd.DataFrame(
            [
                candidate(0, 0, 0.55, 0.0),
                candidate(0, 2, 0.45, 0.0),
                candidate(1, 1, 0.75, 10.0),
                candidate(1, 2, 0.25, 10.0),
                candidate(2, 3, 0.99, 10.0),
            ]
        )
        # 0 -> 1 is a directed connection; arc 2 and arc 3 are separate roads.
        arcs = np.asarray([[0, 1], [1, 2], [8, 9], [10, 11]])
        result = decode_viterbi(candidates, arcs)
        self.assertEqual(result["candidate_arc_id"].iloc[:2].tolist(), [0, 1])
        self.assertFalse(bool(result["reset_before_point"].iloc[1]))
        self.assertTrue(bool(result["reset_before_point"].iloc[2]))
        self.assertEqual(int(result["segment_id"].nunique()), 2)


if __name__ == "__main__":
    unittest.main()