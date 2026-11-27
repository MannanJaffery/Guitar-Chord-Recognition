"""Unit tests for the 5-stage Guitar Chord Recognition pipeline."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from geometry_features import (
    FRET_G1,
    FRET_NUT_S,
    feature_vector,
    fret_estimate,
    fret_slot,
    neck_geometry,
    string_coord,
    tab_features,
)
from live_pipeline import LandmarkTracker, knn_distance


def test_fret_and_string_math():
    assert abs(fret_estimate(FRET_NUT_S)) < 1e-5
    assert fret_slot(FRET_NUT_S + 0.1) == 1
    assert abs(string_coord(0.085) - 6.0) < 1e-5
    assert abs(string_coord(0.085 + 5 * 0.166) - 1.0) < 1e-5


def test_feature_vector_shapes():
    s = np.linspace(0.0, 2.0, 21)
    v = np.linspace(0.1, 0.9, 21)
    tab = tab_features(s, v)
    assert tab.shape == (17,)
    feat, names = feature_vector(s, v)
    assert feat.shape == (64,)
    assert len(names) == 64
    assert "str0" in names and "fr0" in names and "span_fr" in names


def test_neck_geometry_validation():
    valid_quad = np.array([[100, 100], [100, 200], [700, 200], [700, 100]], dtype=np.float32)
    g = neck_geometry(valid_quad, 1280, 720)
    assert g["valid"] is True
    assert g["aspect"] == pytest.approx(6.0, rel=1e-2)

    sliver = np.array([[100, 100], [100, 102], [110, 102], [110, 100]], dtype=np.float32)
    g_bad = neck_geometry(sliver, 1280, 720)
    assert g_bad["valid"] is False


def test_landmark_tracker_persistence():
    tracker = LandmarkTracker(max_drop=3)
    lm1 = np.ones((21, 2), dtype=np.float32) * 100.0
    lm2 = np.ones((21, 2), dtype=np.float32) * 104.0

    out1, ext1 = tracker.update(lm1)
    assert ext1 is False and np.allclose(out1, lm1)

    out2, ext2 = tracker.update(lm2)
    assert ext2 is False and np.allclose(out2, lm2)

    # Dropped frame 1 -> extrapolated
    out3, ext3 = tracker.update(None)
    assert ext3 is True and out3 is not None

    # Dropped frame 4 -> exceeds max_drop=3 -> reset to None
    tracker.update(None)
    tracker.update(None)
    out6, ext6 = tracker.update(None)
    assert out6 is None and ext6 is False


def test_trained_models_loadable():
    models_dir = ROOT / "artifacts" / "models_local"
    assert (models_dir / "neck_v2.pt").exists()
    assert (models_dir / "string_net.pt").exists()
    assert (models_dir / "fret_net.pt").exists()
    assert (models_dir / "chord_clf.joblib").exists()
    assert (models_dir / "chord_meta.json").exists()

    meta = json.loads((models_dir / "chord_meta.json").read_text())
    assert len(meta["classes"]) == 9
    assert len(meta["columns"]) == 17

    bundle = joblib.load(models_dir / "chord_clf.joblib")
    clf = bundle["model"]
    x_dummy = np.zeros((1, 17), dtype=np.float32)
    probs = clf.predict_proba(x_dummy)
    assert probs.shape == (1, 9)

    from train_string_detector import StringNet
    snet = StringNet()
    snet.load_state_dict(torch.load(models_dir / "string_net.pt", map_location="cpu", weights_only=False)["model"])
    snet.eval()
    assert snet(torch.zeros((1, 1, 160, 640))).shape == (1, 6)

    from train_fret_detector import FretNet
    fnet = FretNet()
    fnet.load_state_dict(torch.load(models_dir / "fret_net.pt", map_location="cpu", weights_only=False)["model"])
    fnet.eval()
    assert fnet(torch.zeros((1, 1, 160, 640))).shape == (1, 2)


def test_pipeline_invalid_ema_no_crash():
    """Regression test: Pipeline.step() must not crash with perspectiveTransform when ema becomes invalid."""
    from live_pipeline import Pipeline

    class DummyClf:
        classes_ = np.array(["C", "G"])
        def predict_proba(self, X):
            return np.array([[0.8, 0.2]])

    valid_quad = np.array([[100, 100], [100, 200], [700, 200], [700, 100]], dtype=np.float32)
    invalid_quad = np.array([[100, 100], [100, 101], [105, 101], [105, 100]], dtype=np.float32)
    hand_pts = np.tile([200.0, 150.0], (21, 1)).astype(np.float32)

    state = {"step": 0}
    def fake_neck(bgr):
        return valid_quad if state["step"] == 0 else invalid_quad
    def fake_hand(bgr):
        return [hand_pts] if state["step"] == 0 else []

    pipe = Pipeline(fake_neck, fake_hand, DummyClf(), [f"str{i}" for i in range(4)] + [f"fr{i}" for i in range(4)])
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out0 = pipe.step(dummy_img)
    assert out0["best"] == "C"

    # Force ema invalid while tracker has cached hand
    pipe.ema = invalid_quad
    state["step"] = 1
    out1 = pipe.step(dummy_img)
    assert out1["lm"] is None

