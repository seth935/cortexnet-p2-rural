"""CortexNet-P2-Rural (v3.0.0) — Local Pre-Submission & Empty-Canary Self-Audit."""

import json
import os
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort


def verify_all(repo_dir: str | Path = "."):
    root = Path(repo_dir).resolve()
    manifest = json.loads((root / "model.json").read_text(encoding="utf-8"))

    assert manifest["source_type"] == "registry_fork", "Expected source_type='registry_fork'"
    assert "sub02cortexnetobb1w0000000000002" in manifest["parent_lineage"]
    threads = int(manifest.get("threads", 2))

    model_path = root / manifest["detector"]["model_path"]
    assert model_path.exists(), f"Missing {model_path}"
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert 0.06 <= size_mb <= 5.0, f"Artifact size {size_mb:.3f} MB out of range"

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    h, w = manifest["detector"]["input_size"]
    conf_thr = float(manifest["detector"]["conf_threshold"])

    for label, val in [("all-black", 0.0), ("all-white", 1.0), ("flat-gray", 0.5)]:
        canary = np.full((1, 3, h, w), val, dtype=np.float32)
        out = sess.run(None, {in_name: canary})[0]
        assert out.shape == (1, 100, 8), f"Unexpected shape: {out.shape}"
        active = int(np.sum(out[0, :, 5] >= conf_thr))
        assert active == 0, f"Failed empty-frame control on {label}: {active} false alarms"

    # Low-light + small-horizon active scene test
    scene = np.full((1, 3, h, w), 0.08, dtype=np.float32)
    scene[:, :, 110:170, 130:200] = 0.24  # Low-contrast dark target (LLR simulation)
    scene[:, :, 80:100, 250:268] = 0.28   # Sub-32px rural horizon target
    t0 = time.perf_counter()
    for _ in range(30):
        out = sess.run(None, {in_name: scene})[0]
    avg_ms = ((time.perf_counter() - t0) / 30.0) * 1000.0
    active_dets = int(np.sum(out[0, :, 5] >= conf_thr))
    assert active_dets >= 2, f"Expected >=2 active detections on low-light/horizon scene, got {active_dets}"

    print(f"[PASS] CortexNet-P2-Rural: {size_mb * 1024:.1f} KB ({size_mb:.3f} MB) | Shape: {out.shape}")
    print(f"[PASS] Empty-Scene False Alarms: 0.0000 FPPI | Low-Light Active Detections: {active_dets}")
    print(f"[PASS] Host Latency (2 threads @ {h}x{w}): {avg_ms:.2f} ms ({1000.0 / avg_ms:.1f} FPS)")


if __name__ == "__main__":
    verify_all(Path(__file__).resolve().parent)
