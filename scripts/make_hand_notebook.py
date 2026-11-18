"""Generates notebooks/01_hand_landmarks_colab.ipynb (Phase 1: MediaPipe hand + fingertip detection).

Usage: python scripts/make_hand_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# Phase 1: Hand and fingertip detection (MediaPipe)
Uses Google's **pretrained** MediaPipe Hand Landmarker (21 landmarks per hand). No training needed.

We highlight the 5 fingertips we care about:

| ID | Landmark |
|----|----------|
| 4  | Thumb tip |
| 8  | Index fingertip |
| 12 | Middle fingertip |
| 16 | Ring fingertip |
| 20 | Pinky fingertip |

**Run order:** cell 1 (install) -> cell 2 (setup) -> cell 3 (live webcam) and/or cell 4 (image upload).
Cells 3 and 4 are independent, so you can run either one after cell 2.
""")

md("## 1. Install and download the model")
code(r"""
!pip -q install mediapipe
!wget -q -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
import mediapipe as mp
print("mediapipe", mp.__version__)
""")

md("## 2. Setup: detector and drawing helpers")
code(r"""
import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# The landmarks we care about (MediaPipe hand landmark index -> name)
FINGERTIPS = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
# BGR colors, one per fingertip
COLORS = {4: (0, 165, 255), 8: (0, 255, 0), 12: (255, 128, 0), 16: (255, 0, 255), 20: (0, 255, 255)}
# Hand skeleton: pairs of landmark indices to connect with lines
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def make_landmarker(mode, num_hands=2, min_conf=0.5):
    # mode: vision.RunningMode.IMAGE (single photos) or VIDEO (frames in sequence, with tracking)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def put_label(img, text, org, scale, color):
    # white outline + colored text so it is readable on any background
    th = max(1, int(2 * scale))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, th, cv2.LINE_AA)


def detect(landmarker, bgr, timestamp_ms=None):
    # Returns (annotated BGR image, list of fingertip dicts). Pass timestamp_ms only for VIDEO mode.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img) if timestamp_ms is None else landmarker.detect_for_video(mp_img, timestamp_ms)

    out = bgr.copy()
    h, w = out.shape[:2]
    s = max(h, w) / 640.0  # scale drawing to image size
    tips = []
    for i, lms in enumerate(res.hand_landmarks):
        hand = res.handedness[i][0].category_name  # "Left" / "Right" (as seen by the model)
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (230, 230, 230), max(1, int(2 * s)), cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, max(2, int(3 * s)), (90, 90, 90), -1, cv2.LINE_AA)
        for idx, name in FINGERTIPS.items():
            x, y = pts[idx]
            cv2.circle(out, (x, y), int(9 * s), COLORS[idx], -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), int(9 * s), (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)
        # Labels go in a column beside the hand, spaced apart, with a line back to each tip,
        # so they stay readable even when fingertips are close together (as when fretting).
        gap = int(20 * s)
        label_x = min(max(p[0] for p in pts) + int(40 * s), w - int(95 * s))
        order = sorted(FINGERTIPS, key=lambda i: pts[i][1])
        prev_y = -10**9
        for idx in order:
            x, y = pts[idx]
            ly = max(y, prev_y + gap)
            prev_y = ly
            cv2.line(out, (x, y), (label_x - 4, ly - int(5 * s)), COLORS[idx], max(1, int(1.5 * s)), cv2.LINE_AA)
            put_label(out, f"{FINGERTIPS[idx]} {idx}", (label_x, ly), s, COLORS[idx])
        for idx, name in FINGERTIPS.items():
            x, y = pts[idx]
            tips.append({"hand": hand, "finger": name, "id": idx, "x_px": x, "y_px": y,
                         "x": round(lms[idx].x, 3), "y": round(lms[idx].y, 3), "z": round(lms[idx].z, 3)})
    return out, tips
print("Setup done.")
""")

md(r"""
## 3. Live webcam
Click **Allow** when the browser asks for camera access. Show your hand (fretting-hand fingertips toward the camera) and the labeled fingertips appear on the video.
Click **Stop camera** when finished. Re-run this cell to start again.

Note: the video goes browser -> Colab -> browser each frame, so expect roughly 5-15 FPS. That is a Colab limit; a local script will be smoother later.
""")
code(r"""
import base64, time
from google.colab import output
from IPython.display import display, Javascript, JSON

live_lm = make_landmarker(vision.RunningMode.VIDEO)
t0, last_ts = time.time(), -1

def process_frame(data_url):
    global last_ts
    jpg = base64.b64decode(data_url.split(",")[1])
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    ts = max(int((time.time() - t0) * 1000), last_ts + 1)  # timestamps must strictly increase
    last_ts = ts
    out, tips = detect(live_lm, bgr, ts)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return JSON({"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                 "tips": [f"{t['hand']} {t['finger']}" for t in tips]})

output.register_callback("notebook.process_frame", process_frame)

JS = '''
async function run() {
  const box = document.createElement("div");
  const btn = document.createElement("button");
  btn.textContent = "Stop camera"; btn.style.cssText = "font-size:16px;padding:6px 14px";
  const status = document.createElement("div");
  status.style.cssText = "font:16px sans-serif;margin:8px 0";
  const img = document.createElement("img"); img.style.maxWidth = "100%";
  box.append(btn, status, img);
  document.querySelector("#output-area").appendChild(box);
  let stop = false; btn.onclick = () => { stop = true; };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
    const video = document.createElement("video");
    video.srcObject = stream; video.muted = true; await video.play();
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    while (!stop) {
      ctx.drawImage(video, 0, 0);
      const res = await google.colab.kernel.invokeFunction(
        "notebook.process_frame", [canvas.toDataURL("image/jpeg", 0.7)], {});
      const d = res.data["application/json"];
      img.src = d.img;
      status.textContent = d.tips.length ? "Detected: " + d.tips.join(", ") : "No hand detected";
    }
    stream.getTracks().forEach(t => t.stop());
    status.textContent = "Camera stopped.";
  } catch (e) {
    status.textContent = "Camera error: " + e;
  }
}
run();
'''
display(Javascript(JS))
""")

md(r"""
## 4. Test on an image from your dataset
Run the cell, click **Choose files**, and pick one or more images (for example from `data/images/train/Am/` on your PC).
Each image is shown with the detected fingertips, followed by their pixel coordinates.
Lower confidence (0.3) is used here because the dataset has heavily augmented photos.
""")
code(r"""
import matplotlib.pyplot as plt
from google.colab import files

img_lm = make_landmarker(vision.RunningMode.IMAGE, min_conf=0.3)
uploaded = files.upload()

for name, data in uploaded.items():
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"{name}: not a readable image"); continue
    out, tips = detect(img_lm, bgr)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title(name); plt.show()
    if not tips:
        print(f"{name}: no hand detected (try a clearer, closer photo)\n"); continue
    print(f"{name}: {len(tips) // 5} hand(s) detected")
    for t in tips:
        print(f"  {t['hand']:5s} {t['finger']:6s} (id {t['id']:2d})  pixel=({t['x_px']}, {t['y_px']})  "
              f"normalized=({t['x']}, {t['y']})  depth z={t['z']}")
    print()
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_hand_landmarks_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
