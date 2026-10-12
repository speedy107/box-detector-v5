# V5 blue and red block detector

Based on [the original repository](https://github.com/speedy107/First-respository/tree/3874a735c3bb4539b0f22294c79b1c6091fd51bb).

YOLO11n for `blue_box` (0) and `red_box` (1), fine-tuned from the original V4 model using the 10× hybrid dataset. This is the fifth project version, not the YOLOv5 architecture.

## Run the model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/predict.py photo.jpg
# USB webcam:
python scripts/predict.py 0
```

For a CPU environment, install the matching CPU builds of torch and torchvision from the PyTorch CPU wheel index before installing requirements. For Colab with a GPU, retain its matching preinstalled torch/torchvision pair.

The fixed input is 320×320 RGB, letterboxed with 114 padding and divided by 255. Default confidence is 0.35; class-aware NMS IoU is 0.45. The ONNX output is `[1,6,2100]`, with four box coordinates and two class scores. See `models/model-spec.json`.

## Camera webpage

Serve this repository with `python -m http.server 8000` and open localhost in a desktop browser. For phone camera access, deploy it over HTTPS using GitHub Pages: Settings → Pages → main → root. Open the published page directly in Chrome and allow camera permission. Model inference uses the supplied ONNX Runtime Web WASM files; selected photos and video frames remain on the device. Actual Android performance has not been measured.

## Reproduce training

Download the separately supplied `Blocks_Dataset_10x.zip`; it is excluded from git to avoid duplicating the 125 MiB dataset. Its SHA256 is recorded in `training/dataset-source.json`.

```bash
python scripts/prepare_dataset.py Blocks_Dataset_10x.zip
python scripts/train_v5.py --dataset datasets/blocks_dataset_10x
# Colab GPU alternative: add --device 0
python scripts/evaluate_models.py --dataset datasets/blocks_dataset_10x \
  --v4 datasets/blocks_dataset_10x/weights/boxes-yolo11n-v4.pt \
  --v5 models/boxes-yolo11n-v5.pt --output evaluation/full-comparison.json
python scripts/export_onnx.py --weights models/boxes-yolo11n-v5.pt \
  --dataset datasets/blocks_dataset_10x --output evaluation/onnx-validation.json
npm install
npm run test:wasm -- datasets/blocks_dataset_10x
```

Training uses source-balanced sampling: 1,390 unique training images and 2,502 sample positions per epoch, including the 139 original images repeated nine times. Validation remains 23 original images containing only eight blocks. The added training images combine conventional augmentation, real block cutouts on eight GenAI backgrounds, and negative scenes. Generated backgrounds and labelled cutouts are in `training-assets/`; dataset generation scripts are supplied for inspection.

The trainer uses a 60-epoch learning-rate schedule and stops after nine complete epochs by default. It selects checkpoints using the 60 known cases and a 240-image sample from expanded training, then validation mAP. These regression cases inform checkpoint selection and are not an independent test set.

The public source at that pinned commit also contains `yolo11-boxes-dataset.zip` and `trained-box-model.zip`. To rebuild rather than use the supplied 10× archive, download those original archives, extract the V4 `.pt` checkpoint, then run:

```bash
python scripts/rebuild_dataset.py yolo11-boxes-dataset.zip \
  --v4-weights path/to/boxes-yolo11n.pt --output datasets/rebuilt
python scripts/train_v5.py --dataset datasets/rebuilt
```

This reuses the committed generated backgrounds and real cutouts. Pillow and NumPy versions can affect exact regenerated image bytes; the supplied archive and its checksum are the authoritative dataset used for this run.

## Evaluation and limitations

Nine epochs completed; **epoch 4** was retained because it preserves all known cases and performs best on the sampled expanded-training regressions.

| Check | V4 | V5 |
|---|---:|---:|
| Validation mAP50–95 | 86.46% | 89.98% |
| Validation mAP50 | 99.50% | 99.50% |
| Full-dataset image checks at confidence 0.35 | 1,104 / 1,413 | 1,407 / 1,413 |
| Full-dataset false positives | 177 | 6 |
| Full-dataset false negatives | 410 | 2 |
| Known real regression cases | 60 / 60 | 60 / 60 |

V5 also passes all 60 known cases in ONNX Runtime Web 1.20.1 WASM using the repository's exact decoder. This is an engine test in Node, not a complete browser UI or Android device test. The exported model passes ONNX checking and numerical comparison on all 23 validation images. All 417 floating-point weight tensors changed from V4.

The six remaining failed images are generated overlap/stack scenes, including one blue background distractor. Review `evaluation/remaining-errors.json` and `evaluation/remaining-errors.jpg`. At 0.35 confidence, all 501 background-only images and all 23 real validation images pass. Detection matching requires the correct class and IoU ≥0.5; unmatched or mislocalized predictions count toward false positives and false negatives.

Final measured results are in `evaluation/comparison-summary.json` and `training/training-summary.json`. The full 1,413-image check includes training images: it measures fitting and known regressions, not real-world generalization. Validation is small and draws from two original held-out recordings. Collect new recordings, different rooms and camera distances before drawing broad conclusions about false-positive rates or Robocon reliability.

## Licensing

The model and training use Ultralytics YOLO under its AGPL-3.0/commercial licensing terms; see `LICENSE-YOLO.txt`. ONNX Runtime uses its vendored MIT license. Dataset and cutout source-image rights are not independently established; this repository does not grant new rights to those source images.
