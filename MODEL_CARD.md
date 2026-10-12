# V5 model card

Purpose: detect separate blue and red ground blocks in camera images. The architecture is YOLO11n, with a fixed 320×320 input, two classes and an FP32 ONNX export.

Data: 139 original training photos expanded to 1,390 via conventional augmentation and real block cutouts on eight generated backgrounds. 23 original validation images remain unchanged. The expanded examples reuse eight cutouts and eight backgrounds, so 10× more samples does not mean 10× more independent visual information.

Selection: fine-tune from V4 with AdamW, low learning rate and 50% original-source sampling. Select checkpoints that retain the 60 known cases and perform best on a 240-image sample from expanded training, with a validation mAP floor. The regression set informs selection; it is not an independent benchmark.

Evaluation: validation mAP50–95 is 89.98% versus V4’s 86.46%. V5 passes 1,407/1,413 full-dataset image checks at 0.35 confidence, with six false positives and two false negatives under class/IoU matching. All 60 known real cases pass in PyTorch and WASM. See the JSON files under evaluation for per-image results. Matching requires the correct class and IoU ≥0.5; any unmatched prediction above 0.35 confidence makes the image fail. Validation has only eight labelled blocks from two recordings. Full-dataset regression results largely measure training fit. Do not treat either as a reliable estimate of performance in new arenas.

Limits: unusual block viewpoints, tiny distant targets, motion blur, bright or dark scenes, partial occlusion and same-colour touching stacks need new real footage. The model provides bounding boxes and colours; it does not provide depth, physical dimensions or recognition of other object classes. Duplicate or fragment boxes can still occur on stacked blocks. Hardware speed and full Android UI behaviour have not been validated for this version.

Licensing: Ultralytics AGPL-3.0/commercial terms apply; ONNX Runtime assets use MIT. Source-image rights were not independently verified.
