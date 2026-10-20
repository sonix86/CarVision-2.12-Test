# Dependencies and permitted generic pretrained weights

- Python 3.12: PSF license, https://docs.python.org/3/license.html
- PyTorch 2.6.0 CPU: BSD-style, https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE
- torchvision 0.21.0: BSD-3-Clause, https://github.com/pytorch/vision/blob/v0.21.0/LICENSE
- NumPy: BSD-3-Clause; SciPy: BSD-3-Clause; Matplotlib: PSF-based; Pillow: HPND; ReportLab: BSD; PyInstaller: GPL with bootloader exception allowing bundled applications. Installed license files are included in `licenses/` in the portable package.
- COCO generic detector: torchvision `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html . Download https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
- General ImageNet initialization: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html . Download https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

These are generic object-detection / ImageNet weights, not ready-made car-brand or car-colour classification weights. Academic use is the intended context. Check original model/dataset terms before broader redistribution or commercial deployment; dataset terms are separate from library licenses. Own photographs remain the responsibility of their authors. No public car dataset has been included in the training set.

AI disclosure: OpenAI Codex assisted with requirements analysis, implementation, packaging, tests and documentation. No student data collection, student authorship, model accuracy or student contribution history is asserted on that basis.
