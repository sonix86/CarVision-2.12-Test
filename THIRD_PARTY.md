# Libraries and public photo examples

Owner: София Рахматулаева. OpenAI Codex assisted with implementation, design, tests, documentation and packaging. No student-collected dataset or measured real-world accuracy is claimed.

## Libraries

- Python 3.12: PSF license, https://docs.python.org/3/license.html
- PyTorch 2.6.0 CPU: BSD-style, https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE
- torchvision 0.21.0: BSD-3-Clause, https://github.com/pytorch/vision/blob/v0.21.0/LICENSE
- NumPy/SciPy: BSD-3-Clause; Matplotlib: PSF-based; Pillow: HPND; ReportLab: BSD; PyInstaller: GPL with bootloader exception.
- Installed license notices accompany the portable distribution in `licenses/`.

## No pretrained model weights in release 1.1

The source uses torchvision network architectures, not pretrained parameters. Both SSDLite320 MobileNetV3 Large and the MobileNetV3 Small dual-head classifier start from random initialization. Constructors explicitly use `weights=None`, `weights_backbone=None` as applicable. COCO and ImageNet weights from the initial prototype are not included or loaded. The team's resulting checkpoints record `origin=team-trained-from-scratch`.

## Public photographs

Six original public images in `samples/images` are from Wikimedia Commons. `samples/sources.json` records exact source page URLs, authors, original descriptions/dates, license names and license URLs. The separately authored photo booklet reproduces proportionally resized images and identifies their individual licenses on each page. The collection does not claim ownership of photographs. CC BY / CC BY-SA terms continue to apply to each photo; retain source and license credits on redistribution. Photographs are not automatically used as training data or prewritten inference answers.

Network architecture documentation: https://docs.pytorch.org/vision/0.21/models.html . Own training photographs, badge labels and final report analysis remain the responsibility of the actual contributors.
