# Libraries, general weights and public photographs

Owner: София Рахманкулова. OpenAI Codex assisted with code, training, data preparation, visual colour review, tests, documentation and packaging. No student-collected data is claimed.

## Libraries

Python 3.12 (PSF), PyTorch 2.6.0 CPU (BSD-style), torchvision 0.21.0 (BSD-3-Clause), NumPy/SciPy (BSD), Pillow (HPND), Matplotlib (PSF-based), ReportLab (BSD), OpenCV (Apache-2.0), PyInstaller (GPL with bootloader exception). Installed license notices are included in licenses/. PyArrow is used only for optional public-data preparation, not required by inference.

## General pretrained parameters

The owner explicitly authorized general ImageNet/COCO weights on 2026-10-20. The assignment allows them. Official torchvision Faster R-CNN ResNet50 FPN v2 COCO_V1 supplies generic car/truck boxes. A separate SSDLite general backbone is included for optional local training on personal annotations. Official MobileNetV3 Small IMAGENET1K_V1 supplies the feature initialization. Its original ImageNet classification head is not used. The 18-brand decision head is trained by this project; new regional class rows were randomly initialized. The colour algorithm is original code; no ready-made brand or colour model is used.

- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
- https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html

## Public training data

Stanford Cars, Krause et al. 2013, used as a public research dataset. Image mirror: https://huggingface.co/datasets/tanganke/stanford_cars . Original-image and annotation mirror: https://github.com/jhpohovey/StanfordCars-Dataset . Exact file hashes and original human bounding boxes were joined; split/brand conflicts excluded; original SHA256 duplicates removed. data/public_prototype/provenance.json records counts and mapping. This project does not own those photographs; raw public training images are excluded from the portable distribution. Research use does not establish unrestricted commercial image rights. Brand labels are original dataset labels; colour pseudo-labels are explicitly not human truth.

## City examples

Six photographs in samples/images come from Wikimedia Commons; exact authors, source pages, dates, license names and license URLs are in samples/sources.json. Preserve those credits and the applicable CC BY / CC BY-SA terms when redistributing photographs. They were not used to train or choose the brand model, and are not student-collected photos. The supplementary PDF contains resized reproductions with credits, not a final academic report.

## Independently trained colour decision

VCoR colour photographs and provided labels were obtained from https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset . The authors permit research and education use and request citation: Panetta, K., Kezebou, L., Oludare, V., Intriligator, J., Agaian, S. (2021), Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos, AI 2(4), 684–704, https://doi.org/10.3390/ai2040041 . The dataset authors describe images collected from Google and subsequently cleaned/annotated. This project does not own or redistribute those raw training images in the portable package. Source metadata, archive SHA256, retained image hashes and exact colour mappings are preserved in data/public_prototype and source/data/public_prototype.

Собственные обученные цветовые головы: полная вырезка и участки кузова; веса видов выбраны по development. Готовые vehicle-colour classifiers не используются.. Colour decisions are independently trained in this project. No ready-made vehicle-colour weights were obtained. Original train/validation/test folders are retained after removing exact SHA256 or identical dHash duplicates; vehicle identities are not available, so identity-separated generalization is not certified. Beige/gold/tan map to brown, silver to grey, orange to yellow; pink/purple are excluded. All eight required colours are represented. Reported VCoR test accuracy is a dataset benchmark, not a guarantee for Tashkent streets.


## Additional regional photographs / v1.3

Supplementary Commons category-labelled vehicle photographs, locally resized and automatically cropped. Authors, source URLs, licence names/URLs, changes and hashes are preserved in data/regional/source_manifest_snapshot.json and labels.jsonl. They are public photographs, never attributed to the student; raw training photographs are excluded from the portable package. Author/date session split, near-dHash deduplication and development-demo exclusion are documented in provenance.json.

## Instance contours and visible-paint training / v1.5

Generic Mask R-CNN ResNet50 FPN v2 COCO weights are from the official torchvision distribution. Exact URL and metadata: models/segmenter.json. These weights locate visible objects; vehicle brand/model/colour heads are independently trained in this project. Preserve torchvision and COCO attribution bundled with the distribution.

The visible-paint ExtraTrees head is trained locally on source-labelled VCoR samples. scikit-learn 1.6.1 is used only during training; the portable runtime evaluates exported arrays and excludes sklearn. Training recipes, experiments, splits and calibration results are included in source/tools and models/*.json.

## Visible-part references / v1.7

The 3,740 descriptors come exclusively from the existing 374 regional training photographs and their whole/upper/left/right views. The descriptor network is our own previously trained model-name network; no ready-made vehicle-model or brand classifier was obtained. Public source categories, authors, image changes, licences and group splits remain in data/regional metadata. Validation and test images, city examples and filenames are not references. The portable package distributes learned descriptors, not the raw regional training images. Relative scores only compare the sixteen learned classes; they are not an open-world probability of correctness.
