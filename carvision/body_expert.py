"""Locally trained body-detail expert; ImageNet initialization only.

Padding preserves body proportions. The training and runtime transformations
are deliberately shared; no image filenames or scene location enter the model.
"""
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms


def body_canvas(image, size=224):
    image = image.convert('RGB').copy()
    scale = size/max(image.size)
    image = image.resize((max(1, round(image.width*scale)),
                          max(1, round(image.height*scale))), Image.Resampling.BICUBIC)
    canvas = Image.new('RGB', (size, size), (124, 116, 104))
    canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
    return canvas


def body_tensor(image):
    return transforms.functional.normalize(
        transforms.functional.to_tensor(body_canvas(image)),
        [.485, .456, .406], [.229, .224, .225])


class BodyExpert:
    def __init__(self, checkpoint, pairs):
        if checkpoint.get('origin') != 'project-trained-body-expert-v1':
            raise ValueError('Expected independently trained body expert')
        if checkpoint['pairs'] != [list(p) for p in pairs]:
            raise ValueError('Body expert taxonomy mismatch')
        self.weight = float(checkpoint['weight'])
        self.temperature = float(checkpoint['temperature'])
        self.make_threshold = float(checkpoint.get('make_threshold', .9))
        if not 0 <= self.weight <= 1 or self.temperature <= 0 or not 0 <= self.make_threshold <= 1:
            raise ValueError('Invalid body expert calibration')
        self.network = models.resnet18(weights=None)
        self.network.fc = torch.nn.Linear(512, len(pairs))
        self.network.load_state_dict(checkpoint['state'])
        self.network.eval()

    @torch.inference_mode()
    def probabilities(self, image):
        views = torch.stack([body_tensor(image), body_tensor(ImageOps.mirror(image))])
        return (self.network(views)/self.temperature).softmax(1).mean(0)
