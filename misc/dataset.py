import glob
import os
from typing import Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ExpertDataset(Dataset):
    """
    Each directory in the root directory contains one image(.jpg/.png) and
    one caption.txt file. The returned item contains the following keys:
    - "path": The path to the image file, relative to the root directory.
    - "prompt": The text prompt from the caption.txt file.
    - "image": The loaded image (if `load_image` is True).

    Args:
        root: The root directory of the dataset.
        load_image: Whether to load the image. If False, only the prompt will
            be loaded.
    """
    def __init__(
        self,
        root: str,
        load_image: bool = True,
    ):
        self.root = root
        self.load_image = load_image

        self.paths = []
        self.paths.extend(glob.glob(os.path.join(root, "**/caption.txt")))
        self.paths = sorted(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = dict()
        caption_path = self.paths[idx]

        # Load the prompt
        with open(caption_path) as f:
            prompt = f.read()
        item["prompt"] = prompt

        # Load the image if `load_image` is enabled
        if self.load_image:
            image_path = glob.glob(os.path.join(os.path.dirname(caption_path), "*.png"))
            if len(image_path) == 0:
                raise FileNotFoundError(f"No .png image found in {os.path.dirname(caption_path)}")
            else:
                image_path = image_path[0]
            image = Image.open(image_path)
            # For Aestheticv2, some images are corrupted. We will replace them
            # with a blank image to avoid exceptions.
            if image.size[0] < 10 or image.size[1] < 10:
                image = Image.new("RGB", (512, 512))
            item["image"] = image
            item["path"] = os.path.relpath(image_path, self.root)

        return item


class TrainingDataset(ExpertDataset):
    """
    Dataset for training. It applies the following transformations to the
    image:
    1. Resize to the given resolution (if specified).
    2. Random horizontal flip (if specified).
    3. Random crop or center crop to make the image square (if specified).
    4. Normalize the image to [-1, 1].

    The returned item contains the following keys:
    - "path": The path to the image file, relative to the root directory.
    - "prompt": The text prompt from the caption.txt file.
    - "image": The transformed image.
    - "add_time_ids": A tensor of shape (6,) containing the original image
        height, original image width, y1, x1, crop height, crop width.

    Args:
        root: The root directory of the dataset.
        resolution: The resolution of the image to be cropped.
        random_flip: Whether to apply random horizontal flip.
        random_crop: Whether to apply random resized crop. If False, the center
            crop will be applied to make the image square
    """
    def __init__(
        self,
        root: str,
        resolution: int,
        random_flip: bool = False,
        random_crop: bool = False,
    ):
        super().__init__(root)
        self.resolution = resolution
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.transform = v2.Compose([
            v2.RGB(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(resolution),
            v2.RandomHorizontalFlip() if random_flip else v2.Lambda(lambda x: x),
            v2.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = item['image']
        original_h = image.height
        original_w = image.width
        image = self.transform(image)
        _, resized_h, resized_w = image.shape

        # Apply random crop or center crop.
        if self.random_crop is False:
            y1 = max(0, int(round((resized_h - self.resolution) / 2.0)))
            x1 = max(0, int(round((resized_w - self.resolution) / 2.0)))
            h, w = self.resolution, self.resolution
        elif self.random_crop is True:
            y1, x1, h, w = v2.RandomCrop.get_params(image, output_size=(self.resolution, self.resolution))
        else:
            y1, x1, h, w = 0, 0, resized_h, resized_w
        item['image'] = v2.functional.crop(image, y1, x1, h, w)

        # For SDXL, we need to provide the original image size and the
        # coordinates of the crop.
        item["add_time_ids"] = torch.tensor([original_h, original_w, y1, x1, self.resolution, self.resolution])

        return item


class ScoreDataset(ExpertDataset):
    """
    Dataset for scoring. It applies the given transformation to the image. The
    returned item contains the following keys:
    - "path": The path to the image file, relative to the root directory.
    - "prompt": The text prompt from the caption.txt file.
    - "image": The transformed image.
    """
    def __init__(
        self,
        root: str,
        transform: Callable[[Image.Image], tv_tensors.Image],
    ):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.transform is not None:
            item['image'] = self.transform(item['image'])
        assert item['image'].shape[0] == 3, f"Image {os.path.join(self.root, item['path'])} is not RGB."

        return item


class PromptDataset(ExpertDataset):
    """
    Dataset for loading only prompts. The returned item contains the following
    keys:
    - "path": The path to the image file, relative to the root directory.
    - "prompt": The text prompt from the caption.txt file.
    """
    def __init__(
        self,
        root: str,
    ):
        super().__init__(root, load_image=False)
