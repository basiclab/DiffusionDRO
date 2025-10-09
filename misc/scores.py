import os
import time
import warnings
from contextlib import nullcontext
from typing import Callable, Tuple

import huggingface_hub
import ImageReward as RM
import requests
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import hpsv2
    from hpsv2.utils import hps_version_map

    try:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    except FileNotFoundError as e:
        # Patch the missing vocab file
        VOCAB_URL = "https://github.com/tgxs002/HPSv2/raw/f924ea23004d3870d0fa89532ac67e6fcfd82fcc/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        VOCAB_RELATIVE_PATH = "src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        VOCAB_PATH = os.path.join(os.path.dirname(hpsv2.__file__), VOCAB_RELATIVE_PATH)
        if not os.path.exists(VOCAB_PATH):
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                tmp_path = VOCAB_PATH + ".tmp"
                # issue 30: https://github.com/tgxs002/HPSv2/issues/30#issuecomment-2037345836
                response = requests.get(VOCAB_URL)
                response.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    f.write(response.content)
                os.replace(tmp_path, VOCAB_PATH)
            else:
                while not os.path.exists(VOCAB_PATH):
                    time.sleep(1)
        else:
            raise e

        # Import open_clip module after patching the vocab file
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


def load_hpsv2(hps_version: str = "v2.0"):
    model, _, preprocess = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    model.eval()

    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id="xswu/HPSv2", filename=hps_version_map[hps_version])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    tokenizer = get_tokenizer('ViT-H-14')

    return model, preprocess, tokenizer


def load_clip():
    model, _, preprocess = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    model.eval()

    tokenizer = get_tokenizer('ViT-H-14')

    return model, preprocess, tokenizer


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticV2(nn.Module):
    PREDICTOR_WEIGHTS_URL = "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth"
    PREDICTOR_WEIGHTS_SAVE_PATH = "{cache_directory}/sac+logos+ava1-l14-linearMSE.pth"

    def __init__(self, cache_directory='~/.cache/aesthetic/v2'):
        super().__init__()
        path = self.PREDICTOR_WEIGHTS_SAVE_PATH.format(
            cache_directory=os.path.expanduser(cache_directory))
        # Download the weights if they do not exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with requests.get(self.PREDICTOR_WEIGHTS_URL, stream=True) as r:
                with open(path, 'wb') as f, tqdm(
                        unit='B', unit_scale=True, unit_divisor=1024,
                        total=int(r.headers['Content-Length']),
                        desc="Downloading Aesthetic Predictor V2") as pbar:
                    for chunk in r.iter_content(chunk_size=4096):
                        f.write(chunk)
                        pbar.update(len(chunk))
        # Load the weights
        self.mlp = AestheticMLP()
        self.mlp.load_state_dict(torch.load(path, map_location='cpu'))

        # Load the CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map='cpu')

    def __call__(self, *args, **kwargs):
        embed = self.clip.get_image_features(*args, **kwargs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        scores = self.mlp(embed).squeeze(1)
        return scores


class AestheticV1(nn.Module):
    PREDICTOR_WEIGHTS_URL = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
    PREDICTOR_WEIGHTS_SAVE_PATH = "{cache_directory}/sa_0_4_vit_l_14_linear.pth"

    def __init__(self, cache_directory='~/.cache/aesthetic/v1'):
        super().__init__()
        path = self.PREDICTOR_WEIGHTS_SAVE_PATH.format(
            cache_directory=os.path.expanduser(cache_directory))
        # Download the weights if they do not exist
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with requests.get(self.PREDICTOR_WEIGHTS_URL, stream=True) as r:
                with open(path, 'wb') as f, tqdm(
                        unit='B', unit_scale=True, unit_divisor=1024,
                        total=int(r.headers['Content-Length']),
                        desc="Downloading Aesthetic Predictor V1") as pbar:
                    for chunk in r.iter_content(chunk_size=4096):
                        f.write(chunk)
                        pbar.update(len(chunk))
        # Load the weights
        self.mlp = nn.Linear(768, 1)
        self.mlp.load_state_dict(torch.load(path, map_location='cpu'))

        # Load the CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map='cpu')

    def __call__(self, *args, **kwargs):
        embed = self.clip.get_image_features(*args, **kwargs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        scores = self.mlp(embed).squeeze(1)
        return scores


def load_aestheticv2():
    model = AestheticV2()
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=True)
    model.eval()
    return model, preprocess


def load_aestheticv1():
    model = AestheticV1()
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", clean_up_tokenization_spaces=True)
    model.eval()
    return model, preprocess


def get_score(
    name: str,
    device: torch.device,
    **kwargs
) -> Tuple[Callable, Callable]:
    """Get a score function and a transformation function for the given score name.

    Args:
        name (str): The name of the score. Supported names are
            - "pickscore"
            - "hpsv2"
            - "aesthetic"
            - "clip"
            - "imagereward"
        device (torch.device): The device to run the model on.
        **kwargs: Additional keyword arguments.
            'aesthetic_version' (str): The version of `aesthetic` to use.
                Default is "v1". Supported versions are "v1" and "v2".
            'hps_version' (str): The version of `hpsv2` to use. Default is
                "v2.0". Supported versions are "v2.0" and "v2.1".
            'accelerator' (accelerate.Accelerator): The optional accelerator
                to use for avoiding the duplicate model download in the case of
                `aesthetic`.

    Returns:
        Tuple[Callable, Callable]: The score function and the transformation
            function. The score function takes two arguments: `images`
            (batch images) and `prompts` (list of str), and returns the scores.
            The transformation function takes an PIL.Image and returns the
            transformed image tensor.

    """
    if name == "pickscore":             # PickScore
        model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device)
        model = model.eval()
        processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", clean_up_tokenization_spaces=True)

        @torch.no_grad()
        def compute_score(images, prompts):
            text_inputs = processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            image_embs = model.get_image_features(images)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
            return scores

        def transform(image):
            return processor(images=image, return_tensors="pt")['pixel_values'][0]

    elif name == "hpsv2":               # HPSv2
        if 'hps_version' in kwargs:
            hps_version = kwargs.pop('hps_version')
            model, processor, tokenizer = load_hpsv2(hps_version=hps_version)
        else:
            model, processor, tokenizer = load_hpsv2()
        model = model.to(device)

        @torch.no_grad()
        def compute_score(images, prompts):
            text_inputs = tokenizer(prompts).to(device)
            outputs = model(images, text_inputs)

            scores = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
            return scores * 100

        transform = processor

    elif name == "aesthetic":           # Aestheticv2
        version = kwargs.pop('aesthetic_version', 'v1')
        accelerator = kwargs.pop('accelerator', None)
        with accelerator.main_process_first() if accelerator else nullcontext():
            if version == 'v1':
                model, processor = load_aestheticv1()
            elif version == 'v2':
                model, processor = load_aestheticv2()
            else:
                raise ValueError(f"Unknown Aesthetic version: {version}")
        model = model.to(device)

        @torch.no_grad()
        def compute_score(images, prompts):
            scores = model(images)
            return scores

        def transform(image):
            return processor(images=image, return_tensors="pt")['pixel_values'][0]

    elif name == "clip":                # CLIP Score
        model, processor, tokenizer = load_clip()
        model = model.to(device)

        @torch.no_grad()
        def compute_score(images, prompts):
            text_inputs = tokenizer(prompts).to(device)
            outputs = model(images, text_inputs)

            scores = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
            return scores

        transform = processor

    elif name == "imagereward":         # ImageReward
        model = RM.load("ImageReward-v1.0", device=device)

        @torch.no_grad()
        def compute_score(images, prompts):
            """Compute ImageReward

            reference: https://github.com/THUDM/ImageReward/blob/c69c0f9bf3f385b7b02979746d93fc6acaec463f/ImageReward/ImageReward.py#L120-L136
            """
            text_inputs = model.blip.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=35,
                return_tensors="pt").to(device)
            image_embeds = model.blip.visual_encoder(images)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            text_output = model.blip.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            txt_features = text_output.last_hidden_state[:, 0, :].float()   # (feature_dim)
            rewards = model.mlp(txt_features)
            rewards = (rewards - model.mean) / model.std

            return rewards

        transform = model.preprocess

    else:
        raise ValueError(f"Unknown score name: {name}")

    if len(kwargs) > 0:
        warnings.warn(f"Unused keyword arguments: {kwargs}")

    return compute_score, transform
