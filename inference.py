from functools import partial
import os

import accelerate
import click
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel)
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import (
    AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection)

from misc.dataset import PromptDataset
from misc.utils import EasyDict


@click.command(context_settings={'show_default': True})
@click.option(
    "--pretrained_model_name_or_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    type=str,
    help=(
        "Path to pretrained model or model identifier from "
        "huggingface.co/models."
    )
)
@click.option(
    "--variant", default=None, type=str,
    help=(
        "Variant of the model files of the pretrained model identifier "
        "from huggingface.co/models, 'e.g.' fp16"
    )
)
@click.option(
    "--sdxl/--no-sdxl", default=False,
    help="Whether the model is a Stable Diffusion XL model."
)
@click.option(
    "--unet", default=None, type=str, metavar="DIR",
    help=(
        "Path to the fine-tuned UNet model. If not provided, the UNet model "
        "from the `--pretrained_model_name_or_path` will be used."
    )
)
@click.option(
    "--unet_variant", default=None, type=str,
    help="The variant of the UNet model when `--unet` is provided."
)
@click.option(
    "--unet_subfolder", default=None, type=str,
    help="The subfolder of the UNet model when `--unet` is provided."
)
@click.option(
    "--scheduler", default='DDPM',
    type=click.Choice(["DDPM", "DDIM", "DPMSolver++", "SDXL"]),
    help="The sampling scheduler to use."
)
@click.option(
    "--num_inference_steps", default=50, type=int,
    help="Number of steps for the inference."
)
@click.option(
    "--guidance_scale", default=7.5, type=float,
    help="The guidance scale for the sampling."
)
@click.option(
    "--num_images_per_prompt", default=5, type=int,
    help="Number of images to generate per prompt."
)
@click.option(
    "--test_dataset", default='./data/pickapicv2_test', type=str, metavar="DIR",
    help=(
        "The root directory of the validation dataset. See the `ScoreDataset` "
        "class for more information on the dataset format."
    ),
)
@click.option(
    "--output", default="./output", type=str, metavar="DIR",
    help="The path where the generated image should be saved."
)
@click.option(
    "--batch_size", default=4, type=int,
    help="The number of images to generate in one batch."
)
@click.option(
    "--num_workers", default=8, type=int,
    help="The number of workers for the DataLoader."
)
@click.option(
    "--seed", default=0, type=int,
    help="A seed for reproducible training."
)
@click.option(
    "--mixed_precision", default='bf16',
    type=click.Choice(["no", "fp16", "bf16"]),
    help="The mixed_precision argument for initializing the accelerator."
)
def main(**kwargs):
    """Generate images from given prompt dataset using the trained model."""

    args = EasyDict(kwargs)
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Set the random seed for reproducibility
    accelerate.utils.set_seed(args.seed, device_specific=True)

    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    scheduler_classes = {
        "DDIM": DDIMScheduler,
        "DDPM": DDPMScheduler,
        "DPMSolver++": DPMSolverMultistepScheduler,
        "SDXL": EulerDiscreteScheduler,
    }

    # Load sampling scheduler
    scheduler_cls = scheduler_classes[args.scheduler]
    scheduler: DDIMScheduler = scheduler_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load the tokenizer and text encoder
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        variant=args.variant)
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device, dtype=dtype)

    if args.sdxl:
        # Load the second tokenizer and text encoder
        tokenizer_2 = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            variant=args.variant)
        text_encoder_2.requires_grad_(False)
        text_encoder_2 = text_encoder_2.to(device, dtype=dtype)

    # Load the VAE model
    if args.sdxl:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
            variant=args.variant)
    else:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            variant=args.variant,
            subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype=dtype)

    if args.unet is None:
        args.unet = args.pretrained_model_name_or_path
        args.unet_variant = args.variant
        args.unet_subfolder = "unet"
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.unet, variant=args.unet_variant, subfolder=args.unet_subfolder)
    unet.requires_grad_(False)
    unet = unet.to(device, dtype=dtype)

    if args.sdxl:
        pipeline = StableDiffusionXLPipeline(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
        )
    else:
        pipeline = StableDiffusionPipeline(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    pipeline.set_progress_bar_config(disable=True)

    def encode_prompt(prompts, tokenizers, text_encoders, random_drop_prompt_probability=0, is_train=False):
        prompt_embeds_list = []

        captions = []
        for caption in prompts:
            if is_train and torch.rand([]).item() < random_drop_prompt_probability:
                captions.append("")
            else:
                captions.append(caption)

        with torch.no_grad():
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)

                if args.sdxl:
                    prompt_embeds = text_encoder(
                        text_input_ids, output_hidden_states=True, return_dict=False)
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds[-1][-2]
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)
                else:
                    prompt_embeds = text_encoder(text_input_ids, return_dict=False)[0]

        if args.sdxl:
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
            return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}
        else:
            return {"prompt_embeds": prompt_embeds}

    if args.sdxl:
        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]
    else:
        tokenizers = [tokenizer]
        text_encoders = [text_encoder]
    encode_prompt_fn = partial(
        encode_prompt,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
    )

    dataset = PromptDataset(args.test_dataset)
    loader = DataLoader(
        dataset,
        batch_size=max(args.batch_size // args.num_images_per_prompt, 1),
        num_workers=args.num_workers,
    )
    total_prompts = len(dataset)
    num_digits = len(str(total_prompts - 1))

    # The accelerator will handle the duplicates of last batch
    loader = accelerator.prepare(loader)

    total_images = total_prompts * args.num_images_per_prompt
    done_images = 0
    with tqdm(loader, ncols=0, disable=not accelerator.is_main_process) as pbar:
        for batch_index, batch in enumerate(pbar):
            prompts = batch['prompt']
            B = len(prompts)

            # Base seed for each prompt
            seeds = torch.arange(
                args.seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index),
                args.seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index) + B,
                device=device)
            # Shift base seeds for images in the same prompt
            seeds = [seeds + i * total_prompts
                     for i in range(args.num_images_per_prompt)]
            seeds = torch.stack(seeds, dim=1).view(-1)

            # Get prompt embedding manually to supress the warning of long text
            embeds = encode_prompt_fn(prompts)
            prompt_embeds = embeds["prompt_embeds"]
            _, S, D = prompt_embeds.shape
            prompt_embeds = prompt_embeds.unsqueeze(1).expand(B, args.num_images_per_prompt, S, D)
            prompt_embeds = prompt_embeds.reshape(-1, S, D)

            if args.sdxl:
                pooled_prompt_embeds = embeds["pooled_prompt_embeds"]
                _, D = pooled_prompt_embeds.shape
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(B, args.num_images_per_prompt, D)
                pooled_prompt_embeds = pooled_prompt_embeds.reshape(-1, D)
            else:
                pooled_prompt_embeds = torch.empty(B * args.num_images_per_prompt)

            # Split the prompt_embeds and seeds into batches to avoid OOM
            for prompt_embeds_batch, pooled_prompt_embeds_batch, seeds_batch in zip(
                prompt_embeds.split(args.batch_size),
                pooled_prompt_embeds.split(args.batch_size),
                seeds.split(args.batch_size),
            ):
                all_done = True
                # Check if all images corresponding to `seeds_batch` are already generated
                for seed in seeds_batch:
                    # The index in the dataset
                    dataset_index = (seed - args.seed) % total_prompts
                    # The n-th images corresponding to the prompt
                    image_index_in_prompt = (seed - args.seed) // total_prompts
                    # Skip padding (DDP sampler duplicates)
                    if image_index_in_prompt >= args.num_images_per_prompt:
                        continue
                    # The index of the prompt in this batch
                    prompt_index = (dataset_index % (B * accelerator.num_processes)) % B
                    # Prompt
                    prompt = prompts[prompt_index]
                    # The directory to save the images
                    dir_path = os.path.join(
                        args.output, f"{dataset_index:0{num_digits}d}")
                    # The path to the prompt file
                    prompt_path = os.path.join(dir_path, "caption.txt")
                    if os.path.exists(prompt_path):
                        # Check if the prompt is the same
                        with open(prompt_path, "r") as f:
                            if f.read().strip() != prompt.strip():
                                raise ValueError(
                                    f"Prompt in {prompt_path} is different from "
                                    f"the current prompt: {prompt}"
                                )
                    else:
                        # Save the prompt
                        os.makedirs(dir_path, exist_ok=True)
                        with open(prompt_path, "w") as f:
                            f.write(prompts[prompt_index])
                    # The path to the image
                    image_path = os.path.join(dir_path, f"{seed.item()}.png")
                    if not os.path.exists(image_path):
                        all_done = False

                if not all_done:
                    if args.sdxl:
                        pipeline_kwargs = {"pooled_prompt_embeds": pooled_prompt_embeds_batch}
                    else:
                        pipeline_kwargs = {}
                    # Generate the images
                    generator = [
                        torch.Generator(device=device).manual_seed(seed.item())
                        for seed in seeds_batch]
                    images_batch = pipeline(
                        prompt_embeds=prompt_embeds_batch,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        output_type='pt',
                        **pipeline_kwargs,
                    ).images
                    images_batch = images_batch.float()

                    # Save the images
                    for seed, image in zip(seeds_batch, images_batch):
                        # The index in the dataset
                        dataset_index = (seed - args.seed) % total_prompts
                        # The n-th images corresponding to the prompt
                        image_index_in_prompt = (seed - args.seed) // total_prompts
                        # Skip padding (DDP sampler duplicates)
                        if image_index_in_prompt >= args.num_images_per_prompt:
                            continue
                        # The index of the prompt in this batch
                        prompt_index = (dataset_index % (B * accelerator.num_processes)) % B
                        # Prompt
                        prompt = prompts[prompt_index]
                        # The directory to save the images
                        dir_path = os.path.join(
                            args.output, f"{dataset_index:0{num_digits}d}")
                        # The path to the image
                        image_path = os.path.join(dir_path, f"{seed.item()}.png")
                        # Save the image
                        save_image(image, image_path)

                done_images = min(
                    done_images + len(seeds_batch) * accelerator.num_processes,
                    total_images)
                pbar.set_postfix_str(f"Generated {done_images}/{total_images} images")

            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
