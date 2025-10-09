import json
import os
from functools import partial
from typing import Callable

import accelerate
import click
import deepspeed
import diffusers
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import (
    AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection)

from misc.buffer import ReplayBuffer
from misc.dataset import TrainingDataset, PromptDataset, ScoreDataset
from misc.scores import get_score
from misc.patch import (
    DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler,
    StableDiffusionPipeline, StableDiffusionXLPipeline)
from misc.utils import CommandAwareConfig, EasyDict


@click.command(cls=CommandAwareConfig, context_settings={'show_default': True})
@click.option(
    '--config', default=None, type=str, metavar="FILE",
    help=(
        "Path to the config file. The command line arguments will overwrite "
        "the config file."
    )
)
@click.option(
    "--logdir", default="./logs/sd15_diffusion-dro", type=str, metavar="DIR",
    help=(
        "The output directory where the model predictions and checkpoints "
        "will be written."
    )
)
@click.option(
    "--seed", default=0, type=int,
    help="A seed for reproducible training."
)
@click.option(
    "--pretrained_model_name_or_path",
    default="stable-diffusion-v1-5/stable-diffusion-v1-5", type=str,
    help="Path to pretrained model or model identifier.",
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
    "--train_dataset", default="./data/pickapicv2_hpsv2_500", type=str,
    metavar="DIR",
    help=(
        "The root directory of the training dataset. See the "
        "`TrainingDataset` class for more information on the dataset format."
    ),
)
@click.option(
    "--resolution", default=512, type=int,
    help=(
        "The resolution for input images, all the images in the "
        "train/validation dataset will be resized to this resolution."
    ),
)
@click.option(
    "--random_crop/--no-random_crop", default=False,
    help=(
        "Whether to random crop the input images to the resolution. If "
        "not set, the images will be center cropped. The images will be "
        "resized to the resolution first before cropping."
    ),
)
@click.option(
    "--random_flip/--no-random_flip", default=True,
    help="whether to randomly flip images horizontally",
)
@click.option(
    "--random_drop_prompt_probability", default=0.2, type=float,
    help=(
        "The probability to drop the prompt during training. Set to 0 to "
        "disable."
    ),
)
@click.option(
    "--validation_dataset", default="./data/pickapicv2_test", type=str,
    metavar="DIR",
    help=(
        "The root directory of the validation dataset. See the "
        "`TrainingDataset` class for the dataset format."
    ),
)
@click.option(
    "--validation_scheduler", default="DDPM",
    type=click.Choice(["DDPM", "DDIM", "DPMSolver++"]),
    help="The scheduler to use for the validation.",
)
@click.option(
    "--validation_num_inference_steps", default=50, type=int,
    help=(
        "Number of steps for the inference. The RL training only updates "
        "these many steps."
    )
)
@click.option(
    "--validation_guidance_scale", default=7.5, type=float,
    help="Guidance scale for the validation."
)
@click.option(
    "--score", default=None, type=click.Choice([
        "pickscore", "hpsv2", "aestheticv1", "aestheticv2", "clip",
        "imagereward"
    ]),
    help=(
        "The score to compute for full validation set. If not set, the score "
        "will not be computed."
    )
)
@click.option(
    "--score_batch_size", default=4, type=int,
)
@click.option(
    "--score_num_images_per_prompt", default=1, type=int,
)
@click.option(
    "--batch_size", default=4, type=int,
    help="Batch size (per device) for each forward and backward step."
)
@click.option(
    "--num_steps", default=25600, type=int,
    help="Number of forward and backward steps to take."
)
@click.option(
    "--gradient_accumulation_steps", default=16, type=int,
    help="Number of gradient accumulations steps per update."
)
@click.option(
    "--gradient_checkpointing/--no-gradient_checkpoint", default=False,
    help=(
        "Whether or not to use gradient checkpointing to save memory at "
        "the expense of slower backward pass."
    )
)
@click.option(
    "--learning_rate", default=1e-4, type=float,
    help="Initial learning rate (after the potential warmup period).",
)
@click.option(
    "--scale_lr/--no-scale_lr", default=False,
    help=(
        "Scale the learning rate by the gradient accumulation steps, batch "
        "size and number of GPUs."
    )
)
@click.option(
    "--lr_scheduler", default="constant",
    type=click.Choice([
        "linear", "cosine", "cosine_with_restarts", "polynomial",
        "constant", "constant_with_warmup"
    ]),
    help='The scheduler type to use.',
)
@click.option(
    "--lr_warmup_steps", default=0, type=int,
    help="Number of steps for the warmup in the lr scheduler."
)
@click.option(
    "--max_grad_norm", default=1.0, type=float,
    help="Max gradient norm. Set to 0 to disable gradient clipping."
)
@click.option(
    "--buffer_size", default=4, type=int,
    help=(
        "Size of the replay buffer. Each timestep takes up one slot in the "
        "replay buffer."
    )
)
@click.option(
    "--buffer_scheduler", default='DPMSolver++',
    type=click.Choice(["DDPM", "DDIM", "DPMSolver++"]),
    help="The scheduler to use for the replay buffer."
)
@click.option(
    "--buffer_batch_size", default=4, type=int,
    help="Batch size (per device) for updating the replay buffer."
)
@click.option(
    "--buffer_batch_accumulation", default=1, type=int,
    help="Number of batches to accumulate before updating the replay buffer."
)
@click.option(
    "--buffer_num_inference_steps", default=20, type=int,
    help=(
        "Number of denoising steps for sampling images when updating the "
        "replay buffer."
    )
)
@click.option(
    "--buffer_guidance_scale", default=1.0, type=float,
    help="Guidance scale for the replay buffer."
)
@click.option(
    "--buffer_sample_steps", default=1, type=int,
    help="Append new online samples to the buffer every these many steps."
)
@click.option(
    "--buffer_update_steps", default=16, type=int,
    help="The unet in the buffer is updated every these many steps."
)
@click.option(
    "--buffer_perturb_timesteps/--no-buffer_perturb_timesteps", default=True,
    help="Whether to perturb the timesteps for the buffer sampling."
)
@click.option(
    "--buffer_sync/--no-buffer_sync", default=False,
    help="Whether to synchronize the buffer across all GPUs."
)
@click.option(
    "--margin", default=0.001, type=float,
    help="The margin for the Hinge loss."
)
@click.option(
    "--allow_tf32/--no-allow_tf32", default=False,
    help=(
        "Whether or not to allow TF32 on Ampere GPUs. Can be used to "
        "speed up training."
    ),
)
@click.option(
    "--dataloader_num_workers", default=8, type=int,
    help=(
        "Number of subprocesses to use for data loading. 0 means that "
        "the data will be loaded in the main process."
    ),
)
@click.option(
    "--mixed_precision", default='bf16',
    type=click.Choice(["no", "fp16", "bf16"]),
    help="Whether to use mixed precision training."
)
@click.option(
    "--use_ema/--no-use_ema", default=True,
    help="Whether to use exponential moving average for the policy network."
)
@click.option(
    "--offload_ema/--no-offload_ema", default=True,
    help="Whether to offload the EMA model to CPU."
)
@click.option(
    "--checkpointing_steps", default=1280, type=int,
    help=(
        "Save a checkpoint of the training state every these many steps."
        "These checkpoints are only suitable for resuming training using "
        "`--resume`."
    ),
)
@click.option(
    "--resume/--no-resume", default=False,
    help="Whether to resume training from the state."
)
def main(**kwargs):
    """Fine-tune a Stable Diffusion model with Diffusion-DRO."""

    deepspeed.utils.logger.setLevel("WARNING")

    args = EasyDict(kwargs)
    gradient_accumulation_plugin = accelerate.utils.GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_with_dataloader=False,
    )
    accelerator = accelerate.Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    device = accelerator.device

    # Set the random seed for reproducibility
    accelerate.utils.set_seed(args.seed, device_specific=True)

    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load noise scheduler, tokenizer and models.
    scheduler_classes = {
        "DDIM": DDIMScheduler,
        "DDPM": DDPMScheduler,
        "DPMSolver++": DPMSolverMultistepScheduler,
    }

    # Load training scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load schedulers for validation and buffer sampling
    validation_scheduler_cls = scheduler_classes[args.validation_scheduler]
    validation_scheduler: DDPMScheduler = validation_scheduler_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    buffer_scheduler_cls = scheduler_classes[args.buffer_scheduler]
    buffer_scheduler: DDPMScheduler = buffer_scheduler_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load the tokenizer and text encoder
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        variant=args.variant,
        device_map={"": str(device)},
        torch_dtype=dtype)
    text_encoder.requires_grad_(False)

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
            variant=args.variant,
            device_map={"": str(device)},
            torch_dtype=dtype)
        text_encoder_2.requires_grad_(False)

    # Load the VAE model
    if args.sdxl:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
            variant=args.variant,
            device_map={"": str(device)},
            torch_dtype=dtype)
    else:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            variant=args.variant,
            subfolder="vae",
            device_map={"": str(device)},
            torch_dtype=dtype)
    vae.requires_grad_(False)

    # Load the UNet models
    unet_ref: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        variant=args.variant,
        subfolder="unet",
        device_map={"": str(device)},
        torch_dtype=dtype)
    unet_ref.requires_grad_(False)

    unet_policy: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        variant=args.variant,
        subfolder="unet",
        device_map={"": str(device)},
        torch_dtype=dtype)
    unet_policy.requires_grad_(False)

    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        variant=args.variant,
        subfolder="unet",
        device_map={"": str(device)},
        torch_dtype=dtype)
    unet.requires_grad_(True)

    # Cast the trainable parameters to the desired dtype
    if dtype == torch.float16:
        diffusers.training_utils.cast_training_params(unet, dtype=torch.float32)

    if args.use_ema:
        unet_ema = diffusers.training_utils.EMAModel(
            unet.parameters(), foreach=True)

        if args.offload_ema:
            unet_ema.to('cpu')
            unet_ema.pin_memory()
        else:
            unet_ema.to(device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere and later CUDA devices.
    # cf https://huggingface.co/docs/diffusers/optimization/fp16#tensorfloat-32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # create custom saving & loading hooks so that
    # `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if len(models) != 1:
                raise AssertionError("Only one model is supported for now.")
            unet = accelerator.unwrap_model(models[0])
            if not isinstance(unet, UNet2DConditionModel):
                raise AssertionError(
                    "Only UNet model is supported for now, got {unet.__class__.__name__}.")

            if args.use_ema:
                torch.save(unet_ema.state_dict(), os.path.join(output_dir, "ema.pt"))
            unet.save_pretrained(os.path.join(output_dir, "unet"))

            if weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        if len(models) != 1:
            raise AssertionError("Only one model is supported for now.")
        unet = accelerator.unwrap_model(models[0])
        if not isinstance(unet, UNet2DConditionModel):
            raise AssertionError(
                "Only UNet model is supported for now, got {unet.__class__.__name__}.")

        if args.use_ema:
            state_dict = torch.load(
                os.path.join(input_dir, "ema.pt"), map_location="cpu")
            unet_ema.load_state_dict(state_dict)
            if args.offload_ema:
                unet_ema.to('cpu')
                unet_ema.pin_memory()
            else:
                unet_ema.to(device)
            del state_dict
        load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
        unet.register_to_config(**load_model.config)
        unet.load_state_dict(load_model.state_dict())
        del load_model

        models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps // args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.num_steps // args.gradient_accumulation_steps * accelerator.num_processes,
    )

    if args.sdxl:
        validation_pipeline = StableDiffusionXLPipeline(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            validation_scheduler,
        )

        buffer_pipeline = StableDiffusionXLPipeline(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet_policy,
            buffer_scheduler,
        )
    else:
        validation_pipeline = StableDiffusionPipeline(
            vae,
            text_encoder,
            tokenizer,
            unet,
            validation_scheduler,
            safety_checker=None,            # Disable NSFW checker
            feature_extractor=None,         # Disable NSFW checker
            requires_safety_checker=False,  # Disable NSFW checker
        )
        buffer_pipeline = StableDiffusionPipeline(
            vae,
            text_encoder,
            tokenizer,
            unet_policy,
            buffer_scheduler,
            safety_checker=None,            # Disable NSFW checker
            feature_extractor=None,         # Disable NSFW checker
            requires_safety_checker=False,  # Disable NSFW checker
        )
    validation_pipeline.set_progress_bar_config(disable=True)
    buffer_pipeline.set_progress_bar_config(disable=True)

    def encode_prompt(prompts, tokenizers, text_encoders, random_drop_prompt_probability=0, is_train=False):
        prompt_embeds_list = []

        captions = []
        for caption in prompts:
            if is_train and torch.rand(1) < random_drop_prompt_probability:
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
        random_drop_prompt_probability=args.random_drop_prompt_probability)

    train_dataset = TrainingDataset(
        args.train_dataset, args.resolution, args.random_flip, args.random_crop)

    def collate_fn(batch_list):
        batch = dict()
        for key in batch_list[0].keys():
            batch[key] = [batch[key] for batch in batch_list]
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key], dim=0)
        return batch

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.buffer_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    unet, optimizer, lr_scheduler, train_loader = \
        accelerator.prepare(unet, optimizer, lr_scheduler, train_loader)

    def infinite_loop(loader):
        while True:
            for batch in loader:
                yield batch

    train_loader = infinite_loop(train_loader)

    # Create the log directory and save the arguments
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.logdir)
    else:
        writer = None

    if args.resume:
        state_path = os.path.join(args.logdir, "state")
        accelerator.load_state(state_path)
        training_state = torch.load(os.path.join(state_path, "training_state.pt"))
        init_step = training_state["step"] + 1
        resume_path = state_path
        replay_buffer = ReplayBuffer(args.buffer_size, time_key="t")
    else:
        if accelerator.is_main_process:
            writer.add_text(
                "training_config.json", f"```\n{json.dumps(args, indent=2)}\n```")
            with open(os.path.join(args.logdir, "training_config.json"), "w") as f:
                json.dump(args, f, indent=4)
        init_step = 0
        resume_path = None
        replay_buffer = ReplayBuffer(args.buffer_size, time_key="t")
    unet_policy.load_state_dict(accelerator.unwrap_model(unet).state_dict())

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    accelerator.print(
        f"Training Configurations\n"
        f"- Num GPUs                : {accelerator.num_processes}\n"
        f"- Batch Size per Device   : {args.batch_size}\n"
        f"- Gradient Accumulation   : {args.gradient_accumulation_steps}\n"
        f"- Effective Batch Size    : {effective_batch_size}\n"
        f"- Dataset Size            : {len(train_dataset)}\n"
        f"- Total Optimization Steps: {args.num_steps // args.gradient_accumulation_steps}\n"
        f"- Resuming from states    : {resume_path}\n"
        f"- Mixed Precision         : {accelerator.mixed_precision}\n"
        f"- Training Configurations : {os.path.join(args.logdir, 'training_config.json')}"
    )

    step_loss = EasyDict()
    step_loss.loss = 0
    step_loss.diff = 0
    step_loss.expert = 0
    step_loss.policy = 0
    step_loss.expert0 = 0
    step_loss.policy0 = 0
    step_loss.expert_diff = 0
    step_loss.policy_diff = 0

    progress_bar = tqdm(
        range(init_step + 1, args.num_steps + 1),
        total=args.num_steps,
        initial=init_step,
        ncols=0,
        desc="Steps",
        disable=not accelerator.is_main_process,
    )

    unet.train()
    for step in progress_bar:
        # Update the replay buffer
        if len(replay_buffer) == 0 or (step % args.buffer_sample_steps == 0):
            if len(replay_buffer) == 0:
                # If the buffer is empty, sample a batch of data
                buffer_batch_accumulation = args.batch_size // args.buffer_batch_size + (args.batch_size % args.buffer_batch_size > 0)
                buffer_batch_accumulation = max(buffer_batch_accumulation, args.buffer_batch_accumulation)
            else:
                buffer_batch_accumulation = args.buffer_batch_accumulation

            for _ in range(buffer_batch_accumulation):
                batch = next(train_loader)
                # Get prompt embedding
                embeds = encode_prompt_fn(batch["prompt"], is_train=True)
                replay_buffer.push("c", embeds["prompt_embeds"])
                if args.sdxl:
                    replay_buffer.push("add_time_ids", batch["add_time_ids"])
                    replay_buffer.push("c_pooled", embeds["pooled_prompt_embeds"])
                    pipeline_kwargs = {"pooled_prompt_embeds": embeds["pooled_prompt_embeds"]}
                else:
                    pipeline_kwargs = {}

                # Sample expert demonstration
                x0 = vae.encode(batch["image"].to(dtype=dtype)).latent_dist.sample()
                x0 = x0 * vae.config.scaling_factor
                replay_buffer.push("x0", x0)

                # Sample x_T
                latent_C = buffer_pipeline.unet.config.in_channels
                latent_H = buffer_pipeline.unet.config.sample_size
                latent_W = buffer_pipeline.unet.config.sample_size
                xT = torch.randn(args.buffer_batch_size, latent_C, latent_H, latent_W, device=device, dtype=dtype)
                replay_buffer.push("xt", xT, is_time_dependent=True)

                # Sample timesteps
                if args.buffer_perturb_timesteps:
                    step_ratio = buffer_pipeline.scheduler.config.num_train_timesteps // args.buffer_num_inference_steps
                    timesteps = (torch.arange(0, args.buffer_num_inference_steps) * step_ratio).round().flip(0)
                    perturb = torch.randint(0, step_ratio, (1, args.buffer_batch_size))
                    timesteps = timesteps[:, None] + perturb    # [num_inference_steps, batch_size]
                else:
                    timesteps = None

                # Sample policy demonstrations
                with accelerator.autocast():
                    buffer_pipeline(
                        timesteps=timesteps,
                        latents=xT,
                        prompt_embeds=embeds["prompt_embeds"],
                        num_inference_steps=args.buffer_num_inference_steps,
                        guidance_scale=args.buffer_guidance_scale,
                        output_type='latent',
                        callback_on_step_end=replay_buffer,
                        **pipeline_kwargs,
                    )
                # Push the demonstrations to the replay buffer
                replay_buffer.commit()

                del x0, xT, timesteps, embeds

            if accelerator.is_main_process:
                writer.add_scalar("params/buffer_size", len(replay_buffer), step)

        with accelerator.accumulate(unet):
            # Variable          Paper
            # -----------------------------
            # t              -> $t$
            # c              -> $c$
            # x0             -> $x_0$
            # eps_policy     -> $\epsilon_\theta(x_t, c)$
            # eps_expert     -> $\bar{\epsilon}$
            # xt_policy      -> $x_t$
            # xt_expert      -> $\bar{x}_t$
            # unet           -> $p_{\phi}$
            # unet_ref       -> $p_{\theta_{ref}}$
            # unet_policy    -> $p_{\theta}$

            # Sample demonstration pairs from the replay buffer
            samples = replay_buffer.sample(args.batch_size, device=device)
            t = samples.t
            c = samples.c
            x0_expert = samples.x0
            xt_policy = samples.xt
            eps_policy = samples.eps
            if args.sdxl:
                added_cond_kwargs = {
                    "time_ids": torch.cat([samples.add_time_ids, samples.add_time_ids]).to(dtype=dtype),
                    "text_embeds": torch.cat([samples.c_pooled, samples.c_pooled]).to(dtype=dtype)}
            else:
                added_cond_kwargs = {}

            # forward diffusion
            eps_expert = torch.randn_like(x0_expert)
            xt_expert = noise_scheduler.add_noise(x0_expert, eps_expert, t)

            xt = torch.cat([xt_expert, xt_policy], dim=0)
            eps = torch.cat([eps_expert, eps_policy], dim=0)
            t = torch.cat([t, t], dim=0)
            c = torch.cat([c, c], dim=0)

            pred_eps_ref = unet_ref(
                xt.to(dtype=dtype),
                t,
                c.to(dtype=dtype),
                added_cond_kwargs=added_cond_kwargs).sample
            losses0 = (pred_eps_ref - eps).square().mean(dim=[1, 2, 3])
            loss_expert0, loss_policy0 = losses0.chunk(2)

            pred_eps = unet(xt, t, c, added_cond_kwargs=added_cond_kwargs).sample
            losses = (pred_eps - eps).square().mean(dim=[1, 2, 3])
            loss_expert, loss_policy = losses.chunk(2)
            loss_expert_diff = loss_expert - loss_expert0
            loss_policy_diff = loss_policy - loss_policy0

            diff = loss_expert_diff - loss_policy_diff
            loss = torch.maximum(diff + args.margin, torch.zeros_like(diff)).mean()
            accelerator.backward(loss)

            step_loss.loss += accelerator.gather(loss.unsqueeze(0).detach()).cpu().mean()
            step_loss.diff += accelerator.gather(diff.detach()).cpu().mean()
            step_loss.expert += accelerator.gather(loss_expert.detach()).cpu().mean()
            step_loss.policy += accelerator.gather(loss_policy.detach()).cpu().mean()
            step_loss.expert0 += accelerator.gather(loss_expert0.detach()).cpu().mean()
            step_loss.policy0 += accelerator.gather(loss_policy0.detach()).cpu().mean()
            step_loss.expert_diff += accelerator.gather(loss_expert_diff.detach()).cpu().mean()
            step_loss.policy_diff += accelerator.gather(loss_policy_diff.detach()).cpu().mean()

            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Log the training loss
        if accelerator.sync_gradients and accelerator.is_main_process:
            for tag in step_loss.keys():
                step_loss[tag] /= args.gradient_accumulation_steps
                writer.add_scalar(f"loss/{tag}", step_loss[tag].float(), step)
            writer.add_scalar("params/lr", lr_scheduler.get_last_lr()[0], step)
            progress_bar.set_postfix_str(
                f"loss: {step_loss['loss']: .3E}, lr: {lr_scheduler.get_last_lr()[0]: .3E}")
            for tag in step_loss.keys():
                step_loss[tag] = 0

        # Update the EMA model
        if accelerator.sync_gradients and args.use_ema:
            if args.offload_ema:
                unet_ema.to(device, non_blocking=True)
            unet_ema.step(unet.parameters())
            if args.offload_ema:
                unet_ema.to("cpu", non_blocking=True)

        # Update the buffer unet
        if step % args.buffer_update_steps == 0:
            unet_policy.load_state_dict(accelerator.unwrap_model(unet).state_dict())

        # Save the training state and a checkpoint of the model
        if step % args.checkpointing_steps == 0:
            # State path
            state_path = os.path.join(args.logdir, "state")
            # Save and overwrite the training state
            accelerator.save_state(state_path)
            # Save ReplayBuffer
            replay_buffer.save(state_path, accelerator)
            # Checkpoint path
            ckpt_path = os.path.join(args.logdir, f"ckpt-{step}", "unet")
            if accelerator.is_main_process:
                # Save the training state
                torch.save(
                    {'step': step},
                    os.path.join(state_path, "training_state.pt"),
                )
                # Save unet weights
                unwrapped_unet: UNet2DConditionModel = accelerator.unwrap_model(unet)
                # Save the EMA model
                if args.use_ema:
                    unet_ema.store(unwrapped_unet.parameters())
                    unet_ema.copy_to(unwrapped_unet.parameters())
                    unwrapped_unet.save_pretrained(ckpt_path)
                    unet_ema.restore(unwrapped_unet.parameters())
                    progress_bar.write(f"Saved EMA weights to {ckpt_path}")
                else:
                    unwrapped_unet.save_pretrained(ckpt_path)
                    progress_bar.write(f"Saved weights to {ckpt_path}")

            if args.score is not None:
                if args.use_ema:
                    unet_ema.store(unet.parameters())
                    unet_ema.copy_to(unet.parameters())
                output_dir = os.path.join(args.logdir, "images", f"ckpt-{step}")
                validation_pipeline.unet = accelerator.unwrap_model(unet)
                score = log_score(
                    accelerator=accelerator,
                    writer=writer,
                    dataset=PromptDataset(args.validation_dataset),
                    pipeline=validation_pipeline,
                    encode_prompt_fn=encode_prompt_fn,
                    num_inference_steps=args.validation_num_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                    batch_size=args.score_batch_size,
                    num_images_per_prompt=args.score_num_images_per_prompt,
                    score_name=args.score,
                    output_dir=output_dir,
                    step=step,
                    root_seed=args.seed,
                    sdxl=args.sdxl,
                )
                if args.use_ema:
                    unet_ema.restore(unet.parameters())
                if accelerator.is_main_process:
                    progress_bar.write(f"Step: {step:5d}, {args.score}: {score:.6f}")

        # Wait for main processes to save the state
        accelerator.wait_for_everyone()
    # Destroy process group
    accelerator.end_training()


def log_score(
    accelerator: accelerate.Accelerator,
    writer: SummaryWriter,
    dataset: PromptDataset,
    pipeline: diffusers.DiffusionPipeline,
    encode_prompt_fn: Callable,
    num_inference_steps: int,
    guidance_scale: float,
    batch_size: int,
    num_images_per_prompt: int,
    score_name: str,
    output_dir: str,
    step: int,
    root_seed: int,
    sdxl: bool,
):
    device = accelerator.device

    loader = DataLoader(
        dataset,
        batch_size=max(batch_size // num_images_per_prompt, 1),
        num_workers=4,
    )
    total_prompts = len(dataset)
    num_digits = len(str(total_prompts - 1))

    loader = accelerator.prepare(loader)

    total_images = total_prompts * num_images_per_prompt
    done_images = 0
    with tqdm(
        loader,
        ncols=0,
        leave=False,
        desc=f"Evaluating {score_name} 1/2",
        disable=not accelerator.is_main_process,
    ) as pbar:
        for batch_index, batch in enumerate(pbar):
            prompts = batch['prompt']
            B = len(prompts)

            # Base seed for each prompt
            seeds = torch.arange(
                root_seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index),
                root_seed + batch_index * (B * accelerator.num_processes) + (B * accelerator.process_index) + B,
                device=device)
            # Shift base seeds for images in the same prompt
            seeds = [seeds + i * total_prompts
                     for i in range(num_images_per_prompt)]
            seeds = torch.stack(seeds, dim=1).view(-1)

            # Get prompt embedding manually to supress the warning of long text
            embeds = encode_prompt_fn(prompts)
            prompt_embeds = embeds["prompt_embeds"]
            _, S, D = prompt_embeds.shape
            prompt_embeds = prompt_embeds.unsqueeze(1).expand(B, num_images_per_prompt, S, D)
            prompt_embeds = prompt_embeds.reshape(-1, S, D)

            if sdxl:
                pooled_prompt_embeds = embeds["pooled_prompt_embeds"]
                _, D = pooled_prompt_embeds.shape
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(B, num_images_per_prompt, D)
                pooled_prompt_embeds = pooled_prompt_embeds.reshape(-1, D)
            else:
                pooled_prompt_embeds = torch.empty(B * num_images_per_prompt)

            # Split the prompt_embeds and seeds into batches to avoid OOM
            for prompt_embeds_batch, pooled_prompt_embeds_batch, seeds_batch in zip(
                prompt_embeds.split(batch_size),
                pooled_prompt_embeds.split(batch_size),
                seeds.split(batch_size),
            ):
                if sdxl:
                    pipeline_kwargs = {"pooled_prompt_embeds": pooled_prompt_embeds_batch}
                else:
                    pipeline_kwargs = {}
                # Generate the images
                generator = [
                    torch.Generator(device=device).manual_seed(seed.item())
                    for seed in seeds_batch]
                with accelerator.autocast():
                    images_batch = pipeline(
                        prompt_embeds=prompt_embeds_batch,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        output_type='pt',
                        **pipeline_kwargs,
                    ).images
                images_batch = images_batch.float()

                # Save the images
                for seed, image in zip(seeds_batch, images_batch):
                    # The index in the dataset
                    dataset_index = (seed - root_seed) % total_prompts
                    # The n-th images corresponding to the prompt
                    image_index_in_prompt = (seed - root_seed) // total_prompts
                    # Skip padding (DDP sampler duplicates)
                    if image_index_in_prompt >= num_images_per_prompt:
                        continue
                    # The index of the prompt in this batch
                    prompt_index = (dataset_index % (B * accelerator.num_processes)) % B
                    # The directory to save the images
                    dir_path = os.path.join(
                        output_dir, f"{dataset_index:0{num_digits}d}")
                    os.makedirs(dir_path, exist_ok=True)
                    # The path to the image
                    image_path = os.path.join(dir_path, f"{seed.item()}.png")
                    # Save the image
                    save_image(image, image_path)
                    # The path to the prompt file
                    prompt_path = os.path.join(dir_path, "caption.txt")
                    # Save the prompt
                    with open(prompt_path, "w") as f:
                        f.write(prompts[prompt_index])

                done_images = min(
                    done_images + len(seeds_batch) * accelerator.num_processes,
                    total_images)
                pbar.set_postfix_str(f"Generated {done_images}/{total_images} images")

            accelerator.wait_for_everyone()

    compute_score, transform = get_score(score_name, device)

    # Load the dataset
    dataset = ScoreDataset(root=output_dir, transform=transform)
    loader = DataLoader(dataset, batch_size, num_workers=4)

    # accelerator will handle the duplicates of last batch
    loader = accelerator.prepare(loader)

    # Load the scores from the cache
    cache_path = os.path.join(output_dir, f"{score_name}.json")
    path2score = {}

    # Compute the scores for the images
    total_images = len(dataset)
    done_images = 0
    with tqdm(
        loader,
        ncols=0,
        leave=False,
        desc=f"Evaluating {score_name} 2/2",
        disable=not accelerator.is_main_process
    ) as pbar:
        for batch in pbar:
            images = batch['image']
            prompts = batch['prompt']
            paths = batch['path']

            scores = compute_score(images, prompts)
            paths = accelerator.gather_for_metrics(paths, use_gather_object=True)
            scores = accelerator.gather_for_metrics(scores).cpu()

            for path, score in zip(paths, scores):
                path2score[path] = score.item()

            done_images += len(scores)
            pbar.set_postfix_str(f"Processed {done_images}/{total_images} images")

    # Sanity check
    assert len(path2score) == total_images, f"{len(path2score)} != {total_images}"

    average_score = sum(path2score.values()) / len(path2score)
    if accelerator.is_main_process:
        # Log the average score
        writer.add_scalar(f"metrics/{score_name}", average_score, step)
        # Save the scores to the cache file
        with open(cache_path, "w") as f:
            json.dump(path2score, f)

    return average_score


if __name__ == "__main__":
    main()
