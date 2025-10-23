import argparse
import os
from functools import partial

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


EXAMPLE_PROMPTS = [
    "A Pixar lemon wearing sunglasses on a beach.",
    "A dragon sitting on a couch in a digital illustration.",
]
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate(prompt, pipes, **kwargs):
    images = []
    with torch.inference_mode():
        for pipe in pipes:
            image = pipe(prompt, **kwargs).images[0]
            images.append(image)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inference_steps', type=int, default=20,
                        help='Number of inference steps for generation')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for generation')
    parser.add_argument('--port', type=int, default=5678,
                        help='Port to run the Gradio app on')
    parser.add_argument('--root_path', type=str, default='/gradio',
                        help='Root path for the Gradio app')
    args = parser.parse_args()

    args.num_inference_steps = int(os.getenv('NUM_INFERENCE_STEPS', args.num_inference_steps))
    args.guidance_scale = float(os.getenv('GUIDANCE_SCALE', args.guidance_scale))
    args.port = int(os.getenv('PORT', args.port))
    args.root_path = os.getenv('ROOT_PATH', args.root_path)

    print("Loading SD1.5 pipeline...")
    sd15_pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ).to(dtype=torch.bfloat16, device=device)

    print("Loading DRO UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "ylwu/diffusion-dro-sd1.5",
        subfolder="unet",
    ).to(dtype=torch.bfloat16, device=device)

    ddro_pipe = StableDiffusionPipeline(
        vae=sd15_pipe.vae,
        text_encoder=sd15_pipe.text_encoder,
        tokenizer=sd15_pipe.tokenizer,
        unet=unet,
        scheduler=sd15_pipe.scheduler,
        safety_checker=sd15_pipe.safety_checker,
        feature_extractor=sd15_pipe.feature_extractor,
        image_encoder=sd15_pipe.image_encoder,
        requires_safety_checker=sd15_pipe.config.requires_safety_checker
    )

    # Determine image dimensions for better UI layout
    # vae_scale_factor = 2 ** (len(sd15_pipe.vae.config.block_out_channels) - 1)
    # height = sd15_pipe.unet.config.sample_size * vae_scale_factor
    # width = sd15_pipe.unet.config.sample_size * vae_scale_factor

    with gr.Blocks(theme=gr.themes.Default()) as demo:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...", lines=1)
        gr.Examples(
            examples=[[prompt] for prompt in EXAMPLE_PROMPTS],
            inputs=[prompt],
            label="Example Prompts",
            examples_per_page=2,
        )
        gr.Markdown("⚠️ **Safety Notice:** This demo uses the built-in Safety Checker. "
                    "If the generated image appears completely black, it means the content "
                    "was detected as sensitive and has been automatically blocked.")
        with gr.Row():
            generate_btn = gr.Button("Generate 🚀")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Diffusion-DRO**")
                ddro_image = gr.Image(label="Fine-tuned Model Output", show_download_button=True)
            with gr.Column():
                gr.Markdown("**SD1.5**")
                sd15_image = gr.Image(label="Base Model Output", show_download_button=True)
        generate_btn.click(
            fn=partial(
                generate,
                pipes=[ddro_pipe, sd15_pipe],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale),
            inputs=[prompt],
            outputs=[ddro_image, sd15_image],
            api_name="generate")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        root_path="/gradio",
    )


if __name__ == "__main__":
    main()
