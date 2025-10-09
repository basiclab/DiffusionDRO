import json
import os

import click
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from misc.utils import EasyDict


@click.command(context_settings={'show_default': True})
@click.option(
    "--cache", default="./data/cache", type=str,
    help="Cache directory for the `hf_hub_download`."
)
@click.option(
    "--output", default="./data/hpdv2_benchmark", type=str,
    help="The path where the labels should be saved."
)
def main(**kwargs):
    args = EasyDict(kwargs)
    cache_dir = os.path.join(args.cache, "hpdv2_benchmark")
    os.makedirs(cache_dir, exist_ok=True)

    repo_id = "zhwang/HPDv2"
    filenames = [
        "benchmark/anime.json",
        "benchmark/concept-art.json",
        "benchmark/paintings.json",
        "benchmark/photo.json",
    ]

    captions = []
    for fname in filenames:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=fname,
            local_dir=cache_dir,
        )
        with open(os.path.join(cache_dir, fname), "r") as f:
            captions_json = json.load(f)
        captions.extend(captions_json)

    num_digits = len(str(len(captions)))

    os.makedirs(args.output, exist_ok=True)
    saving_counter = 0
    with tqdm(captions, desc="Saving Test Prompts") as pbar:
        for caption in pbar:
            if not caption.strip():
                continue
            target_dir = os.path.join(args.output, f"{saving_counter:0{num_digits}}")
            caption_path = os.path.join(target_dir, "caption.txt")
            os.makedirs(target_dir, exist_ok=True)
            with open(caption_path, "w") as f:
                f.write(caption.strip())

            saving_counter += 1
            pbar.set_postfix_str(
                f"# saved images: {saving_counter}/{len(captions)} "
                f"({saving_counter / len(captions):.2%})")


if __name__ == '__main__':
    main()
