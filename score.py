import json
import os
import logging
from collections import defaultdict
from typing import Callable, List

import accelerate
import click
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.dataset import ScoreDataset
from misc.scores import get_score
from misc.utils import EasyDict


@click.command(context_settings={'show_default': True})
@click.option(
    "--pickscore/--no-pickscore", default=False,
    help="Whether to compute the PickScore."
)
@click.option(
    "--hpsv2/--no-hpsv2", default=False,
    help="Whether to compute the Human Preference Score v2 (HPSv2)."
)
@click.option(
    "--aestheticv1/--no-aestheticv1", default=False,
    help="Whether to compute the Aesthetic score."
)
@click.option(
    "--aestheticv2/--no-aestheticv2", default=False,
    help="Whether to compute the Aesthetic score."
)
@click.option(
    "--clip/--no-clip", default=False,
    help="Whether to compute the CLIP score."
)
@click.option(
    "--imagereward/--no-imagereward", default=False,
    help="Whether to compute the CLIP score."
)
@click.option(
    "--dir", multiple=True, type=str, required=True,
    help=(
        "The path to the directory, which can be same as the `--output` "
        "argument of the inference.py script."
    )
)
@click.option(
    "--batch_size", default=32, type=int,
    help="The batch size for the DataLoader."
)
@click.option(
    "--num_workers", default=8, type=int,
    help="The number of workers for the DataLoader."
)
@click.option(
    "--hps_version", default="v2.0", type=click.Choice(["v2.0", "v2.1"]),
    help="The version of the Human Preference Score to compute."
)
def main(**kwargs):
    args = EasyDict(kwargs)
    if not args.hpsv2 and not args.pickscore and not args.aestheticv1 and not args.aestheticv2 and not args.clip:
        raise click.UsageError(
            "At least one of `--hpsv2` or `--pickscore` or `--aesthetic` "
            "or `--clip` must be set to `True`."
        )

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    logging.basicConfig(level=logging.WARNING)

    scores = dict()
    for directory in args.dir:
        accelerator.print(f"Processing {directory}")
        scores[directory] = defaultdict(list)

        # Calculate the PickScore
        if args.pickscore:
            compute_score, transform = get_score("pickscore", device)
            score = compute(args, directory, accelerator, compute_score, transform, "pickscore")
            scores[directory]["PickScore"] = score
            del compute_score, transform

        # Calculate the HPSv2 score
        if args.hpsv2:
            compute_score, transform = get_score("hpsv2", device, hps_version=args.hps_version)
            score = compute(args, directory, accelerator, compute_score, transform, "hpsv2")
            scores[directory]["hpsv2"] = score
            del compute_score, transform

        # Calculate the Aesthetic v1 score
        if args.aestheticv1:
            compute_score, transform = get_score("aesthetic", device, accelerator=accelerator, aesthetic_version="v1")
            score = compute(args, directory, accelerator, compute_score, transform, "aestheticv1")
            scores[directory]["aestheticv1"] = score
            del compute_score, transform

        # Calculate the Aesthetic v2 score
        if args.aestheticv2:
            compute_score, transform = get_score("aesthetic", device, accelerator=accelerator, aesthetic_version="v2")
            score = compute(args, directory, accelerator, compute_score, transform, "aestheticv2")
            scores[directory]["aestheticv2"] = score
            del compute_score, transform

        # Calculate the CLIP score
        if args.clip:
            compute_score, transform = get_score("clip", device)
            score = compute(args, directory, accelerator, compute_score, transform, "clip")
            scores[directory]["clip"] = score
            del compute_score, transform

        if args.imagereward:
            compute_score, transform = get_score("imagereward", device)
            score = compute(args, directory, accelerator, compute_score, transform, "imagereward")
            scores[directory]["imagereward"] = score
            del compute_score, transform

        name_len = max(len(score_name) for score_name in scores[directory].keys())
        for score_name, score in scores[directory].items():
            accelerator.print(f"{score_name:{name_len}}: {score:7.4f}")

    groups = defaultdict(list)
    for directory in args.dir:
        group_name = os.path.basename(directory).split('_')[0]
        groups[group_name].append(directory)

    for group, directories in groups.items():
        accelerator.print(f"Group: {group}")
        for directory in directories:
            accelerator.print(" ".join(f"{score}" for score in scores[directory].values()), end=" ")
        accelerator.print("")


def compute(
    args: EasyDict,
    image_dir: str,
    accelerator: accelerate.Accelerator,
    compute_score: Callable[[torch.Tensor, List[str]], torch.Tensor],
    transform: Callable[[Image.Image], torch.Tensor],
    score_name: str,
):
    device = accelerator.device

    # Load the dataset
    dataset = ScoreDataset(root=image_dir, transform=transform)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    # accelerator will handle the duplicates of last batch
    loader = accelerator.prepare(loader)

    # Load the scores from the cache
    cache_path = os.path.join(image_dir, f"{score_name}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            path2score = json.load(f)
    else:
        path2score = {}

    # Compute the scores for the images
    total_images = len(dataset)
    done_images = 0
    with tqdm(
        loader,
        ncols=0,
        leave=False,
        desc=score_name,
        disable=not accelerator.is_main_process
    ) as pbar:
        for batch in pbar:
            images = batch['image']
            prompts = batch['prompt']
            paths = batch['path']

            # Skip batch if all the images in the batch are already scored
            done = []
            scores = []
            for path in paths:
                if path in path2score:
                    done.append(True)
                    scores.append(path2score[path])
                else:
                    done.append(False)
            done = accelerator.gather_for_metrics(torch.tensor(done).to(device)).cpu()

            if not done.all():
                # Compute the scores for the images in the batch
                scores = compute_score(images, prompts)
            else:
                # The scores are already computed
                scores = torch.tensor(scores).to(device)

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
        # Save the scores to the cache
        with open(cache_path, "w") as f:
            json.dump(path2score, f)

    accelerator.wait_for_everyone()
    return average_score


if __name__ == "__main__":
    main()
