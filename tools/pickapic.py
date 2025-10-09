import copy
import io
import json
import os
from collections import defaultdict
from functools import partial
from time import time
from typing import Callable, Dict

import accelerate
import click
import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.scores import get_score
from misc.utils import EasyDict


@click.command(context_settings={'show_default': True})
# General options
@click.option(
    "--cache", default="./data/cache", type=str,
    help="Cache directory for the `load_dataset`."
)
@click.option(
    "--version", default="v2", type=click.Choice(["v1", "v2"]),
    help="The version of the Pick-a-Pic dataset to load."
)
@click.option(
    "--split", default="train", type=str,
    help="The split of the dataset to load."
)
@click.option(
    "--score", default="pickscore", type=click.Choice([
        "pickscore", "hpsv2", "aesthetic", "clip", "imagereward"]),
    help="The score to filter the dataset."
)
@click.option(
    "--top", default=500, type=int,
    help="The number of top images (sorted by score) to retrieve."
)
@click.option(
    "--output", default="./data/pickapicv2_pickscore_500", type=str,
    help="The path where the labels should be saved."
)
@click.option(
    "--num_proc", default=8, type=int,
    help=(
        "The number of processes to use when loading and filtering the "
        "dataset."
    )
)
@click.option(
    "--batch_size", default=64, type=int,
    help="The batch size to use when filtering the dataset."
)
def main(**kwargs):
    args = EasyDict(kwargs)
    score_cache_path = os.path.join(
        args.cache, f"pickapic{args.version}_{args.split}_{args.score}.jsonl")

    calc_scores(args, score_cache_path)
    save_expert(args, score_cache_path)


def calc_scores(args: EasyDict, score_cache_path: str):
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # Create the cache directory
    if accelerator.is_main_process:
        cache_dir = os.path.dirname(score_cache_path)
        os.makedirs(cache_dir, exist_ok=True)
    else:
        disable_progress_bar()

    # Load the already done captions
    uid2caption2score = defaultdict(dict)
    if os.path.exists(score_cache_path):
        accelerator.print(f"Loading cached scores from {score_cache_path} ... ", end="")
        start_time = time()
        with open(score_cache_path) as f:
            for line in f:
                uid2caption2score_single = json.loads(line)
                for uid in uid2caption2score_single:
                    uid2caption2score[uid].update(uid2caption2score_single[uid])
        elapsed_time = time() - start_time
        accelerator.print(f"{elapsed_time:.2f}s")

    # Count the number of done uid-caption pairs
    done_uid_caption = 0
    for uid in uid2caption2score:
        done_uid_caption += len(uid2caption2score[uid])

    # Load the dataset without images to speed up the process
    dataset = load_dataset(
        f"yuvalkirstain/pickapic_{args.version}_no_images", split=args.split)
    dataset = dataset.select_columns(['caption', 'image_0_uid', 'image_1_uid'])

    # Count the unique uid-caption pairs
    unique_uid_caption = set()
    for caption, uid0, uid1 in zip(
        dataset['caption'], dataset['image_0_uid'], dataset['image_1_uid']
    ):
        unique_uid_caption.add((uid0, caption))
        unique_uid_caption.add((uid1, caption))
    num_total = len(unique_uid_caption)

    # Add index to dataset for subset selection
    dataset = dataset.map(
        lambda examples, indices: {"index": indices},
        with_indices=True,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Adding Indices",
    )

    # Filter the dataset index to read as few rows as possible
    def filter_fn(examples):
        keep = []
        for caption, uid0, uid1 in zip(
            examples['caption'], examples['image_0_uid'], examples['image_1_uid']
        ):
            keep_example = False
            if uid0 not in uid2caption2score or caption not in uid2caption2score[uid0]:
                keep_example = True
            if uid1 not in uid2caption2score or caption not in uid2caption2score[uid1]:
                keep_example = True
            keep.append(keep_example)
        return keep
    dataset = dataset.filter(
        filter_fn,
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=False,
        desc="Filtering Indices",
    )
    # Collect the indices that contains the remaining captions
    indices = sorted(dataset['index'])

    # Exit if there are no more captions to process
    if len(indices) == 0:
        if not accelerator.is_main_process:
            accelerator.end_training()
            exit(0)
        else:
            accelerator.end_training()
            return

    # Load the score model and image processor
    compute_score, processor = get_score(args.score, device)

    # Load the full dataset
    pickapic = load_dataset(
        f"yuvalkirstain/pickapic_{args.version}",
        split=args.split,
        num_proc=args.num_proc)
    # Select only the necessary columns
    pickapic = pickapic.select_columns(
        ['caption', 'image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1'])
    # Select the remaining indices
    pickapic = pickapic.select(indices)

    # Transform jpg bytes to PIL.Image and then to torch.Tensor
    def transform(examples: Dict[str, list], processor: Callable) -> Dict[str, list]:
        examples['image_0'] = []
        examples['image_1'] = []
        for jpg_0, jpg_1 in zip(examples["jpg_0"], examples["jpg_1"]):
            with io.BytesIO(jpg_0) as jpg_0_bytes:
                image_0 = Image.open(jpg_0_bytes).copy()
            with io.BytesIO(jpg_1) as jpg_1_bytes:
                image_1 = Image.open(jpg_1_bytes).copy()
            examples["image_0"].append(processor(image_0))
            examples["image_1"].append(processor(image_1))
        return examples
    pickapic = pickapic.with_transform(partial(transform, processor=processor))

    def collat_fn(items):
        # List of dictionaries to dictionary of lists
        batch = {key: [item[key] for item in items] for key in items[0]}
        # Stack the torch.Tensor objects
        for key in batch:
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key])
        return batch
    loader = DataLoader(
        pickapic,
        batch_size=args.batch_size,
        num_workers=args.num_proc,
        collate_fn=collat_fn,
    )

    # Let the accelerator handle the last batch
    loader = accelerator.prepare(loader)

    # Run inference on the Pickapic dataset
    rel_err_pass = 0
    rel_err_fail = 0
    rel_err_pass_rate = 0.0
    progress_bar = tqdm(
        loader,
        desc=args.score,
        disable=not accelerator.is_main_process,
    )
    for batch in progress_bar:
        captions = batch['caption']
        scores_0 = compute_score(batch['image_0'], captions)
        scores_1 = compute_score(batch['image_1'], captions)

        # Gather the results
        captions = accelerator.gather_for_metrics(captions, use_gather_object=True)
        captions = captions + captions
        scores_0 = accelerator.gather_for_metrics(scores_0)
        scores_1 = accelerator.gather_for_metrics(scores_1)
        scores = torch.cat([scores_0, scores_1])
        uids0 = accelerator.gather_for_metrics(batch['image_0_uid'], use_gather_object=True)
        uids1 = accelerator.gather_for_metrics(batch['image_1_uid'], use_gather_object=True)
        uids = uids0 + uids1

        # Update uid2caption2score
        if accelerator.is_main_process:
            for caption, score, uid in zip(captions, scores, uids):
                if uid not in uid2caption2score or caption not in uid2caption2score[uid]:
                    uid2caption2score[uid][caption] = score.item()
                    # Save the score to the cache file
                    line = json.dumps({
                        uid: {
                            caption: score.item()
                        }
                    })
                    with open(score_cache_path, 'a') as f:
                        f.write(line + '\n')
                    done_uid_caption += 1
                else:
                    # Sanity check - the score should be the same
                    err = abs(uid2caption2score[uid][caption] - score.item())
                    rel_err = err / uid2caption2score[uid][caption]
                    if rel_err > 0.01:
                        progress_bar.write(
                            f"Large Relative Error: {rel_err:.4f}. "
                            f"UID: {uid}. Caption: {caption}")
                        rel_err_fail += 1
                    else:
                        rel_err_pass += 1
                    if rel_err_fail + rel_err_pass > 0:
                        rel_err_pass_rate = rel_err_pass / (rel_err_pass + rel_err_fail)
                progress_bar.set_postfix_str(
                    f"Finished captions: {done_uid_caption}/{num_total} ({done_uid_caption / num_total:.2%}). "
                    f"Relative Error Pass Rate: {rel_err_pass_rate:.2%}.")
        accelerator.wait_for_everyone()

    progress_bar.close()

    if not accelerator.is_main_process:
        accelerator.end_training()
        exit(0)
    else:
        accelerator.end_training()
        torch.cuda.empty_cache()
        return


def save_expert(args, score_cache_path):
    print(f"Loading cached scores from {score_cache_path} ... ", end="")
    start_time = time()
    uid2caption2score = defaultdict(dict)  # Target caption and its uids
    with open(score_cache_path) as f:
        for line in f:
            uid2caption2score_single = json.loads(line)
            for uid in uid2caption2score_single:
                uid2caption2score[uid].update(uid2caption2score_single[uid])
    elapsed_time = time() - start_time
    print(f"{elapsed_time:.2f}s")

    # Sort uid-caption pairs by scores
    all_items = [(score, uid, caption)
                 for uid, caption2score in uid2caption2score.items()
                 for caption, score in caption2score.items()]
    all_items = sorted(all_items, reverse=True)

    # Top K scoring images
    num_images = max(min(args.top, len(all_items)), 1)
    top_images = all_items[:num_images]
    uid2caption2score = defaultdict(dict)
    score_sum = 0
    for score, uid, caption in top_images:
        uid2caption2score[uid][caption] = score
        score_sum += score
    print(f"Average score for top {num_images} images: {score_sum / num_images:.4f}")

    # Load the dataset without images to speed up the sampling process
    dataset = load_dataset(
        f"yuvalkirstain/pickapic_{args.version}_no_images", split=args.split)
    dataset = dataset.select_columns(['caption', 'image_0_uid', 'image_1_uid'])
    dataset = dataset.map(
        lambda examples, indices: {"index": indices},
        with_indices=True,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Adding Indices",
    )

    # Filter the dataset index to read as few rows as possible
    def filter_fn(examples, uid2caption2score):
        keep = []
        for caption, uid0, uid1 in zip(
            examples['caption'], examples['image_0_uid'], examples['image_1_uid']
        ):
            keep_example = False
            if uid0 in uid2caption2score and caption in uid2caption2score[uid0]:
                keep_example = True
                del uid2caption2score[uid0][caption]
                if len(uid2caption2score[uid0]) == 0:
                    del uid2caption2score[uid0]
            if uid1 in uid2caption2score and caption in uid2caption2score[uid1]:
                keep_example = True
                del uid2caption2score[uid1][caption]
                if len(uid2caption2score[uid1]) == 0:
                    del uid2caption2score[uid1]
            keep.append(keep_example)
        return keep
    uid2caption2score_copy = copy.deepcopy(uid2caption2score)
    dataset = dataset.filter(
        partial(filter_fn, uid2caption2score=uid2caption2score_copy),
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=False,
        desc="Filtering Indices",
    )
    # Collect the indices that contains the remaining captions
    indices = sorted(dataset['index'])
    # Sanity check
    assert len(uid2caption2score_copy) == 0, f"Remaining captions: {len(uid2caption2score_copy)}"

    # Load the full dataset
    pickapic = load_dataset(
        f"yuvalkirstain/pickapic_{args.version}",
        split=args.split,
        num_proc=args.num_proc)
    pickapic = pickapic.select(indices)
    pickapic = pickapic.select_columns(
        ['caption', 'image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1'])
    loader = DataLoader(
        pickapic,
        batch_size=args.batch_size,
        num_workers=args.num_proc,
    )

    num_digits = len(str(num_images))

    # Save the images
    with tqdm(loader, desc="Saving Subset Images") as pbar:
        saving_counter = 0
        for batch in pbar:
            for caption, uid0, uid1, jpg0, jpg1 in zip(
                batch['caption'],
                batch['image_0_uid'], batch['image_1_uid'],
                batch['jpg_0'], batch['jpg_1'],
            ):
                for uid, jpg in [(uid0, jpg0), (uid1, jpg1)]:
                    if uid in uid2caption2score and caption in uid2caption2score[uid]:
                        target_dir = os.path.join(args.output, f"{saving_counter:0{num_digits}}")
                        image_path = os.path.join(target_dir, f"{saving_counter:0{num_digits}}.png")
                        caption_path = os.path.join(target_dir, 'caption.txt')
                        os.makedirs(target_dir, exist_ok=True)

                        # Save the caption
                        if not os.path.exists(caption_path):
                            with open(caption_path, 'w') as f:
                                f.write(caption)
                        # Save the image
                        if not os.path.exists(image_path):
                            with io.BytesIO(jpg) as jpg_bytes:
                                Image.open(jpg_bytes).save(image_path)
                        del uid2caption2score[uid][caption]
                        if len(uid2caption2score[uid]) == 0:
                            del uid2caption2score[uid]
                        saving_counter += 1
                pbar.set_postfix_str(
                    f"# saved images: {saving_counter}/{num_images} "
                    f"({saving_counter / num_images:.2%})")
    # Sanity check
    assert len(uid2caption2score) == 0, f"Remaining captions: {len(uid2caption2score)}"


if __name__ == '__main__':
    main()
