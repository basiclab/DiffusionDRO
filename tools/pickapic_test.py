import io
import os

import click
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.utils import EasyDict


@click.command(context_settings={'show_default': True})
# General options
@click.option(
    "--version", default="v2", type=click.Choice(["v1", "v2"]),
    help="The version of the Pick-a-Pic dataset to load."
)
@click.option(
    "--split", default="test_unique", type=str,
    help="The split of the dataset to load."
)
@click.option(
    "--output", default="./data/pickapicv2_test", type=str,
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
    disable_progress_bar()

    # Load the dataset without images to speed up the sampling process
    dataset = load_dataset(
        f"yuvalkirstain/pickapic_{args.version}", split=args.split, num_proc=args.num_proc)
    dataset = dataset.select_columns(['caption', 'best_image_uid', 'image_0_uid', 'image_1_uid', 'jpg_0', 'jpg_1'])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_proc,
    )

    num_digits = len(str(len(dataset)))

    # Save the images
    with tqdm(loader, desc="Saving Test Images") as pbar:
        saving_counter = 0
        for batch in pbar:
            for caption, best_uid, uid0, uid1, jpg0, jpg1 in zip(
                batch['caption'],
                batch['best_image_uid'],
                batch['image_0_uid'], batch['image_1_uid'],
                batch['jpg_0'], batch['jpg_1'],
            ):
                if best_uid == uid0:
                    best_jpg = jpg0
                else:
                    best_jpg = jpg1
                target_dir = os.path.join(args.output, f"{saving_counter:0{num_digits}}")
                image_path = os.path.join(target_dir, f"{saving_counter:0{num_digits}}.png")
                caption_path = os.path.join(target_dir, 'caption.txt')
                os.makedirs(target_dir, exist_ok=True)
                # Save the caption
                with open(caption_path, 'w') as f:
                    f.write(caption)
                # Save the image
                with io.BytesIO(best_jpg) as jpg_bytes:
                    Image.open(jpg_bytes).save(image_path)

                saving_counter += 1
                pbar.set_postfix_str(
                    f"# saved images: {saving_counter}/{len(dataset)} "
                    f"({saving_counter / len(dataset):.2%})")


if __name__ == '__main__':
    main()
