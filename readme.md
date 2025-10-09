<h1 align="center">
  Ranking-based Preference Optimization </br> for Diffusion Models from Implicit User Feedback
</h1>

This is the official implementation of the paper, *Ranking-based Preference Optimization for Diffusion Models from Implicit User Feedback*.

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

**Note**: The results in the paper were obtained using `Python 3.9.20` and `torch==2.3.1` with `cuda-12.1`.

## Datasets

<details>
<summary>Pick-a-Pic v2</summary>

The script `tools/pickapic.py` can automatically download and preprocess the Pick-a-Pic v2 dataset. It will select top-500 images for training.

- Use PickScore to select top-500 images:
    ```bash
    accelerate launch --multi_gpu --num_processes 8 \
        -m tools.pickapic \
            --score pickscore \
            --output ./data/pickapicv2_pickscore_500
    ```
- Use HPSv2 to select top-500 images:
    ```bash
    accelerate launch --multi_gpu --num_processes 8 \
        -m tools.pickapic \
            --score hpsv2 \
            --output ./data/pickapicv2_hpsv2_500
    ```

For testing, we use the official Pick-a-Pic v2 test set. Run the following script to download and organize the test set:

```bash
python -m tools.pickapic_test \
    --output ./data/pickapicv2_test
```

</details>

<details>
<summary>HPDv2</summary>

The script `tools.hpdv2_benchmark.py` can automatically download and organize the HPDv2 benchmark dataset for testing.

```bash
python -m tools.hpdv2_benchmark \
    --output ./data/hpdv2_benchmark
```

</details>

## Models

### Train From Scratch

```bash
accelerate launch --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4 train.py \
    --train_dataset ./data/pickapicv2_hpsv2_500 \
    --logdir ./logs/sd15_diffusion-dro
```

For more training options, please refer to `python train.py --help`.

### Inference

Inference with the [pre-trained model](https://huggingface.co/ylwu/diffusion-dro-sd1.5) from huggingface hub:

- Pick-a-Pic v2 test:
    ```bash
    accelerate launch --gpu_ids 0,1,2,3 --multi_gpu --num_processes 4 inference.py \
        --unet ylwu/diffusion-dro-sd1.5 \
        --unet_subfolder unet \
        --test_dataset ./data/pickapicv2_test \
        --output ./output/pickapicv2_test
    ```

- HPDv2 Benchmark
    ```bash
    accelerate launch --gpu_ids 0,1,2,3 --multi_gpu --num_processes 4 inference.py \
        --unet ylwu/diffusion-dro-sd1.5 \
        --unet_subfolder unet \
        --test_dataset ./data/hpdv2_benchmark \
        --output ./output/hpdv2_benchmark
    ```

It also supports inference with a local checkpoint by providing the path to `--unet`, e.g., `--unet ./logs/sd15_diffusion-dro/ckpt-25600`.

## Evaluation

Calculate PickScore, HPSv2, Aesthetic Score, CLIP Score, and ImageReward for the generated images

- Pick-a-Pic v2 test:

    ```bash
    accelerate launch --gpu_ids 0,1,2,3 --multi_gpu --num_processes 4 score.py \
        --pickscore --hpsv2 --aestheticv1 --clip --imagereward \
        --dir ./output/pickapicv2_test
    ```

- HPDv2 Benchmark:

    ```bash
    accelerate launch --gpu_ids 0,1,2,3 --multi_gpu --num_processes 4 score.py \
        --pickscore --hpsv2 --aestheticv1 --clip --imagereward \
        --dir ./output/hpdv2_benchmark
    ```

## LICENSE
This model is a fine-tuned version of Stable Diffusion, released under the CreativeML Open RAIL-M License.
