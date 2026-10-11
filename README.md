# Deep Learning for Perception — Assignment 2, Section B

Robust CIFAR-10 classification with only 500 training images (50 per class), evaluated on a shifted test set (rotation, brightness/contrast, blur, noise, occlusion and one hidden corruption).

**Roll No:** 23F-0600.

## Files

| File | Purpose |
|---|---|
| `DL_ASS02_23F_0600.ipynb` | Completed, fully executed notebook (Parts A–F) |
| `README.md` | This file |

## How to reproduce

1. Open `DL_ASS02_23F_0600.ipynb` in Google Colab or Kaggle.
2. Select a GPU runtime (Colab: *Runtime → Change runtime type → T4 GPU*; Kaggle: *Settings → Accelerator → GPU*). On Kaggle also turn **Internet ON**.
3. Choose **Run all**. No dataset upload is needed: CIFAR-10 is loaded with `keras.datasets.cifar10.load_data()`.

The full run trains 33 models (at most 60 epochs each) and takes roughly 30 minutes on a T4 GPU, after the dataset download.

Libraries: TensorFlow / Keras, NumPy, pandas, matplotlib, seaborn, SciPy, scikit-learn (all preinstalled on Colab and Kaggle).

## Seeds

| Seed | Used for |
|---|---|
| `42, 123, 2026` | the 3 training runs of every configuration (results are mean ± std) |
| `DATA_SEED = 42` | picking the 500 images and the 400 / 100 train / validation split |
| `SHIFT_SEED = 1234` | the corruption pipeline that builds the shifted test set |

`tf.config.experimental.enable_op_determinism()` is enabled. The reported numbers come from a Google Colab GPU run with TensorFlow 2.21; a different GPU or TensorFlow version can change them slightly.

## Rules followed

- No pretrained weights; the model is trained from scratch.
- 178,682 trainable parameters (limit: 1,000,000).
- Two Inception modules (1×1, 3×3 and 5×5 branches with 1×1 reductions, plus a pooling branch).
- 400 / 100 train / validation split; every choice is made on validation images only.
- Augmentation is applied on the fly, to training data only.
- At most 60 epochs; 3 seeds; mean ± std reported.

## Results (accuracy %, mean ± std over 3 seeds)

| Model | Validation | Clean test | Shifted test |
|---|---|---|---|
| Baseline (no augmentation, no regularization) | 47.67 ± 5.03 | 43.01 ± 2.77 | 18.83 ± 0.44 |
| Final model (geometric + photometric + Random Erasing + Mixup + label smoothing) | 47.67 ± 2.08 | 38.06 ± 3.68 | **23.20 ± 2.97** |

- **Change that helped most:** photometric augmentation (+2.95 points of shifted-test accuracy on its own; +4.32 together with geometric augmentation).
- **Limitations:** the shifted-test gain costs about 5 points of clean-test accuracy; Random Erasing, Mixup and test-time augmentation gave no measurable benefit; robustness did not transfer to the hidden channel-swap corruption.
