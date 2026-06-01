# Datasets

This repository does not commit datasets, checkpoints, W&B runs, or generated
figures. Configure dataset locations with:

```bash
export GPCAGEN_DATA_DIR=/path/to/data/root
export GPCAGEN_OUTPUT_DIR=/path/to/output/root
```

If these variables are not set, anonymized defaults resolve to `NONE`, so set
them before launching an experiment.

## MNIST

MNIST is loaded by `src/utils/load_mnist.py`. The loader downloads the raw IDX
files if they are missing from the configured MNIST directory.

Expected layout:

```text
$GPCAGEN_DATA_DIR/mnist/
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
```

## ModelNet / ShapeNet-Style Point Clouds

The lamp and chair experiments expect HDF5 point-cloud files under the same
category-id folders used in the paper runs:

```text
$GPCAGEN_DATA_DIR/shapenet/train/gt/03636649/  # lamp
$GPCAGEN_DATA_DIR/shapenet/train/gt/03001627/  # chair
```

The paper experiments use 100 randomly selected point clouds per class. Keep
large HDF5 files outside Git.

## Landscape Pictures

The landscape image experiment expects image files under:

```text
$GPCAGEN_DATA_DIR/kagglehub/datasets/arnaud58/landscape-pictures/versions/2/
```

The original dataset is available on Kaggle as "Landscape Pictures" by Arnaud
Rougetet.
