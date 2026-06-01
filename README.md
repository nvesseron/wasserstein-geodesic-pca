# GPCAGEN

Code for the paper **On the Wasserstein Geodesic Principal Component Analysis
of Probability Measures**.

- OpenReview: <https://openreview.net/forum?id=OJupg4mDjS>
- arXiv: <https://arxiv.org/abs/2506.04480>

## Environment

This repository intentionally does not commit a `pyproject.toml`, lockfile, or
virtual environment. Create and check the Python/JAX environment separately
before running the experiments.

## Main Jean Zay Command

After activating your environment, the main public example is:

```bash
python run_experiment.py -m hydra/launcher=jz_submitit +experiments=point_cloud_lamps
```

Set `IDRPROJ` as required by the Jean Zay submitit launcher.

## Paper Experiment Configs

| Experiment | Hydra config | Dataset |
| --- | --- | --- |
| Point-cloud lamps | `+experiments=point_cloud_lamps` | ShapeNet point clouds |
| Point-cloud chairs | `+experiments=point_cloud_chairs` | ShapeNet point clouds |
| Landscape images | `+experiments=landscape_images` | image color distributions |
| Colored MNIST | `+experiments=colored_mnist` | MNIST |

Data and output roots can be configured with:

```bash
export GPCAGEN_DATA_DIR=/path/to/data/root
export GPCAGEN_OUTPUT_DIR=/path/to/output/root
```

If unset, anonymized defaults resolve to `NONE`, so set these variables before
launching an experiment.
See [`docs/datasets.md`](docs/datasets.md) for expected dataset layouts.

## Citation

```bibtex
@inproceedings{vesseron2026wasserstein,
  title = {On the Wasserstein Geodesic Principal Component Analysis of Probability Measures},
  author = {Vesseron, Nina and Cazelles, Elsa and Le Brigant, Alice and Klein, Thierry},
  booktitle = {International Conference on Learning Representations},
  year = {2026}
}
```
