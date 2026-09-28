# ML Foundations to Deep Learning

Four notebooks tracing a single arc: implementing a classifier from raw math, then progressively handing more of the work to better tools — classical ML, then a shallow neural net, then a convolutional one — while testing each model's actual understanding rather than just its accuracy.

## Overview

Each notebook picks one algorithm, implements or applies it, and then tries to break it: pushing on hyperparameters, mislabeling data on purpose, planting shortcuts, or shuffling pixels to see whether the model actually learned structure or just memorized noise. Three of the four converge on the same dataset (MNIST), so later notebooks can be read as direct upgrades over earlier ones — same problem, better model.

This is coursework from a graduate machine learning class, cleaned up for public presentation: institution-facing material (grading rubrics, "for grader" cells, boilerplate instructions) has been removed, and each notebook has been retitled and reframed around what it demonstrates rather than which assignment it was.

## Notebooks

| # | Notebook | What it does | Key result |
|---|---|---|---|
| 1 | [`01-perceptron-from-scratch`](01-perceptron-from-scratch/perceptron_from_scratch.ipynb) | Implements and fixes a perceptron from scratch; studies how learning rate, data ordering, and random seed affect convergence; visualizes decision boundaries evolving over training | Binary search on learning rate to hit a target iteration count fails — the relationship is non-monotonic and also depends on random seed |
| 2 | [`02-pca-knn-mnist`](02-pca-knn-mnist/pca_knn_mnist.ipynb) | Compresses MNIST from 784 pixels to a handful of principal components, then classifies digits with k-NN on the reduced representation, tuned via grid search | Confirms accuracy is far above the 10%-random baseline while training on a fraction of the original feature count |
| 3 | [`03-mlp-mnist`](03-mlp-mnist/mlp_mnist.ipynb) | Trains an MLP on MNIST; analyzes second-best predictions to see what the model confuses; tests scikit-learn's warm-start (incremental) training; plants a label-shortcut in the input to see if the model exploits it | Warm-start training on a 50K-point split then a 10K-point split raised test accuracy from ~90.5% to ~90.9%; when a label-derived shortcut was planted in the first pixel, the MLP did **not** learn to exploit it (97.13% vs. 97.02% accuracy, i.e. no shortcut-driven jump to 100%) |
| 4 | [`04-cnn-mnist-pytorch`](04-cnn-mnist-pytorch/cnn_mnist_pytorch.ipynb) | Builds a CNN in PyTorch; visualizes training convergence; trains on randomly-assigned labels; adapts the model for MNIST; tests it on images with a fixed random pixel permutation | Confirms a random-guess baseline of 10% on a 10-class problem, and probes how a CNN's spatial assumptions hold up (or don't) once pixel locality is destroyed |

## Tech stack

- **Python**, **NumPy**, **pandas**, **matplotlib**
- **scikit-learn** — `LogisticRegression`, `KNeighborsClassifier`, `MLPClassifier`, `PCA`, `GridSearchCV`
- **PyTorch** / **torchvision** — custom `nn.Module` CNN, `DataLoader`, SGD training loop
- **Datasets**: Iris (via UCI ML repository), MNIST (via `sklearn.datasets.fetch_openml`)

## Getting started

Each notebook is self-contained and can be opened directly on GitHub (which renders `.ipynb` files natively, including the embedded plots) or run locally.

To run locally:

```bash
git clone https://github.com/VS-2K01/ml-foundations-to-deep-learning.git
cd ml-foundations-to-deep-learning
pip install numpy pandas matplotlib scikit-learn scipy torch torchvision jupyter
jupyter notebook
```

Then open any of the four `.ipynb` files. Notebooks 2–4 download MNIST via `fetch_openml` on first run, which requires an internet connection and can take a minute.

## Project structure

```
ml-foundations-to-deep-learning/
├── 01-perceptron-from-scratch/
│   └── perceptron_from_scratch.ipynb
├── 02-pca-knn-mnist/
│   └── pca_knn_mnist.ipynb
├── 03-mlp-mnist/
│   └── mlp_mnist.ipynb
├── 04-cnn-mnist-pytorch/
│   └── cnn_mnist_pytorch.ipynb
├── LICENSE
└── README.md
```

## Notes on presentation

These notebooks originated as individual assignments in a graduate machine learning course. Before publishing, I removed grading-related scaffolding (rubric cells, "for grader use" markers, and generic submission instructions) and retitled each notebook around the technique it demonstrates. The analysis, code, and written conclusions inside each notebook are my own, unedited work.

## My role

I wrote the implementations, ran the experiments, and wrote the analysis in every notebook above — this was individual (not team) coursework.

## License and contact

Licensed under the [MIT License](LICENSE).

**Contact:** [github.com/VS-2K01](https://github.com/VS-2K01)
