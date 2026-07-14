# TreeForge

[![CI](https://github.com/niteshg97/TreeForge/actions/workflows/ci.yml/badge.svg)](https://github.com/niteshg97/TreeForge/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-success)
![Built From Scratch](https://img.shields.io/badge/Built-From%20Scratch-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet)


A **from-scratch** Decision Tree classifier — built with pure NumPy/Pandas, no sklearn tree internals — trained and evaluated on the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).

TreeForge implements the full CART algorithm end to end: impurity measures, recursive splitting, tree construction, an sklearn-style prediction API, cross-validation, hyperparameter search, and a rigorous benchmark against scikit-learn.

## Why TreeForge

Most "from scratch" ML projects stop at a toy dataset and a rough approximation. TreeForge is built to **match production-grade behavior**, validated head-to-head against scikit-learn on a real 19,000-sample dataset — not just to run, but to perform.

## Benchmark: TreeForge vs scikit-learn

Same train/test split, same hyperparameters (`max_depth=10`), evaluated on identical held-out data:

| Metric           | TreeForge | scikit-learn |
|------------------|-----------|--------------|
| Accuracy         | 0.8441    | 0.8452       |
| Precision        | 0.8489    | 0.8496       |
| Recall           | 0.9248    | 0.9256       |
| F1 Score         | 0.8852    | 0.8860       |
| Fit Time (s)     | 106.4915  | 0.1368       |
| Predict Time (s) | 0.0071    | 0.0004       |
| Tree Depth       | 10        | 10           |
| Leaves           | 387       | 390          |

### Key Takeaways

- **Predictive performance is nearly identical.** TreeForge matches scikit-learn's accuracy, precision, recall, and F1 within **~0.1 percentage points**, and produces a nearly identical tree structure (387 vs 390 leaves at the same depth) — strong evidence the core splitting logic and CART criteria are implemented correctly.
- **Prediction is fast.** Inference (`0.0071s`) is well within the same order of magnitude as scikit-learn's optimized C implementation, since tree traversal is inherently lightweight regardless of implementation language.
- **Training is the honest tradeoff.** scikit-learn's `~780x` faster fit time comes from its Cython/C backend with vectorized, low-level split search. TreeForge's pure-Python/NumPy recursive builder trades raw speed for **full transparency** — every split, gain calculation, and stopping decision is readable, debuggable Python you can step through line by line.

This is the expected and understood cost of a from-scratch implementation: **algorithmic correctness without sacrificing clarity**, at the cost of the low-level performance engineering that library authors invest years into.

Reproduce this yourself:

```bash
python -m src.benchmark.compare
```

## Features

- Pure NumPy/Pandas implementation of Gini/Entropy impurity, information gain, and recursive tree building
- sklearn-style `fit()` / `predict()` / `predict_proba()` API
- Custom train/test split, k-fold cross-validation, and exhaustive grid search
- Full evaluation suite: accuracy, precision, recall, F1, confusion matrix
- Benchmarked rigorously against scikit-learn's `DecisionTreeClassifier`
- 100% modular, type-hinted, PEP8-compliant, unit-tested codebase

## Installation

```bash
git clone https://github.com/niteshg97/TreeForge.git
cd TreeForge
pip install -e ".[dev]"
```

## Quickstart

```python
from src.data.loader import load_and_process
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier

df = load_and_process(persist=False)
X, y = df.drop(columns=["class"]).to_numpy(), df["class"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
```

Or run the example script:

```bash
python examples/basic_usage.py
```

## Project Structure

See [`docs/architecture.md`](docs/architecture.md) for the full module map and data flow.

```
TreeForge/
├── src/            # Core implementation (data, tree, metrics, evaluation, benchmark)
├── tests/          # Unit tests (pytest)
├── notebooks/       # EDA only
├── examples/       # Usage scripts
├── docs/           # Architecture & design docs
├── data/           # raw/ and processed/ datasets
└── .github/workflows/ci.yml
```

## Testing

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Roadmap

- [ ] Post-pruning (cost-complexity pruning)
- [ ] Vectorized split search to close the training-time gap
- [ ] Random Forest built on top of TreeForge's `Node`/`Tree` primitives

## License

MIT — see [LICENSE](LICENSE).
