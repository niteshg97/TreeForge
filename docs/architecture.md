# TreeForge Architecture

## Module Map

```
src/
├── config.py              # Paths & constants
├── data/loader.py         # Raw → processed data pipeline
├── eda/                   # Summary stats & visualizations
├── metrics/impurity.py    # Gini, entropy, information gain
├── tree/
│   ├── node.py            # Node data structure
│   ├── tree.py            # Tree container (depth/nodes/leaves)
│   ├── splitter.py        # Best-split search
│   ├── builder.py         # Recursive tree construction
│   └── classifier.py      # Public fit/predict API
├── model_selection/
│   ├── split.py               # train_test_split
│   ├── cross_validation.py    # k-fold CV
│   └── grid_search.py         # GridSearchCV
├── evaluation/metrics.py  # Accuracy, precision, recall, F1, confusion matrix
├── benchmark/compare.py   # TreeForge vs sklearn comparison
└── pipeline/train.py      # End-to-end training script
```

## Data Flow

```
raw .data file
     │
     ▼
data/loader.py  →  processed DataFrame (encoded labels)
     │
     ▼
model_selection/split.py  →  train/test arrays
     │
     ▼
tree/classifier.py.fit()
     │
     ▼
tree/builder.py._build_node()  (recursive)
     │        │
     ▼        ▼
tree/splitter.py      metrics/impurity.py
(best split search)   (Gini/Entropy/IG)
     │
     ▼
tree/node.py + tree/tree.py  →  fitted Tree
     │
     ▼
tree/classifier.py.predict()
     │
     ▼
evaluation/metrics.py  →  accuracy, precision, recall, F1
```

## Design Principles

- Every mathematical operation (impurity, gain, split search) is a pure, independently testable function.
- `Node`/`Tree` are dumb data containers; `TreeBuilder` owns all training logic.
- `DecisionTreeClassifier` is the only public-facing class — sklearn-style API, internal mechanics hidden.
- sklearn is used exclusively in `src/benchmark/` for validation, never in the core algorithm.