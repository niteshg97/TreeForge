## Benchmark
- `src/benchmark/compare.py`: side-by-side comparison of TreeForge vs sklearn `DecisionTreeClassifier`
- Compares accuracy, precision, recall, F1, timing, and tree structure on identical splits
- Run: `python -m src.benchmark.compare`
- **Note:** sklearn is used here only for benchmarking, not in the core implementation.