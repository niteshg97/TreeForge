## Classifier API
- `src/tree/classifier.py`: `DecisionTreeClassifier` — sklearn-style `fit`/`predict`/`predict_proba`
- Handles arbitrary label types via internal encoding; raises `NotFittedError` if used before `fit()`