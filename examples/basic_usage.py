"""Minimal end-to-end usage example for TreeForge's public API."""

from src.data.loader import load_and_process
from src.evaluation.metrics import accuracy_score
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier

TARGET_COLUMN = "class"


def main() -> None:
    """Load data, train a Decision Tree, and print test accuracy."""
    df = load_and_process(persist=False)
    X = df.drop(columns=[TARGET_COLUMN]).to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tree Depth: {clf.tree_.get_depth()}")
    print(f"Number of Leaves: {clf.tree_.get_n_leaves()}")


if __name__ == "__main__":
    main()