from src.data.loader import load_and_process
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier
from src.visualization.tree_plot import save_tree_plot


def main() -> None:
    df = load_and_process(persist=False)

    X = df.drop(columns=["class"]).to_numpy()
    y = df["class"].to_numpy()

    feature_names = df.drop(columns=["class"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    clf = DecisionTreeClassifier(max_depth=3)

    clf.fit(X_train, y_train)

    save_tree_plot(
        clf.tree_,
        filename="assets/images/decision_tree",
        feature_names=feature_names,
    )

    print("Tree saved successfully!")
    print("assets/images/decision_tree.png")


if __name__ == "__main__":
    main()