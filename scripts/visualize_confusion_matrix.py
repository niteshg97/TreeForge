from src.data.loader import load_and_process
from src.model_selection.split import train_test_split
from src.tree.classifier import DecisionTreeClassifier
from src.visualization.confusion_matrix import (
    save_confusion_matrix_plot,
)


def main():

    df = load_and_process(persist=False)

    X = df.drop(columns=["class"]).to_numpy()
    y = df["class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    clf = DecisionTreeClassifier(max_depth=10)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    save_confusion_matrix_plot(
        y_test,
        y_pred,
        class_names=[str(c) for c in clf.classes_],
    )

    print(" Saved confusion matrix")
    print("assets/images/confusion_matrix.png")


if __name__ == "__main__":
    main()