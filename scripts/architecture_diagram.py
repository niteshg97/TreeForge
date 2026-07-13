from graphviz import Digraph

dot = Digraph(
    "TreeForge",
    filename="assets/treeforge_architecture",
    format="png",
)

dot.attr(rankdir="TB", fontsize="18")

dot.attr("node",
         shape="box",
         style="rounded,filled",
         fillcolor="#E8F4FF",
         color="#4A90E2")

dot.node("data", "Load Dataset\nload_and_process()")
dot.node("split", "train_test_split()")

dot.node("fit", "DecisionTreeClassifier.fit()")

dot.node("builder", "TreeBuilder")

dot.node("splitter", "find_best_split()")

dot.node("gain", "Information Gain")

dot.node("impurity", "Gini / Entropy")

dot.node("tree", "Trained Decision Tree")

dot.node("predict", "predict()\npredict_proba()")

dot.node("metrics",
         "Evaluation\nAccuracy\nPrecision\nRecall\nF1\nConfusion Matrix")

dot.node("benchmark", "Benchmark\nvs sklearn")

dot.edges([
    ("data", "split"),
    ("split", "fit"),
    ("fit", "builder"),
    ("builder", "splitter"),
    ("splitter", "gain"),
    ("gain", "impurity"),
    ("impurity", "tree"),
    ("tree", "predict"),
    ("predict", "metrics"),
    ("metrics", "benchmark"),
])

dot.render(cleanup=True)

print("Saved to assets/treeforge_architecture.png")