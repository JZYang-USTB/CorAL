from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .molecular import scaled_descriptor_matrix


ORGANIC_GROUP_COUNTS = range(5, 16)
INORGANIC_GROUP = 11


def cluster_catalog(rows, plot_dir=None):
    inorganic = {row["molecule_id"]: INORGANIC_GROUP for row in rows
                 if str(row.get("group")) == str(INORGANIC_GROUP)}
    organic = [row for row in rows if row["molecule_id"] not in inorganic]
    candidate_counts = [count for count in ORGANIC_GROUP_COUNTS if count < len(organic)]
    if not candidate_counts:
        raise ValueError("Silhouette analysis needs at least six organic entries.")
    features = scaled_descriptor_matrix(organic)
    tree = linkage(features, method="average", metric="euclidean")
    partitions = {}
    scores = {}
    for count in candidate_counts:
        labels = cut_tree(tree, n_clusters=[count]).ravel() + 1
        if len(set(labels)) != count:
            raise RuntimeError(f"The dendrogram cut at {count} groups returned a different number of groups.")
        partitions[count] = labels
        scores[count] = float(silhouette_score(features, labels, metric="euclidean"))

    selected = max(scores, key=scores.get)
    labels = partitions[selected]
    organic_ids = [row["molecule_id"] for row in organic]
    assignments = {key: int(label) + int(label >= INORGANIC_GROUP)
                   for key, label in zip(organic_ids, labels, strict=True)}
    archived = [row.get("group") for row in organic]
    if plot_dir is not None:
        _save_plots(plot_dir, tree, organic_ids, scores, selected)

    return {
        "organic_count": len(organic), "inorganic_count": len(inorganic),
        "linkage": "average", "distance_metric": "euclidean",
        "candidate_organic_group_counts": candidate_counts,
        "selection_rule": "Highest silhouette score, with ties resolved in favor of the smaller count.",
        "selected_organic_group_count": selected, "silhouette_scores": scores,
        "adjusted_rand_against_archived_organic_labels": (
            float(adjusted_rand_score(archived, labels)) if all(archived) else None
        ),
        "assignments": assignments | inorganic,
    }


def _save_plots(directory, tree, labels, scores, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(list(scores), list(scores.values()), marker="o")
    axes.axvline(selected, linestyle="--", color="gray")
    axes.set(xlabel="Number of organic clusters", ylabel="Silhouette score", xticks=list(scores))
    figure.savefig(directory / "silhouette.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(9, 6))
    dendrogram(tree, labels=labels, orientation="right", leaf_font_size=8,
               truncate_mode="lastp", p=30, ax=axes)
    axes.set(xlabel="Euclidean linkage distance", title="Average-linkage dendrogram")
    figure.savefig(directory / "dendrogram.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
