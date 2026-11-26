# Scientific tools

Run these commands from the repository root after [installation](../README.md#python-setup).

| Tool | Role in CorAL |
| --- | --- |
| Retrieval and literature mining | Retrieve literature passages and derive molecular group pair frequencies. |
| Clustering and PCA | Define molecular groups and rank descriptors for a target group. The downstream examples use the archived group assignments. |
| Bayesian optimization | Rank new composition ratios from prior measurements using bootstrap forests and expected improvement. |
| SHAP | Explain the surrogate's predictions for the tabulated composition data. |
| EIS | Fit example impedance spectra and derive inhibition efficiency from a matched blank. |

```sh
python -m coral pca --group 3 --out run/pca-group-3.json
python -m coral bo --through-round 0 --workers 4 --out run/bo-round-0.json
python -m coral shap --out run/shap.json --plot run/shap.png
python -m coral fit-eis --data examples/demo/eis_inhibited.csv --blank examples/demo/eis_blank.csv --out run/eis.json
```

## Hierarchical clustering

[`coral/clustering.py`](../coral/clustering.py) adapts the authors' hierarchical clustering script. It preserves column-mean imputation, standardization, Euclidean distance and the silhouette scan over candidate counts from 5 through 15. Average linkage follows the Supplementary Information. The selected count is taken directly from the tested counts.

```sh
python -m coral cluster --catalog coral/data/molecular_descriptors.csv --out run/clustering.json --plot-dir run/clustering
```

The command returns the scores, selected organic cluster count and recomputed assignments. A tie selects the smaller count. Inorganic salts stay in group 11 and are excluded from the silhouette scan. The optional plots are `silhouette.png` and `dendrogram.png`. The study catalog preserves the reported ten organic groups and inorganic group 11. ForGen and the downstream case replays use those archived labels. The clustering command recalculates the partition from the supplied descriptor export.

## Literature mining and group frequencies

The review command starts with a unique `publication_id` and the abstract or conclusion text prepared from each study. It takes chemical aliases and any available PubChem CIDs or InChIKeys from the supplied catalog identity list. Prepare that list by checking names against PubChem and manually resolving ambiguous identities, as described in the Supplementary Information. See the [published laminarin and sodium molybdate case](../examples/literature_fixture.json).

1. Screen the supplied sections for combination terms, including synergistic, mixture, combination and binary inhibitor.
2. Extract binary and ternary formulations. The offline mode identifies candidate formulations in sentences or clauses with catalog aliases. Review these candidates against the recorded excerpts. The optional DeepSeek mode uses the same screened sections and validates its quoted evidence against the input text. Manually reviewed formulations can be supplied in `studies[].formulations` with `section`, `evidence` and `components`.
3. Resolve each inhibitor to a catalog molecule and cluster. The report retains ambiguous or unmatched entries for review and excludes their formulations from the frequency table.
4. Expand each ternary formulation into three binary molecular pairs. Map each pair to its unordered group pair and rank the frequencies across publications. In the review command, an identical set of standardized molecules counts once within a publication. A molecular pair shared by two distinct formulations contributes once per formulation. The Supplementary Information describes group-level counting across publications without specifying this edge case.

```sh
python -m coral mine --input examples/literature_fixture.json --out run/literature-pairs.json
python -m coral mine --input examples/literature_fixture.json --clusters run/clustering.json --out run/literature-reclustered.json
```

Use the `cluster` command above to generate `run/clustering.json` before running the second command. The default uses the archived group labels. The optional second command uses assignments from the same descriptor catalog's clustering run. Inspect `classified_formulations` for the source excerpt, reported names, catalog identities and group assignments. The `molecular_pairs` entries link each binary molecular pair to its group pair. The report collects those counts in `by_publication` and ranks the totals in `pair_frequencies`.

With a configured `CORAL_LLM_API_KEY`, add `--extractor deepseek` to send screened abstracts and conclusions to DeepSeek. The [included case output](../coral_output/literature_pairs.json) uses the archived cluster labels and counts group pair 8–11 once from the supplied publication.

### Figure 2 data

- **Fig. 2b** shows ten organic clusters. Inorganic salts are assigned to group 11. The [study catalog](../coral/data/molecular_descriptors.csv) preserves these assignments.
- **Fig. 2c** starts from PDF files. The review command accepts the prepared abstracts and conclusions and a curated identity table. The publication-level records underlying the figure are unavailable in this package.
- **Fig. 2d** shows group-pair occurrence frequencies. The [archived group prior](../coral/data/cluster_priors.json) exports the author's frequency table for 22 distinct group pairs. The top six have frequencies of 25 for 4–11, 24 for 4–4, 9 for 1–11, 7 for 11–11, 6 for 3–4 and 6 for 1–4. These pairs span groups 1, 3, 4 and 11. ForGen uses the archived prior with retrieved passages when proposing formulations. The PDF retrieval corpus in Fig. 2a serves that separate retrieval step.

## Bayesian optimization

[`coral/optimization.py`](../coral/optimization.py) adapts the authors' ratio-search script. It retains the bootstrap random forest, ten-fold grid search and expected-improvement workflow. The review implementation uses the manuscript's four-component grid at 5% increments, 1,771 feasible compositions, 50 bootstraps and maximization of inhibition efficiency. ExpOpt calls this module for its ratio proposals.

The replay uses the initial observations in [Supplementary Table 3](../coral/data/tsc_bo.csv). It can take several minutes. Compare the acquisition values and ranked candidates with [the included output](../coral_output/bo-round-0.json). Later archived observations are available through `--through-round`.

The retriever is accessible through the local API. Run `python -m coral --help` for all command options. The demo spectra are synthetic. PCA and BO use the included study data.
