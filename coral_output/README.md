# CorAL results and source data

These files show how the reported measurements, literature evidence and workflow records are used in CorAL. The source files and evidence type are given alongside each result.

| Manuscript stage | Result | Source and scope |
| --- | --- | --- |
| Fig. 2c and 2d literature mining | [Published literature case](literature_pairs.json) | Evidence from [one published study](../examples/literature_fixture.json) mapped to the archived groups. The study-wide group frequencies are in [cluster priors](../coral/data/cluster_priors.json). |
| Fig. 3e and Fig. 4g optimization and redesign | [Experimental optimization history](optimization-history.json) | Round-level row counts, mean and best observed inhibition efficiency, and winning compositions calculated from [Supplementary Table 3](../coral/data/tsc_bo.csv), [Table 4](../coral/data/pst_mechanism.csv) and [Table 5](../coral/data/pst_bo.csv). Each best result retains the original data row. Row counts refer to each table. |
| Bayesian optimization tool | [Initial candidate ranking](bo-round-0.json) | Recomputed expected-improvement scores and proposed compositions using the initial observations in [Supplementary Table 3](../coral/data/tsc_bo.csv). |
| Fig. 4b–c and Fig. 5b–c principal component analysis | [PCA figure results](pca.json) | Archived Prism plot data for 583 points, the group 1 and group 3 overlays, and the ten contribution scores displayed per panel. Coincident coordinates retain their original point counts. |
| Fig. 4e SHAP attribution | [SHAP figure results](shap.json) | All 50 saved rows of component percentages and SHAP values from the author's plotting workbook, including values outside the final panel's visible range. |
| Fig. 5f experimental comparison | [Three-formulation validation](figure5-validation.json) | Nine plotted efficiency values transcribed at the reported precision. The changes are percentage-point differences from each matched CTAB value. |
| Fig. 5d–f descriptor reassessment | [Formulation 1 reflection replay](figure5-case.json) | Retrospective replay of the expanded descriptor rules, using the [figure values](../examples/figure5_case_inputs.json). |
| Agent and laboratory integration | [Reflection trace](reflection.json) and [equipment dispatch trace](dispatch.json) | Offline integration examples with simulated feedback and equipment acknowledgments. |

The assigned groups behind Fig. 2 and the reflection replay are in the [descriptor catalog](../coral/data/molecular_descriptors.csv). Commands for generating the analysis and replay outputs are in the [scientific tool guide](../docs/reproduction.md) and [reflection case guide](../docs/reflection-case.md).

The PCA and SHAP figure files preserve archived plot values. The command-line tools also support a fresh PCA on the bundled descriptor catalog and a fresh SHAP fit on the archived Supplementary Table 3 measurements.
