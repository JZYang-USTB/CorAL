# Reflection case from Figure 5

This case follows **Formulation 1 in Figure 5f**. The numbers below are transcribed from the manuscript's displayed labels. The replay uses the same feedback check, surfactant criteria and descriptor screening functions as CorAL's offline workflow.

See the [plotted values for all three formulations](../coral_output/figure5-validation.json) for the complete experimental comparison in Figure 5f.

## Inputs and reported outcomes

| Surfactant | LogP | Radius of gyration | Labute surface area | Inhibition efficiency |
| --- | ---: | ---: | ---: | ---: |
| CTAB | 6.17 | 5.98 | 188.29 | 92.5% |
| TTAB | 5.39 | 5.39 | 169.94 | 91.4% |
| DDBAB | 6.18 | 5.12 | 192.26 | 94.4% |

Figure 5d supplies the descriptor values. Figure 5f supplies the displayed efficiencies. The [case input](../examples/figure5_case_inputs.json) records those sources and identifies the reference, initial feedback and follow-up. The archived catalog places all three surfactants in group 3.

## Replay the decision

```sh
python -m coral reflection-case --out run/figure5-case.json
```

1. **Initial screen.** Lower LogP favors TTAB within the two-candidate comparison.
2. **Feedback.** The displayed TTAB result is 1.1 percentage points below the CTAB reference, triggering reassessment.
3. **Revised screen.** Keep LogP as a similarity criterion, then require smaller radius of gyration and greater Labute surface area. DDBAB satisfies all three rules. The replay reads the reported DDBAB efficiency for comparison after selection.

The [generated trace](../coral_output/figure5-case.json) lists the value and pass result for each rule and each candidate. The [MecIns rules](../coral/agents/mecins.py) and [ForGen checks](../coral/agents/forgen.py) are the functions used by both the case and the offline demo. The LogP tolerances of 1.7 for the initial screen and 0.3 for the revised screen are review implementation settings.

## Evidence scope

The case is a retrospective replay on the compounds displayed in the figure. The three plotted efficiencies are transcriptions at the figure's precision. The replay checks whether the supplied descriptor criteria produce the illustrated decisions on that shortlist. Verification of the original execution and decision timing requires contemporaneous model and task records linked to instrument data. The software demo supplies a separate [synthetic workflow trace](../coral_output/reflection.json) for inspecting system behavior end to end.
