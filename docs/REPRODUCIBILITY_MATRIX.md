# Reproducibility matrix

| Manuscript result | Pipeline entry point | Principal generated output |
|---|---|---|
| Primary 100-trial MICrONS--ER interaction and Table 1 | `pipelines/simulation/run_main_pipeline.py`; `pipelines/analysis/run_main_analysis.py` | `results/mixed_ei_topology_test/Table1_coefficients_with_95CI.csv` |
| E/I-stratified 40-trial validation | `pipelines/analysis/run_main_analysis.py` | `results/stratified_subset_validation/` |
| Robustness at `m=0.90` and `m=0.95` | `pipelines/operating_point/run_operating_point_pipeline.py`; `run_operating_point_analysis.py` | `results/m_0.90/` and `results/m_0.95/` |
| MICrONS--DP--ER topology comparison and Table 3 | `pipelines/topology_control/run_dp_pipeline.py`; `run_three_topology_analysis.py` | `results/m_0.99/real_er_dp40/` |
| Joint biophysical-profile robustness | `pipelines/biophysical_robustness/run_biophysical_pipeline.py` | `results/m_0.99/biophys_endpoint_sensitivity/` |
| Combined Table 2 and submission-format Tables 1--3 | `pipelines/analysis/build_manuscript_tables.py` | `results/manuscript_tables/` |
| Juvenile-pig propofol analysis | `pipelines/empirical/run_empirical_pipeline.py` | `results/empirical_propofol/` |

Every path in the final column is relative to `work/ALLEN/`. Generated trial
trees, LFP arrays, FOOOF fits, and statistical tables are intentionally absent
from version control. The pipeline entry points recreate them from the fixed
inputs or, for the pig analysis, from a local copy of OpenNeuro `ds003380`.

## Shared E/I definition

All REAL, ER, DP, operating-point, and biophysical regression analyses use the
same realized synaptic event-count ratio:

```text
E/I = total excitatory synaptic spike events / total inhibitory synaptic spike events
```

For REAL trials, presynaptic MICrONS cell classes identify excitatory and
inhibitory synapses. ER and DP synthetic synapse identifiers encode the same
classification. Nominal drive settings are not substituted for the realized
event-count ratio in any interaction regression.
