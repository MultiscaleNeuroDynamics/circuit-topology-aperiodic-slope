# Circuit topology and the E/I–aperiodic-exponent mapping

This repository reproduces the simulations and analyses for the manuscript
*Circuit Topology Shapes the Mapping between Excitatory–Inhibitory Input
Composition and Aperiodic Spectral Slope*. It contains one authoritative
pipeline for each analysis, fixed MICrONS-derived inputs, morphology files,
NEURON mechanism sources, and restart-safe controllers for expensive runs.

Generated trial directories, LFP arrays, and final statistical tables are not
stored in Git. They are regenerated below `work/ALLEN/` from the included
inputs. Raw juvenile-pig EDF/BIDS data are also not redistributed and must be
downloaded from OpenNeuro dataset `ds003380`.

## Repository layout

```text
data/microns/                  Fixed MICrONS-derived inputs and 80 SWC files
data/external/                 Small non-recording figure assets only
mechanisms/                    NMODL source files
src/circuit_topology/          Shared morphology and neuron-model code
pipelines/simulation/          Primary 100-trial dynamics-to-LFP pipeline
pipelines/analysis/            Primary spectral and regression analyses
pipelines/operating_point/     m=0.90 and m=0.95 robustness pipelines
pipelines/topology_control/    Degree-preserving topology control
pipelines/biophysical_robustness/  Joint biophysical-parameter controls
pipelines/empirical/           OpenNeuro pig-data validation
scripts/                       Workspace preparation and release checks
work/                          Generated locally; ignored by Git
```

## Installation

Linux is recommended for the NEURON/LFP simulations.
Clone the repository and place any external workspace under an ASCII-only
absolute path; NEURON/HOC 8.x may fail when morphology or mechanism paths
contain non-ASCII characters.

```bash
conda env create -f environment.yml
conda activate circuit-topology-aperiodic
python scripts/bootstrap_workspace.py
cd mechanisms
nrnivmodl
cd ..
```

`bootstrap_workspace.py` copies the version-controlled fixed inputs into the
generated compatibility workspace without replacing existing results. To use
another generated-data location, pass `--workspace-root /absolute/ascii/path`
and use the same path as `--project-root` in later commands.

The MICrONS coordinate transform is supplied by the PyPI distribution
`standard-transform` (imported as `standard_transform`). It is pinned in
`environment.yml`; no separate local package is required.

## 1. Primary 100-trial MICrONS–ER analysis

Prepare dynamics and all NEURON inputs:

```bash
python pipelines/simulation/run_main_pipeline.py --max-workers 10
```

Run/restart the expensive REAL and ER LFP simulations:

```bash
python pipelines/simulation/run_main_pipeline.py --max-workers 10 --run-neuron
```

Compute realised synaptic E/I ratios, 1–100 Hz aperiodic exponents, Table 1
cluster-robust regressions with 95% confidence intervals, and the stratified
40-trial validation:

```bash
python pipelines/analysis/run_main_analysis.py --max-workers 10
```

Primary outputs are written to:

```text
work/ALLEN/results/ei_ratio_stats/
work/ALLEN/results/fooof_group_compare/
work/ALLEN/results/mixed_ei_topology_test/
work/ALLEN/results/stratified_subset_validation/
```

## 2. Robustness across operating points

The first command generates 40 evenly spaced E/I trials at each operating
point, calibrates the REAL and ER gains, and prepares NEURON inputs. The second
form launches/restarts LFP simulations.

```bash
python pipelines/operating_point/run_operating_point_pipeline.py --max-workers 10
python pipelines/operating_point/run_operating_point_pipeline.py --max-workers 10 --run-neuron
python pipelines/operating_point/run_operating_point_analysis.py --max-workers 10
```

Use `--only-m 0.90` or `--only-m 0.95` to process one condition. Results are
written to `work/ALLEN/results/m_0.90/` and `m_0.95/`.

## 3. Degree-preserving topology control

The DP controller selects the fixed, E/I-stratified set of 40 manuscript trial
identifiers, generates degree-preserving E→E and I→E rewires matched on measured
effective branching ratio, reuses the corresponding morphology and synaptic
embedding assets, and resumes LFP simulations without replacing completed files.

```bash
python pipelines/topology_control/run_dp_pipeline.py --max-workers 10
python pipelines/topology_control/run_dp_pipeline.py --max-workers 10 --run-neuron
python pipelines/topology_control/run_three_topology_analysis.py
```

The Table 3 analysis and `RESULTS_SUMMARY.txt` are generated under
`work/ALLEN/results/m_0.99/real_er_dp40/`.

## 4. Joint biophysical-parameter robustness

This endpoint analysis uses trials 0001, 0004, 0097, and 0100 and five
deterministic joint parameter profiles. Profiles are robustness scenarios, not
independent replicates.

```bash
python pipelines/biophysical_robustness/run_biophysical_pipeline.py --max-workers 10
```

The restart-safe controller generates the profiles, runs the 160 LFP tasks,
and creates `profile_robustness_summary.csv` and `RESULTS_SUMMARY.txt` under
`work/ALLEN/results/m_0.99/biophys_endpoint_sensitivity/`.

## 5. Juvenile-pig validation

Raw EDF files, `participants.tsv`, and `channels.tsv` are not included. Download
OpenNeuro `ds003380` (DOI `10.18112/openneuro.ds003380.v1.0.0`) and run:

```bash
python pipelines/empirical/run_empirical_pipeline.py \
  --data-root /absolute/path/to/ds003380
```

The pipeline checks the BIDS tree, extracts FOOOF and adaptive-Ising features,
performs animal-level matched-channel statistics, and creates the manuscript
figures under `work/ALLEN/results/empirical_propofol/`. The raw dataset is read
in place and is never copied into the repository workspace.

## 6. Assemble manuscript Tables 1--3

After the primary, operating-point, DP, and biophysical pipelines have
completed, assemble the final submission-format CSV tables without refitting
any models:

```bash
python pipelines/analysis/build_manuscript_tables.py
```

The three files are written to `work/ALLEN/results/manuscript_tables/`.

## Reproducibility and restart behaviour

- Trial-level and LFP-level outputs are checked before expensive work starts.
- Completed LFP files are skipped individually after an interrupted run.
- Primary or operating-point dynamics generators stop on a partial output tree
  instead of silently replacing completed trials.
- All stochastic analysis choices use fixed seeds recorded in the scripts.
- The REAL, ER, and DP regression pipelines use the same realised synaptic E/I
  definition: excitatory event count divided by inhibitory event count.
- Statistical CSV files and summaries are generated outputs, not hidden inputs.

Run the lightweight release audit before archiving or uploading:

```bash
python scripts/validate_release.py
```

## Data provenance

See `docs/DATA_SOURCES.md` for MICrONS inputs, morphology counts, the
`standard-transform` dependency, and the public OpenNeuro source. Users should
also cite the MICrONS and ds003380 data sources described in the manuscript.
`docs/REPRODUCIBILITY_MATRIX.md` maps each manuscript result to its controller,
generated inputs, and final output directory.

## License

The source code in this repository is released under the MIT License; see
`LICENSE`. Included MICrONS-derived inputs, morphology files, external assets,
and any locally downloaded OpenNeuro recordings retain their original licenses
and terms of use and are not relicensed by the software license.

Repository-level citation metadata are provided in `CITATION.cff`. The formal
article DOI and journal citation will be added upon publication.
