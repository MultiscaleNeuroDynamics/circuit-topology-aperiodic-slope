# Data sources and redistribution

## MICrONS inputs

The fixed MICrONS-derived tables, patch matrices, synapse-to-segment mappings,
and morphology files used by the simulations are provided under
`data/microns/`. The release contains 80 morphology files: 20 each for L2/3,
L4, L5, and L6. Generated trial directories and LFP arrays are intentionally
excluded because the repository pipelines recreate them.
`data/microns/CHECKSUMS.sha256` records the SHA-256 digest of every fixed input;
`scripts/validate_release.py` verifies the complete set before release.

The coordinate conversion used by the code is provided by the
`standard-transform` Python package. The MICrONS tutorial documents the
`standard_transform.minnie_ds`/Minnie coordinate workflow and the package is
installed by `environment.yml`.

- MICrONS standard-transform tutorial: https://tutorial.microns-explorer.org/quickstart_notebooks/08-standard-transform.html
- PyPI distribution: https://pypi.org/project/standard-transform/

## Juvenile-pig recordings

Raw EDF recordings and BIDS sidecars are not redistributed. Download OpenNeuro
dataset `ds003380`, version 1.0.0, from:

- Dataset DOI: https://doi.org/10.18112/openneuro.ds003380.v1.0.0
- Public dataset mirror: https://github.com/OpenNeuroDatasets/ds003380
- Associated data paper: https://doi.org/10.1038/s41597-020-00781-y

The empirical controller requires a local dataset directory containing
`participants.tsv` and the `sub-*/eeg/` trees. Pass that directory with
`--data-root`; the repository never copies the raw recordings into its output.

## Generated outputs

All generated trials, LFP arrays, FOOOF tables, regression coefficients,
confidence intervals, robustness summaries, and figures are written below
`work/ALLEN/`. These files are excluded by `.gitignore` and are recreated by
the commands in the top-level README.
