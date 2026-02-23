# ants-template

Python equivalent of `antsMultivariateTemplateConstruction2.sh` using
[NiftyReg](https://github.com/KCL-BMEIS/niftyreg) (via
[niftyregw](https://github.com/fepegar/niftyregw)) for registration and
[Typer](https://typer.tiangolo.com/) for the CLI.

## Installation

```bash
uv pip install .
```

Or in development mode:

```bash
uv pip install -e .
```

You also need to install the NiftyReg binaries:

```bash
niftyregw install
```

## Usage

```bash
ants-template --help
```

### Single-modality template

```bash
ants-template --iterations 3 --transform SyN --metric LNCC --rigid-init sub*_T1w.nii.gz
```

### Multi-modal template (e.g., T1 + T2)

```bash
ants-template --iterations 3 --num-modalities 2 --transform SyN --metric LNCC \
    subA_T1.nii.gz subA_T2.nii.gz subB_T1.nii.gz subB_T2.nii.gz
```

### Two-stage: affine first, then refine with SyN

```bash
ants-template --iterations 2 --transform Affine --output-prefix MY_Affine_ --rigid-init sub*.nii.gz
ants-template --iterations 5 --transform SyN --initial-template MY_Affine_template0.nii.gz \
    --output-prefix MY_SyN_ sub*.nii.gz
```

### Using Python

```python
from pathlib import Path
from ants_template.template import build_template

build_template(
    images=[Path("subA.nii.gz"), Path("subB.nii.gz"), Path("subC.nii.gz")],
    num_iterations=3,
    rigid_init=True,
)
```

## Output

- `{output_prefix}template{m}.nii.gz` — final template for each modality `m`
- `intermediateTemplates/` — templates from each iteration

## Mapping from ANTs to NiftyReg

| ANTs | NiftyReg | Description |
| --- | --- | --- |
| `antsRegistration` (rigid/affine) | `reg_aladin` | Global registration |
| `antsRegistration` (SyN) | `reg_f3d` | Non-rigid registration |
| `antsApplyTransforms` | `reg_resample` | Apply transforms |
| `AverageImages` | `reg_average -avg` | Average images |
| `AverageAffineTransform` | `reg_average -avg` | Average affines |
| Shape update (warp averaging) | `reg_average -demean_noaff` | Demean transforms |