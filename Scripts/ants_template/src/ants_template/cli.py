"""CLI entry point for template construction."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .enums import ImageStatistic, SimilarityMetric, TransformationType
from .template import build_template

app = typer.Typer(
    name="ants-template",
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Build a population template from a set of images using iterative"
        " registration.\n\n"
        "This is a Python equivalent of antsMultivariateTemplateConstruction2.sh,"
        " using NiftyReg (via niftyregw) for registration.\n\n"
        "The algorithm iteratively registers each input image to the current"
        " template, averages the resampled images, and demeans the"
        " transformations to keep the template unbiased."
    ),
    rich_markup_mode="rich",
)


@app.command()
def main(
    images: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "Input images. For single-modality, list all images."
                " For multi-modal, provide images interleaved by subject"
                " (e.g., subA_t1 subA_t2 subB_t1 subB_t2) and set"
                " --num-modalities accordingly."
            ),
            exists=True,
            dir_okay=False,
        ),
    ],
    output_prefix: Annotated[
        str,
        typer.Option(
            "--output-prefix",
            "-o",
            help=(
                "Prefix prepended to all output files."
                " Can include a directory path."
            ),
        ),
    ] = "antsBTP",
    initial_template: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--initial-template",
            "-z",
            help=(
                "Initial template image(s). Provide one per modality."
                " If not given, the template is initialized by averaging"
                " all inputs."
            ),
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    num_iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-i",
            help="Number of template construction iterations.",
            min=1,
        ),
    ] = 4,
    rigid_init: Annotated[
        bool,
        typer.Option(
            "--rigid-init",
            "-r",
            help=(
                "Perform rigid alignment of inputs to the initial template"
                " before iterative registration. Recommended when not"
                " providing an initial template."
            ),
        ),
    ] = False,
    transformation: Annotated[
        TransformationType,
        typer.Option(
            "--transform",
            "-t",
            help=(
                "Transformation model."
                " SyN = non-rigid (default),"
                " Affine = affine only,"
                " Rigid = rigid only."
            ),
            case_sensitive=False,
        ),
    ] = TransformationType.SyN,
    metric: Annotated[
        SimilarityMetric,
        typer.Option(
            "--metric",
            "-m",
            help=(
                "Similarity metric for non-rigid registration."
                " NMI = Normalized Mutual Information (default),"
                " LNCC = Local Normalized Cross-Correlation,"
                " SSD = Sum of Squared Differences,"
                " MIND/MINDSSC = Modality-independent descriptors."
            ),
            case_sensitive=False,
        ),
    ] = SimilarityMetric.NMI,
    lncc_sigma: Annotated[
        float,
        typer.Option(
            "--lncc-sigma",
            help="Gaussian kernel standard deviation for LNCC metric.",
            min=0.0,
        ),
    ] = 5.0,
    statistic: Annotated[
        ImageStatistic,
        typer.Option(
            "--statistic",
            "-a",
            help=(
                "Image statistic used to summarize images when building"
                " the template."
            ),
            case_sensitive=False,
        ),
    ] = ImageStatistic.NORMALIZED_MEAN,
    num_modalities: Annotated[
        int,
        typer.Option(
            "--num-modalities",
            "-k",
            help=(
                "Number of modalities per subject."
                " E.g., use 2 for T1 + T2 multi-modal template."
            ),
            min=1,
        ),
    ] = 1,
    affine_update_full: Annotated[
        bool,
        typer.Option(
            "--full-affine/--no-rigid-in-affine",
            "-y/-Y",
            help=(
                "Use full affine for template update."
                " If disabled, the rigid component is excluded,"
                " which can help if the template drifts."
            ),
        ),
    ] = True,
) -> None:
    """Build a population template from a set of images.

    This command implements the iterative template construction algorithm:

    1. Initialize the template (from average or user-provided image).

    2. Optionally align all inputs rigidly to the template.

    3. For each iteration, register all inputs, average, and demean.

    \b
    Example (single-modality):
        ants-template -i 3 -t SyN -m LNCC -r sub*_T1w.nii.gz

    \b
    Example (multi-modal):
        ants-template -i 3 -k 2 -t SyN -m LNCC \\
            subA_T1.nii.gz subA_T2.nii.gz subB_T1.nii.gz subB_T2.nii.gz

    \b
    Example (affine template first, then refine with SyN):
        ants-template -i 2 -t Affine -o MY_Affine_ -r sub*.nii.gz
        ants-template -i 5 -t SyN -z MY_Affine_template0.nii.gz \\
            -o MY_SyN_ sub*.nii.gz
    """
    build_template(
        images=images,
        output_prefix=output_prefix,
        initial_templates=initial_template,
        num_iterations=num_iterations,
        rigid_init=rigid_init,
        transformation=transformation,
        metric=metric,
        lncc_sigma=lncc_sigma,
        statistic=statistic,
        num_modalities=num_modalities,
        affine_update_full=affine_update_full,
    )
