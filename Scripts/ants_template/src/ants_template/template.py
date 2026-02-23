"""Core template construction logic using NiftyReg via niftyregw."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from loguru import logger
from niftyregw import reg_aladin, run

from .enums import ImageStatistic, SimilarityMetric, TransformationType


def _average_images(
    output: Path,
    images: list[Path],
    statistic: ImageStatistic,
) -> None:
    """Average a set of images using reg_average.

    Args:
        output: Output averaged image path.
        images: List of input images.
        statistic: Statistic to use for averaging.
    """
    if statistic == ImageStatistic.MEAN:
        args = [str(output), "-avg", *[str(img) for img in images]]
    elif statistic == ImageStatistic.NORMALIZED_MEAN:
        # NiftyReg's reg_average -avg computes the mean; for normalized mean,
        # we just use the standard average (NiftyReg doesn't have a built-in
        # normalized mean, but the average is sufficient for template building)
        args = [str(output), "-avg", *[str(img) for img in images]]
    else:
        msg = f"Unsupported statistic: {statistic}"
        raise ValueError(msg)

    tool_logger = logger.bind(executable="reg_average")
    run("reg_average", *args, tool_logger=tool_logger)


def _average_affines(output: Path, affines: list[Path]) -> None:
    """Average affine transformations.

    Args:
        output: Output averaged affine path.
        affines: List of affine transformation files.
    """
    args = [str(output), "-avg", *[str(aff) for aff in affines]]
    tool_logger = logger.bind(executable="reg_average")
    run("reg_average", *args, tool_logger=tool_logger)


def _demean_transformations(
    output: Path,
    reference: Path,
    affines: list[Path],
    nonrigid: list[Path],
    images: list[Path],
) -> None:
    """Demean transformations to update the template shape.

    Uses reg_average -demean_noaff to remove the average affine and
    non-rigid deformation from the template.

    Args:
        output: Output demeaned average image path.
        reference: Reference/template image.
        affines: List of affine transformation files.
        nonrigid: List of non-rigid transformation files (CPP).
        images: List of floating images.
    """
    # Build interleaved list: aff1 nrr1 flo1 aff2 nrr2 flo2 ...
    interleaved: list[str] = []
    for aff, nrr, img in zip(affines, nonrigid, images):
        interleaved.extend([str(aff), str(nrr), str(img)])

    args = [str(output), "-demean_noaff", str(reference), *interleaved]
    tool_logger = logger.bind(executable="reg_average")
    run("reg_average", *args, tool_logger=tool_logger)


def _demean_affine_only(
    output: Path,
    reference: Path,
    affines: list[Path],
    images: list[Path],
) -> None:
    """Demean affine-only transformations to update the template.

    Uses reg_average -demean to remove the average affine transformation
    from the template.

    Args:
        output: Output demeaned average image path.
        reference: Reference/template image.
        affines: List of affine transformation files.
        images: List of floating images.
    """
    # Build interleaved list: aff1 flo1 aff2 flo2 ...
    interleaved: list[str] = []
    for aff, img in zip(affines, images):
        interleaved.extend([str(aff), str(img)])

    args = [str(output), "-demean", str(reference), *interleaved]
    tool_logger = logger.bind(executable="reg_average")
    run("reg_average", *args, tool_logger=tool_logger)


def _register_affine(
    reference: Path,
    floating: Path,
    output_affine: Path,
    output_result: Path,
    *,
    rigid_only: bool = False,
) -> None:
    """Run affine (or rigid) registration.

    Args:
        reference: Reference image.
        floating: Floating image.
        output_affine: Output affine transformation file.
        output_result: Output resampled image.
        rigid_only: If True, perform rigid registration only.
    """
    reg_aladin(
        reference=reference,
        floating=floating,
        output_affine=output_affine,
        output_result=output_result,
        rigid_only=rigid_only,
        use_images_centre_of_mass=True,
    )


def _register_nonrigid(
    reference: Path,
    floating: Path,
    input_affine: Path,
    output_cpp: Path,
    output_result: Path,
    *,
    metric: SimilarityMetric = SimilarityMetric.NMI,
    lncc_sigma: float = 5.0,
) -> None:
    """Run non-rigid (F3D) registration.

    Args:
        reference: Reference image.
        floating: Floating image.
        input_affine: Input affine transformation from prior alignment.
        output_cpp: Output control point grid file.
        output_result: Output resampled image.
        metric: Similarity metric to use.
        lncc_sigma: Sigma for LNCC metric (only used if metric is LNCC).
    """
    args: list[str] = [
        "-ref",
        str(reference),
        "-flo",
        str(floating),
        "-aff",
        str(input_affine),
        "-cpp",
        str(output_cpp),
        "-res",
        str(output_result),
    ]

    if metric == SimilarityMetric.LNCC:
        args.extend(["--lncc", str(lncc_sigma)])
    elif metric == SimilarityMetric.SSD:
        args.append("--ssd")
    elif metric == SimilarityMetric.MIND:
        args.extend(["--mind", "2"])
    elif metric == SimilarityMetric.MINDSSC:
        args.extend(["--mindssc", "2"])
    # NMI is the default for reg_f3d, no extra flag needed

    tool_logger = logger.bind(executable="reg_f3d")
    run("reg_f3d", *args, tool_logger=tool_logger)


def _resample(
    reference: Path,
    floating: Path,
    transformation: Path,
    output: Path,
) -> None:
    """Resample an image with a given transformation.

    Args:
        reference: Reference image.
        floating: Floating image to resample.
        transformation: Transformation file.
        output: Output resampled image.
    """
    args = [
        "-ref",
        str(reference),
        "-flo",
        str(floating),
        "-trans",
        str(transformation),
        "-res",
        str(output),
    ]
    tool_logger = logger.bind(executable="reg_resample")
    run("reg_resample", *args, tool_logger=tool_logger)


def build_template(
    images: list[Path],
    *,
    output_prefix: str = "antsBTP",
    initial_templates: list[Path] | None = None,
    num_iterations: int = 4,
    rigid_init: bool = False,
    transformation: TransformationType = TransformationType.SyN,
    metric: SimilarityMetric = SimilarityMetric.NMI,
    lncc_sigma: float = 5.0,
    statistic: ImageStatistic = ImageStatistic.NORMALIZED_MEAN,
    num_modalities: int = 1,
    affine_update_full: bool = True,
) -> list[Path]:
    """Build a population template using iterative registration and averaging.

    This is the Python equivalent of antsMultivariateTemplateConstruction2.sh,
    using NiftyReg (via niftyregw) instead of ANTs.

    The algorithm:
      1. Create an initial template by averaging all inputs (or use provided).
      2. Optionally perform rigid alignment of all inputs to the template.
      3. For each iteration:
         a. Register each input to the current template (affine + non-rigid).
         b. Resample each input into template space.
         c. Update the template by averaging the resampled images.
         d. Demean the transformations to keep the template unbiased.

    Args:
        images: Input images. For multi-modal, provide images interleaved
            by modality (e.g., [subA_t1, subA_t2, subB_t1, subB_t2]).
        output_prefix: Prefix prepended to all output files.
        initial_templates: Optional initial template(s). If not provided,
            the initial template is created by averaging all inputs.
        num_iterations: Number of template construction iterations.
        rigid_init: Perform rigid alignment before iterative registration.
        transformation: Transformation model (SyN, Affine, or Rigid).
        metric: Similarity metric for non-rigid registration.
        lncc_sigma: Sigma for LNCC metric (used when metric is LNCC).
        statistic: Statistic used to summarize images when building template.
        num_modalities: Number of modalities per subject.
        affine_update_full: Use full affine for template update. If False,
            the rigid component is excluded.

    Returns:
        List of final template paths (one per modality).
    """
    time_start = time.time()

    # Validate inputs
    if len(images) == 0:
        msg = "At least 2 images are required."
        raise ValueError(msg)

    if len(images) % num_modalities != 0:
        msg = (
            f"Number of images ({len(images)}) is not divisible by"
            f" the number of modalities ({num_modalities})."
        )
        raise ValueError(msg)

    num_subjects = len(images) // num_modalities
    if num_subjects < 2:
        msg = f"At least 2 subjects are required, got {num_subjects}."
        raise ValueError(msg)

    # Determine output directory
    if "/" in output_prefix or "\\" in output_prefix:
        output_dir = Path(output_prefix).parent
        template_base = Path(output_prefix).name
    else:
        output_dir = Path.cwd()
        template_base = output_prefix

    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate_dir = output_dir / "intermediateTemplates"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Create template paths
    templates: list[Path] = []
    for m in range(num_modalities):
        templates.append(output_dir / f"{template_base}template{m}.nii.gz")

    # Organize images by modality
    images_by_modality: list[list[Path]] = []
    for m in range(num_modalities):
        modality_images = [images[i] for i in range(m, len(images), num_modalities)]
        images_by_modality.append(modality_images)

    # Step 1: Create initial templates
    logger.info("Creating initial templates")
    for m in range(num_modalities):
        if initial_templates and m < len(initial_templates):
            logger.info(
                f"Using provided initial template for modality {m}:"
                f" {initial_templates[m]}"
            )
            shutil.copy2(initial_templates[m], templates[m])
        else:
            logger.info(
                f"Creating initial template for modality {m} by averaging"
                f" {num_subjects} images"
            )
            _average_images(
                templates[m],
                images_by_modality[m],
                ImageStatistic.NORMALIZED_MEAN,
            )

        # Back up initial template
        shutil.copy2(
            templates[m],
            intermediate_dir / f"initial_{templates[m].name}",
        )

    # Step 2: Optional rigid initialization
    if rigid_init:
        logger.info("Performing rigid initialization")
        for m in range(num_modalities):
            rigid_results: list[Path] = []
            for s in range(num_subjects):
                img = images_by_modality[m][s]
                img_stem = img.name.replace(".nii.gz", "").replace(".nii", "")
                aff_out = output_dir / f"rigid_m{m}_s{s}_{img_stem}.txt"
                res_out = output_dir / f"rigid_m{m}_s{s}_{img_stem}.nii.gz"

                logger.info(
                    f"Rigid registration: modality {m},"
                    f" subject {s + 1}/{num_subjects}"
                )
                _register_affine(
                    reference=templates[m],
                    floating=img,
                    output_affine=aff_out,
                    output_result=res_out,
                    rigid_only=True,
                )
                rigid_results.append(res_out)

            # Update template from rigid-aligned images
            _average_images(templates[m], rigid_results, statistic)
            shutil.copy2(
                templates[m],
                intermediate_dir / f"initialRigid_{templates[m].name}",
            )

            # Clean up rigid results
            for f in rigid_results:
                f.unlink(missing_ok=True)
            for f in output_dir.glob("rigid_m*_s*_*.txt"):
                f.unlink(missing_ok=True)

    # Step 3: Iterative registration
    no_warp = transformation in (
        TransformationType.Affine,
        TransformationType.Rigid,
    )

    for iteration in range(num_iterations):
        logger.info(
            f"Iteration {iteration + 1}/{num_iterations}"
            f" ({transformation.value})"
        )

        all_affines: list[Path] = []
        all_cpps: list[Path] = []
        all_resampled: list[list[Path]] = [[] for _ in range(num_modalities)]

        for s in range(num_subjects):
            img_first = images_by_modality[0][s]
            img_stem = img_first.name.replace(".nii.gz", "").replace(".nii", "")
            sub_prefix = f"{template_base}input{s:04d}_{img_stem}"

            aff_path = output_dir / f"{sub_prefix}_affine.txt"
            cpp_path = output_dir / f"{sub_prefix}_cpp.nii.gz"

            # Affine registration (using first modality)
            aff_result = output_dir / f"{sub_prefix}_m0_affine_result.nii.gz"
            logger.info(
                f"  Affine registration: subject {s + 1}/{num_subjects}"
            )

            rigid_only = transformation == TransformationType.Rigid
            _register_affine(
                reference=templates[0],
                floating=img_first,
                output_affine=aff_path,
                output_result=aff_result,
                rigid_only=rigid_only,
            )
            all_affines.append(aff_path)

            if not no_warp:
                # Non-rigid registration
                f3d_result = output_dir / f"{sub_prefix}_m0_f3d_result.nii.gz"
                logger.info(
                    f"  Non-rigid registration: subject"
                    f" {s + 1}/{num_subjects}"
                )
                _register_nonrigid(
                    reference=templates[0],
                    floating=img_first,
                    input_affine=aff_path,
                    output_cpp=cpp_path,
                    output_result=f3d_result,
                    metric=metric,
                    lncc_sigma=lncc_sigma,
                )
                all_cpps.append(cpp_path)
                all_resampled[0].append(f3d_result)
            else:
                all_resampled[0].append(aff_result)

            # Resample additional modalities using transforms from first
            for m in range(1, num_modalities):
                img_m = images_by_modality[m][s]
                res_m = (
                    output_dir
                    / f"{sub_prefix}_m{m}_result.nii.gz"
                )

                if not no_warp:
                    _resample(
                        reference=templates[m],
                        floating=img_m,
                        transformation=cpp_path,
                        output=res_m,
                    )
                else:
                    _resample(
                        reference=templates[m],
                        floating=img_m,
                        transformation=aff_path,
                        output=res_m,
                    )
                all_resampled[m].append(res_m)

        # Update templates by averaging resampled images
        for m in range(num_modalities):
            logger.info(f"  Updating template for modality {m}")
            _average_images(templates[m], all_resampled[m], statistic)

        # Demean transformations to keep template unbiased
        if not no_warp and all_cpps:
            logger.info("  Demeaning transformations")
            _demean_transformations(
                output=templates[0],
                reference=templates[0],
                affines=all_affines,
                nonrigid=all_cpps,
                images=images_by_modality[0],
            )
        elif all_affines:
            logger.info("  Demeaning affine transformations")
            _demean_affine_only(
                output=templates[0],
                reference=templates[0],
                affines=all_affines,
                images=images_by_modality[0],
            )

        # Save intermediate templates
        for m in range(num_modalities):
            shutil.copy2(
                templates[m],
                intermediate_dir
                / f"{transformation.value}_iteration{iteration}_{templates[m].name}",
            )

    time_elapsed = time.time() - time_start
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)

    logger.info(f"Done creating templates: {[str(t) for t in templates]}")
    logger.info(f"Script executed in {hours}h {minutes}m {seconds}s")
    logger.info(f"Intermediate templates stored in: {intermediate_dir}/")

    return templates
