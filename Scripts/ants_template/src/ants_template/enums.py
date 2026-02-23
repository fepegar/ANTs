"""Enums for template construction options."""

from enum import Enum


class SimilarityMetric(str, Enum):
    """Similarity metric for nonlinear registration."""

    NMI = "NMI"
    LNCC = "LNCC"
    SSD = "SSD"
    MIND = "MIND"
    MINDSSC = "MINDSSC"


class TransformationType(str, Enum):
    """Transformation model for registration."""

    SyN = "SyN"
    Affine = "Affine"
    Rigid = "Rigid"


class ImageStatistic(str, Enum):
    """Image statistic used to summarize images."""

    MEAN = "mean"
    NORMALIZED_MEAN = "normalized-mean"

    @classmethod
    def default(cls) -> "ImageStatistic":
        return cls.NORMALIZED_MEAN
