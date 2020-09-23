from enum import auto
from typing import List

import utils


class Mode(utils.CustomEnum):
    Debug = auto(),
    Release = auto()


class DetectionAlgorithm(utils.CustomEnum):
    MorphologyIteration1 = "MI1",
    MorphologyIteration2 = "MI2",

    LineSegmentation = "LS",
    MSER = "MSER",
    MajorVoting = "MV"

    @staticmethod
    def load_from_config(keyword: str) -> List['DetectionAlgorithm']:
        if keyword.lower() == 'all':
            return DetectionAlgorithm.to_list()


class ApproximationMethod(utils.CustomEnum):
    Contour = auto(),
    Brect = auto(),
    Rrect = auto(),
    Hull = auto(),
    Approximation = auto(),


class FileDatabase(utils.CustomEnum):
    Enrollment = auto(),

    PharmaPack_R_I_S1 = auto(),
    PharmaPack_R_I_S2 = auto(),
    PharmaPack_R_I_S3 = auto(),

    @staticmethod
    def get_list_of_recognition_databases() -> List['FileDatabase']:
        return [FileDatabase.PharmaPack_R_I_S1, FileDatabase.PharmaPack_R_I_S2, FileDatabase.PharmaPack_R_I_S3]


class FilePhone(utils.CustomEnum):
    Phone1 = auto(),
    Phone2 = auto(),
    Phone3 = auto()


class AnnotationLabel(utils.CustomEnum):

    def get_color(self):
        return self.value[0]

    Text = (255, 0, 255),
    Number = (0, 255, 255),

    Watermark = (0, 0, 255),
    Image = (255, 0, 0),
    Barcode = (255, 255, 0),

    Unknown = (255, 255, 255)

    @staticmethod
    def get_list_of_text_labels() -> List['AnnotationLabel']:
        return [AnnotationLabel.Text, AnnotationLabel.Number]

    @staticmethod
    def get_list_of_graphic_labels() -> List['AnnotationLabel']:
        return [AnnotationLabel.Watermark, AnnotationLabel.Image, AnnotationLabel.Barcode]


class EvalMetric(utils.CustomEnum):
    IntersectionOverUnion = "IOU",

    TruePositiveRate = "TPR",  # tp / (tp + fn) or 1 - FalseNegativeRate
    FalsePositiveRate = "FPR", #
    TrueNegativeRate = "TNR", # tn / (tn + fp) or 1 - FalsePositiveRate
    FalseNegativeRate = "FNR",

    Prevalence = "PRV",
    Accuracy = "ACC",     # (tp + tn) / (tp + fp + tn + fn)
    FalseDiscoveryRate = "FDR"
    Precision = "PRC",    # tp / (tp + fp) or 1 - FalseDiscoveryRate
    FalseOmissionRate = "FOR",
    NegativePredictiveValue = "NPV"

    PositiveLikelihoodRatio = "PLR",
    NegativeLikelihoodRatio = "NLR",
    # DiagnosticOddsRatio = "DOR",
    F1Score = "F1S"

    @classmethod
    def get_essential(cls):
        return [
            cls.IntersectionOverUnion,
            cls.TruePositiveRate, cls.FalsePositiveRate, cls.TrueNegativeRate, cls.FalseNegativeRate,
            cls.Accuracy, cls.Precision, cls.F1Score
        ]


class AlignmentMethod(utils.CustomEnum):
    NoAlignment = auto(),
    Reference = auto(),
    ToCorners = auto()
