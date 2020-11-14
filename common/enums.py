from enum import auto
from typing import List, Any

import cv2 as cv
from utils import CustomEnum


class Mode(CustomEnum):
    Debug = auto(),
    Release = auto()


class SegmentationAlgorithm(CustomEnum):
    MorphologyIteration1 = "MI1",
    MorphologyIteration2 = "MI2",

    LineSegmentation = "LS",
    MSER = "MSER",
    MajorVoting = "MV"


class ApproximationMethod(CustomEnum):
    Contour = "Contour",
    Brect = "Brect",
    Rrect = "Rrect",
    Hull = "Hull",
    HullApproximated = "HullApproximated",


class FileDatabase(CustomEnum):
    Enrollment = auto(),

    PharmaPack_R_I_S1 = auto(),
    PharmaPack_R_I_S2 = auto(),
    PharmaPack_R_I_S3 = auto(),

    @staticmethod
    def get_recognition_databases_list() -> List['FileDatabase']:
        return [FileDatabase.PharmaPack_R_I_S1, FileDatabase.PharmaPack_R_I_S2, FileDatabase.PharmaPack_R_I_S3]


class AnnotationLabel(CustomEnum):

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


class EvalMetric(CustomEnum):
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
    def get_essential_metrics(cls) -> List['EvalMetric']:
        return [
            cls.IntersectionOverUnion,
            cls.TruePositiveRate, cls.FalsePositiveRate, cls.TrueNegativeRate, cls.FalseNegativeRate,
            cls.Accuracy, cls.Precision, cls.F1Score
        ]


class AlignmentMethod(CustomEnum):
    NoAlignment = auto(),
    Reference = auto(),
    ToCorners = auto()


class Model(CustomEnum):
    Triplet = auto(),
    Resnet50 = auto(),
    Siamese = auto(),
    SimCLR = auto(),
    Hash = auto()

class Descriptor(CustomEnum):

    AKAZE = "AKAZE",
    SIFT = "SIFT",
    # SURF = "SURF",
    ORB = "ORB",
    BRISK = "BRISK",
    KAZE = "KAZE",
