from enum import auto
from typing import List

import utils


class DetectionAlgorithm(utils.CustomEnum):
    MorphologyIteration1 = "MI1",
    MorphologyIteration2 = "MI2",

    LineSegmentation = "LS",
    MSER = "MSER",
    MajorVoting = "MV"


class ResultMethod(utils.CustomEnum):
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
    TruePositive = "TP",
    TrueNegative = "TN",
    FalsePositive = "FP",
    FalseNegative = "FN",

    IntersectionOverUnion = "IOU",
    Accuracy = "ACC",
    Sensitivity = "SNS",
    Precision = "PRC",
    Specificity = "SPC",

    RegionsAmount = "RA"


class AlignmentMethod(utils.CustomEnum):
    Reference = auto(),
    ToCorners = auto()
