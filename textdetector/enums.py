from enum import auto

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