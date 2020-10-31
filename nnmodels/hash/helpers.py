from pathlib import Path
from typing import Tuple


separator = '+'


def encode_image_to_uuid(base_model: str, vector_size: int, path: Path) -> str:
    return separator.join([base_model, str(vector_size), path.parent.parent.stem, path.stem])


def decode_image_from_uuid(uuid: str) -> Tuple[str, int, str, str]:
    parts = uuid.split(separator)
    base_model = parts[0]
    descriptor_length = int(parts[1])
    algorithm = parts[2]
    filename = parts[3]
    return base_model, descriptor_length, algorithm, filename
