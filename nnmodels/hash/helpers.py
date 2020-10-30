from pathlib import Path


def get_image_uuid(base_model: str, vector_size: int, path: Path) -> str:
    return "+".join([base_model, str(vector_size), path.parent.parent.stem, path.stem])
