import io
import numpy as np
from PIL import Image
import openvino as ov


def matplotlib_to_ov_tensor(fig, dpi: int = 130) -> ov.Tensor:
    """
    Convertit une figure matplotlib en tensor OpenVINO.
    """

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight"
    )

    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")

    arr = np.array(image, dtype=np.uint8)

    arr = np.expand_dims(arr, axis=0)

    return ov.Tensor(arr)