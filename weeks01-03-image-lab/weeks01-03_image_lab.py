

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _resolve_image_path(image_path: str) -> str:
    """Resolve the actual image file path from common project layouts."""
    raw_path = Path(image_path)
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        root = Path(__file__).resolve().parent
        candidates = [
            root / raw_path,
            root / "images" / "original.jpg",
            root / "images" / "original.jpg.jpeg",
            root / "Images" / "original.jpg",
            root / "Images" / "original.jpg.jpeg",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(raw_path if raw_path.is_absolute() else (Path(__file__).resolve().parent / raw_path))


def _load_image(image_path: str):
    """Load an image using OpenCV and return it in RGB order."""
    resolved_path = _resolve_image_path(image_path)
    image = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Image not found or could not be opened: {image_path}\n"
            "Place your own photo at images/original.jpg."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _save_montage(images, titles, output_path, cols=3, figsize=(12, 8)):
    """Save a labeled grid of images without opening a GUI."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, image, title in zip(axes, images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    image = _load_image(image_path)
    height, width, channels = image.shape
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "shape": list(image.shape),
        "pixel_count": int(width * height),
        "estimated_bytes": int(width * height * channels),  # 8 bits/channel = 1 byte
        "color_order": "RGB",
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    image = _load_image(image_path)
    height, width = image.shape[:2]

    red = np.zeros_like(image)
    green = np.zeros_like(image)
    blue = np.zeros_like(image)
    red[:, :, 0] = image[:, :, 0]
    green[:, :, 1] = image[:, :, 1]
    blue[:, :, 2] = image[:, :, 2]

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    half_width = max(1, width // 2)
    half_height = max(1, height // 2)
    downsampled = cv2.resize(
        image, (half_width, half_height), interpolation=cv2.INTER_AREA
    )

    output_path = str(Path(output_dir) / "pixel_views.png")
    _save_montage(
        [image, red, green, blue, grayscale, downsampled],
        ["Original RGB", "Red channel", "Green channel", "Blue channel",
         "Grayscale", "Half-size image"],
        output_path,
        cols=3,
        figsize=(13, 8),
    )
    return {
        "original_size": [int(width), int(height)],
        "downsampled_size": [int(half_width), int(half_height)],
        "output_path": output_path,
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in the range 0-255.")

    image = _load_image(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brighter = np.clip(
        grayscale.astype(np.int16) + int(brightness_delta), 0, 255
    ).astype(np.uint8)
    higher_contrast = np.clip(
        grayscale.astype(np.float32) * float(contrast_factor), 0, 255
    ).astype(np.uint8)
    _, thresholded = cv2.threshold(
        grayscale, int(threshold), 255, cv2.THRESH_BINARY
    )

    output_path = str(Path(output_dir) / "adjustments.png")
    _save_montage(
        [grayscale, brighter, higher_contrast, thresholded],
        ["Original grayscale", f"Brighter (+{brightness_delta})",
         f"Higher contrast (x{contrast_factor})",
         f"Threshold (>{threshold} white, otherwise black)"],
        output_path,
        cols=2,
        figsize=(12, 8),
    )
    return {
        "brightness_delta": int(brightness_delta),
        "contrast_factor": float(contrast_factor),
        "threshold": int(threshold),
        "output_path": output_path,
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    image = _load_image(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(grayscale, (kernel_size, kernel_size))

    def sobel_edges(gray):
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.convertScaleAbs(magnitude)

    original_edges = sobel_edges(grayscale)
    blurred_edges = sobel_edges(blurred)

    output_path = str(Path(output_dir) / "blur_and_edges.png")
    _save_montage(
        [grayscale, blurred, original_edges, blurred_edges],
        ["Grayscale", f"Mean blur ({kernel_size}x{kernel_size})",
         "Sobel edges - original", "Sobel edges - blurred"],
        output_path,
        cols=2,
        figsize=(12, 8),
    )
    return {"kernel_size": kernel_size, "output_path": output_path}


def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1-4 and return their results together."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "task1": inspect_image(image_path),
        "task2": create_pixel_views(image_path, output_dir),
        "task3": create_adjustments(image_path, output_dir),
        "task4": create_blur_and_edges(image_path, output_dir),
    }


def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = _resolve_image_path("images/original.jpg")
    results = run_lab(image_path, "outputs")
    print("Image Processing Lab completed.")
    for task, result in results.items():
        print(f"{task}: {result}")
    print(f"Using image: {image_path}")
    print("Created: outputs/pixel_views.png")
    print("Created: outputs/adjustments.png")
    print("Created: outputs/blur_and_edges.png")


if __name__ == "__main__":
    main()
