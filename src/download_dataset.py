import gzip
import os
import random
import shutil
import struct
import urllib.request

from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
# URLs for MNIST binary files hosted on Internet Archive
IMAGE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
LABEL_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"

DATA_DIR = "mnist_dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")

IMAGE_FILE = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
LABEL_FILE = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")


def download_file(url: str, target_path: str):
    """Downloads a file with a custom User-Agent to bypass archive.org requests blocking."""
    print(f"Downloading {url}...")
    req = urllib.request.Request(
        url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with (
        urllib.request.urlopen(req) as response,
        open(target_path, "wb") as out_file,
    ):
        shutil.copyfileobj(response, out_file)
    print(f"Saved to {target_path}")


def load_mnist_images(filepath: str):
    """Parses IDX3 binary format for MNIST images."""
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file.")
        image_data = f.read()
        return num_images, rows, cols, image_data


def load_mnist_labels(filepath: str):
    """Parses IDX1 binary format for MNIST labels."""
    with gzip.open(filepath, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file.")
        label_data = f.read()
        return num_items, label_data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download image and label binary archives if not already locally cached
    if not os.path.exists(IMAGE_FILE):
        download_file(IMAGE_URL, IMAGE_FILE)
    if not os.path.exists(LABEL_FILE):
        download_file(LABEL_URL, LABEL_FILE)

    # 2. Extract dataset binary structures
    print("Extracting MNIST dataset...")
    num_images, rows, cols, image_bytes = load_mnist_images(IMAGE_FILE)
    num_labels, label_bytes = load_mnist_labels(LABEL_FILE)

    assert num_images == num_labels, "Mismatch between image and label counts!"

    image_size = rows * cols

    # 3. Prepare training directory structure (0-9)
    for digit in range(10):
        os.makedirs(os.path.join(TRAIN_DIR, str(digit)), exist_ok=True)

    # 4. Save PNG images in their corresponding digit folder in "training"
    print("Saving images to initial training directory structure...")
    for i in range(num_images):
        digit = str(label_bytes[i])
        start_offset = i * image_size
        end_offset = start_offset + image_size
        img_data = image_bytes[start_offset:end_offset]

        # Convert 28x28 byte buffer into PIL image
        img = Image.frombytes("L", (cols, rows), img_data)

        save_path = os.path.join(TRAIN_DIR, digit, f"image_{i:05d}.png")
        img.save(save_path)

    print(f"Saved {num_images} images into '{TRAIN_DIR}'.")

    # 5. Move exactly 30% of images per digit folder into "testing"
    print("Moving exactly 30% of data to testing folder...")
    random.seed(42)  # Seed for reproducible random splitting

    for digit in range(10):
        digit_str = str(digit)
        src_digit_dir = os.path.join(TRAIN_DIR, digit_str)
        dst_digit_dir = os.path.join(TEST_DIR, digit_str)

        os.makedirs(dst_digit_dir, exist_ok=True)

        files = [
            f
            for f in os.listdir(src_digit_dir)
            if os.path.isfile(os.path.join(src_digit_dir, f))
        ]
        random.shuffle(files)

        # Exactly 30% calculation
        num_test_files = int(len(files) * 0.30)
        test_files = files[:num_test_files]

        for fname in test_files:
            src_file = os.path.join(src_digit_dir, fname)
            dst_file = os.path.join(dst_digit_dir, fname)
            _ = shutil.move(src_file, dst_file)

        print(
            f"Digit '{digit}': Moved {len(test_files)} images to testing/ ({len(files) - len(test_files)} remain in training/)"
        )

    print("\nDataset ready! Directory tree structure:")
    print(f"├── {DATA_DIR}/")
    print("│   ├── training/ (0, 1, 2, ..., 9)")
    print("│   └── testing/  (0, 1, 2, ..., 9)")


if __name__ == "__main__":
    main()
