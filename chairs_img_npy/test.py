import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from scrape import DATA_DIR

import torch
from torchvision import transforms

NUMPY_DIR = '/mnt/fs5/wumike/datasets/chairs2k/numpy'


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                        default=DATA_DIR, help='where to save stuff')
    args = parser.parse_args()

    if not os.path.isdir(NUMPY_DIR):
        os.makedirs(NUMPY_DIR)

    image_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    names, images = [], []
    image_paths = glob(os.path.join(args.data_dir, '*.png'))
    pbar = tqdm(total=len(image_paths))
    for path in image_paths:
        name = os.path.basename(path)
        image = Image.open(path).convert('RGB')
        image = image_transform(image)
        image = image.numpy()

        names.append(name)
        images.append(image)
        pbar.update()
    pbar.close()

    names = np.array(names)
    images = np.stack(images)

    np.save(os.path.join(NUMPY_DIR, 'names.npy'), names)
    np.save(os.path.join(NUMPY_DIR, 'images.npy'), images)
