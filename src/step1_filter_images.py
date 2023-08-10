import shutil
from pathlib import Path

import numpy as np
from PIL import Image

for path in Path('data/source_data/Furniture - Images').iterdir():
    img_ = Image.open(path).convert('RGB')
    img = np.array(img_)

    # filtering too close views
    a = img[0].mean(axis=0)
    b = img[:, 0].mean(axis=0)
    c = img[-1].mean(axis=0)
    d = img[:, -1].mean(axis=0)
    norm = np.linalg.norm([a - b, a - c, a - d])
    corners_similar = np.isclose(norm, 0, atol=120, rtol=1)

    # filtering too close views
    luminosity = np.array(img_.convert('L'))
    white_percent = (luminosity > 180).sum() / (luminosity.shape[0] * luminosity.shape[1])

    # filtering textures
    std = np.std(luminosity)

    print(path, white_percent, norm, std)

    if corners_similar and 0.1 < white_percent < 0.99 and std > 24:
        shutil.copyfile(path, Path('data/good') / path.name)
    else:
        shutil.copyfile(path, Path('data/bad') / path.name)
