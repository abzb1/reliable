import os
from math import cos, sin, radians

import numpy as np
from tqdm.auto import tqdm
from PIL import Image, ImageDraw

from datasets import load_dataset, Dataset, concatenate_datasets

def get_affine_rotation_matrix(angle_rad, origin):
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    cx, cy = origin
    return np.array([
        [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
        [sin_a,  cos_a, cy - sin_a * cx - cos_a * cy]
    ])

def rotate_and_cut_image(img, angles, origins, width, height):
    for i, (angle, origin) in enumerate(zip(angles, origins)):
        angle_rad = radians(angle)
        affine_matrix = get_affine_rotation_matrix(angle_rad, origin)

        rotated_img = img.transform(img.size,
            Image.AFFINE,
            data=affine_matrix.flatten(),
            resample=Image.BICUBIC
            )

        blend_height = 3
        mask = Image.new('L', (width, blend_height), 0)
        draw = ImageDraw.Draw(mask)
        for y in reversed(range(blend_height)):
            alpha = 126
            draw.line([(0, y), (width, y)], fill=alpha)

        rotate_mix = Image.new('RGB', (width, height))

        half_upside_cut_rotated = rotated_img.crop((0, 0, width, origin[1] - blend_height))
        half_downside_cut_origin = img.crop((0, origin[1], width, height))

        rotate_mix.paste(half_upside_cut_rotated, (0, 0))
        rotate_mix.paste(mask, (0, origin[1] - blend_height))
        rotate_mix.paste(half_downside_cut_origin, (0, origin[1]))

        img = rotate_mix

    return img

def angle_origin_pairing(width, height, cut, tangent):
    cut_positions = {
        "half_cut": {
            "angles":[-1*tangent*90],
            "origins": [(0, height // 2)]
            },
        "third_cut": {
            "angles":[-1*tangent*90, 2*tangent*90],
            "origins": [(0, 2*height//3), (width, height//3)]
            },
        "quarter_cut":{
            "angles":[-1*tangent*90, 2*tangent*90, -3*tangent*90],
            "origins": [(0, 3*height//4), (width, 2*height//4), (0, height//4)]
            }
        }

    angles = cut_positions[cut]["angles"]
    origins = cut_positions[cut]["origins"]

    return angles, origins

ds = load_dataset("naver-clova-ix/cord-v2", split="train")

images = []
gts = []

cuts = ["half_cut", "third_cut", "quarter_cut"]
tangents = [1/60, 2/60, 3/60, 4/60]

for img_idx, sample in enumerate(tqdm(ds)):
    img = sample["image"]

    images.append(img)
    gts.append(sample["ground_truth"])

    width, height = img.size
    for tan_idx, tangent in enumerate(tangents):
        for cut in cuts:
            angles, origins = angle_origin_pairing(width, height, cut, tangent)
            rotated_img = rotate_and_cut_image(img, angles, origins, width, height)

            images.append(rotated_img)
            gts.append(sample["ground_truth"])

print(f"Total images: {len(images)}")

new_ds=[]
chunk_size = 5
for i in tqdm(range(0, len(images), chunk_size)):
    chunk = {
        "image": images[i:i + chunk_size],
        "ground_truth": gts[i:i + chunk_size]
    }
    new_ds.append(Dataset.from_dict(chunk))

new_ds = concatenate_datasets(new_ds)
new_ds.push_to_hub("doolayer/cord-v2-rotated-cut", private=True)