import os
import json

import tqdm

from deepface import DeepFace

if __name__ == '__main__':
    imgs_path = 'sdxl-faces/imgs'
    imgs = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path)]
    imgs.sort()
    with open("labels.json") as f:
        labels = json.load(f)
    img2label = {}

    for label in labels:
        img2label[label['face_image']] = label['age']

    ages = []
    categories = []

    pbar = tqdm.tqdm(total=len(imgs))
    for img in imgs:
        if os.path.basename(img) not in img2label:
            continue
        res = DeepFace.analyze(img, ['age'], enforce_detection=False)
        categories.append(img2label[os.path.basename(img)])
        ages.append(res[0]['age'])

        pbar.update(1)

    with open("ages.txt", "w") as f:
        f.write('\n'.join(map(str, ages)))
    with open("categories.txt", "w") as f:
        f.write('\n'.join(map(str, categories)))
