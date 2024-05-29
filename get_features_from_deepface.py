import os

import numpy as np
import tqdm

from deepface import DeepFace

if __name__ == '__main__':
    imgs_path = 'sdxl-faces/imgs'
    imgs = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path)]
    imgs.sort()

    features = []
    img_names = []
    pbar = tqdm.tqdm(total=len(imgs))
    for img in imgs:
        try:
            res = DeepFace.represent(img, model_name="VGG-Face")
            features.append(res[0]['embedding'])
            img_names.append(img)
        except Exception as e:
            pass
        pbar.update(1)

    features_np = np.array(features)
    np.save("features_vgg_.npy", features_np)
    with open("img_names.txt", "w") as f:
        f.write('\n'.join(img_names))
