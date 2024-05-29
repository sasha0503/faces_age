import json
import os

import cv2

if __name__ == '__main__':
    imgs_path = 'sdxl-faces/imgs'
    imgs = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path)]
    imgs.sort()
    with open("labels.json") as f:
        labels = json.load(f)

    j = 1051
    while True:
        img, label = imgs[j], labels[j]
        age = label['age']
        if age == "child":
            j += 1
            continue
        img_cv2 = cv2.imread(img)
        cv2.putText(img_cv2, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('img', img_cv2)
        k = cv2.waitKey(0)
        if k == ord('f'):
            if label['age'] == "child":
                labels[j]['age'] = "adult"
            else:
                labels[j]['age'] = "child"
            with open("labels.json", "w") as f:
                json.dump(labels, f)
        elif k == ord('q'):
            break
        elif k == ord('z'):
            j -= 1
        else:
            j += 1
        print(j, len(imgs))
