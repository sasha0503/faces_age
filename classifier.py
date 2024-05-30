import json
import os.path
import shutil

import pickle

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np

if __name__ == '__main__':
    for file in os.listdir('errors'):
        os.remove(os.path.join('errors', file))

    with open("labels.json") as f:
        labels = json.load(f)
    labels.sort(key=lambda x: x['source_image'])

    img2label = {}
    for label in labels:
        age = label['age']
        ethnicity = label['ethnicity']
        sex = label['sex']
        img = label['face_image']

        img2label[img] = f"{age}_{sex}_{ethnicity}-{img}"

    features = np.load("features_vgg.npy")
    with open("img_names.txt") as f:
        img_names = f.read().split()

    X, y = [], []
    for img_i, img in enumerate(img_names):
        img = os.path.basename(img)
        if img not in img2label:
            continue
        X.append(features[img_i])
        y.append(img2label[img])

    indx = np.random.permutation(len(X))
    X = np.array(X)[indx]
    y = np.array(y)[indx]

    chunk_size = len(X) // 10
    scores = []
    f1_scores = []
    for i in range(10):
        print(f"Fold {i}")
        X_train = np.concatenate([X[:i * chunk_size], X[(i + 1) * chunk_size:]])
        X_test = X[i * chunk_size:(i + 1) * chunk_size]
        y_train_img = np.concatenate([y[:i * chunk_size], y[(i + 1) * chunk_size:]])
        y_test_img = y[i * chunk_size:(i + 1) * chunk_size]
        y_test = [i.split('-')[0] for i in y_test_img]
        y_train = [i.split('-')[0] for i in y_train_img]

        svc_clf = SVC(C=0.5, random_state=0).fit(X_train, y_train)

        res = svc_clf.predict(X_test)
        res = ['child' in i for i in res]
        simple_y_test = ['child' in i for i in y_test]

        scores.append(accuracy_score(simple_y_test, res))

        # calculate f1 score
        tp = sum([1 for i in range(len(simple_y_test)) if simple_y_test[i] and res[i]])
        fp = sum([1 for i in range(len(simple_y_test)) if not simple_y_test[i] and res[i]])
        fn = sum([1 for i in range(len(simple_y_test)) if simple_y_test[i] and not res[i]])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

        error_idx = []
        for j, error in enumerate(simple_y_test):
            if res[j] != error:
                error_idx.append(j)

        errors = [os.path.join('sdxl-faces/imgs', y_test_img[i].split('-')[-1]) for i in error_idx]
        os.makedirs('errors', exist_ok=True)
        for j, error in enumerate(errors):
            idx = error_idx[j]
            correct = 'adult' if not simple_y_test[idx] else "child"
            shutil.copy(error, f'errors/{correct}_{j}_{i}.jpg')

    print("Mean accuracy:", np.mean(scores))
    print("Mean f1 score:", np.mean(f1_scores))

    y = [i.split('-')[0] for i in y]
    svc_clf = SVC(C=0.5, random_state=0).fit(X, y)
    with open("svc_clf.pkl", "wb") as f:
        pickle.dump(svc_clf, f)
