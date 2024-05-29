import pickle

import numpy as np
from deepface import DeepFace

if __name__ == '__main__':
    classifier = pickle.load(open("svc_clf.pkl", "rb"))

    img = "sdxl-faces/imgs/00000000.png"
    embedding = DeepFace.represent(img, model_name="VGG-Face")[0]['embedding']
    embedding = np.array(embedding)
    print(classifier.predict(embedding.reshape(1, -1)))
