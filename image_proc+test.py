import cv2
import numpy as np
from matplotlib import pyplot as plt
import keras
keras.backend.set_image_data_format('channels_first')
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pickle

img = cv2.imread('test2.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow("wo", img)
cv2.waitKey(0)

train_data = []

threshed = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    img2 = threshed[y:y + h + 10, x:x + w + 10]
    image = cv2.resize(img2, (45, 45))
    cv2.imshow("work", image)
    cv2.waitKey(0)

    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    cv2.rectangle(threshed, (x, y), (x + w, y + h), (127, 255, 127), 2)

    train_data.append(image)

cv2.imshow('result', threshed)
cv2.waitKey()
cv2.destroyAllWindows()

def classify(image):
    model_file = "model.model"
    labels = "labels.pickle"
    model = load_model(model_file)
    file = open(labels, "rb").read()
    mlb = pickle.loads(file)
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]

    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))

    print("This Symbol is :", ' '.join(mlb.classes_[proba.argmax(axis=-1)]))

for image in train_data:
    classify(image)
