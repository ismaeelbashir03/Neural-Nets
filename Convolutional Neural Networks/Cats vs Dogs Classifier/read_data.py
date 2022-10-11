import pickle
import numpy as np
import os
import cv2
import random

# getting our data directory
data_dir = 'Petimages'

# creating labels
categories = ['Cat', 'Dog']

training_data = []

#setting image size
img_size = 80

# reading our data in
def create_training_data():

    # looping through each folder and each image and appending it to our training data
    for category in categories:
        path = os.path.join(data_dir, category)

        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append([new_array, class_num])
                print(img)
            except Exception as e: 
                pass # some images are broken
    

# running our function
create_training_data()

#shuffling our data
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()