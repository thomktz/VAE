######
# Crops, rotates and scales the original pictures using facial landmarks
######
# %% Imports
import numpy as np
import cv2
import os
import tqdm
import glob
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition
import matplotlib
# %%
image_size = 128
intermediate_size = 512
SAVE_PATH = "D:\\Github\\Data\\new_treated\\"
PATH = "D:\\Github\\Data\\new_images\\"

# %%

def avg(points):
    return np.average(points, axis = 0)

def get_landmarks(path):
    dict = face_recognition.face_landmarks(cv2.imread(path))[0]
    left_eye = avg(np.array(dict["left_eye"]))
    right_eye = avg(np.array(dict["right_eye"]))
    mouth = avg(np.array(dict["bottom_lip"]) + np.array(dict["top_lip"]))
    return left_eye, right_eye, mouth

class Facealigner():
    def __init__(self, desired_left_eye = (0.375, 0.47), desired_right_eye = (0.625,0.47), desired_mouth = (0.5,0.7)):
        self.dle = desired_left_eye
        self.dre = desired_right_eye
        self.dfw = desired_right_eye[0]-desired_left_eye[0] #desired face width
        self.dfh = desired_mouth[1]-desired_left_eye[1]
        self.mouth = desired_mouth

    def align(self, image_path):
        lm = get_landmarks(image_path)
        if lm is not None:
            image = matplotlib.image.imread(image_path)
            left_eye, right_eye, mouth = lm
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            dist = np.sqrt(dX**2 + dY**2)
            desired_dist = intermediate_size * self.dfw
            scale = desired_dist/dist
            rotation_point = ((left_eye[0]+right_eye[0])//2, (left_eye[1]+ right_eye[1])//2)
            M = cv2.getRotationMatrix2D(rotation_point, angle, scale)
            tX = intermediate_size/2 - rotation_point[0]
            tY = intermediate_size * self.dle[1] - rotation_point[1]
            M[0, 2] += tX
            M[1, 2] += tY
            output = cv2.warpAffine(image, M, (intermediate_size, intermediate_size))
            #print(output.shape)
            return cv2.resize(output, (128,128))
# %%
def align_and_show(image_path):
    f = Facealigner()
    img = f.align(image_path)
    #print(img)
    #print(img.shape)
    plt.matshow(img)

def save(image_path, name):
    f = Facealigner()
    img = f.align(PATH + image_path)
    print(img.shape)
    matplotlib.image.imsave(SAVE_PATH + name + ".png", img)


    
"""
import time
paths = [f"D:\\Github\\Data\\cats\\CAT_00\\00000001_{s}.jpg" for s in ["000", "005", "008", "011", "012", "016", "017"]]
for p in paths:
    align_and_show(p)
    plt.show()
    time.sleep(1)
"""
# %%

def treat_all_images():
    fa = Facealigner()
    total_nb_of_images = 0
    paths = glob.glob(PATH+"*")
    for path in tqdm.tqdm(paths):
        out = fa.align(path)
        matplotlib.image.imsave(SAVE_PATH  + str(total_nb_of_images) + ".png", out)
        total_nb_of_images +=1

# %%
