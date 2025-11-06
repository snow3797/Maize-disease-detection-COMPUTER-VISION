import cv2
import numpy as np
from skimage.feature import hog


def extract_hog_batch(X, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
    feats = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h = hog(gray,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys')
        feats.append(h)
    return np.array(feats)


def extract_sift_batch(X, nfeatures=0):
    # Requires opencv-contrib
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    all_desc = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            des = np.zeros((1,128), dtype=np.float32)
        # here we average descriptors as a simple global descriptor
        desc_mean = des.mean(axis=0)
        all_desc.append(desc_mean)
    return np.array(all_desc)