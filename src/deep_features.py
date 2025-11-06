import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model


class DeepFeatureExtractor:
    def __init__(self, input_shape=(224,224,3), pooling='avg'):
        base = ResNet50(weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)
        self.model = base

    def extract(self, X):
        # X expected as uint8 images shaped (N,H,W,3)
        X_proc = preprocess_input(X.astype('float32'))
        feats = self.model.predict(X_proc, verbose=0)
        return feats