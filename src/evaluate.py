import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from utils import load_images_from_folder, load_label_encoder
from deep_features import DeepFeatureExtractor
from feature_extraction import extract_hog_batch


def eval_classical(model_path, data_dir, method='hog'):
    model = joblib.load(model_path)
    X, y = load_images_from_folder(data_dir, target_size=(224,224))
    le = load_label_encoder('artifacts/label_encoder.joblib')
    y_enc = le.transform(y)
    if method == 'hog':
        Xf = extract_hog_batch(X)
    else:
        ext = DeepFeatureExtractor()
        Xf = ext.extract(X)
    y_pred = model.predict(Xf)
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    print(confusion_matrix(y_enc, y_pred))


def eval_cnn(model_path, data_dir):
    model = load_model(model_path)
    X, y = load_images_from_folder(data_dir, target_size=(224,224))
    X = X.astype('float32')/255.0
    preds = model.predict(X, verbose=1)
    import numpy as np
    y_pred = preds.argmax(axis=1)
    le = load_label_encoder('artifacts/label_encoder.joblib')
    y_enc = le.transform(y)
    print(classification_report(y_enc, y_pred, target_names=le.classes_))
    print(confusion_matrix(y_enc, y_pred))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--type', choices=['cnn','classical'], default='cnn')
    args = parser.parse_args()
    if args.type == 'cnn':
        eval_cnn(args.model, args.data)
    else:
        eval_classical(args.model, args.data)