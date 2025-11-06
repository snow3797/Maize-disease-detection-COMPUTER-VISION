# src/train_classical.py
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_images_from_folder, save_label_encoder
from feature_extraction import extract_hog_batch
from deep_features import DeepFeatureExtractor


def main(data_dir, model_out='models/classical_model.joblib', le_out='artifacts/label_encoder.joblib', method='hog'):
    X, y = load_images_from_folder(data_dir, target_size=(224,224))
    le = save_label_encoder(y, le_out)
    y_enc = le.transform(y)

    if method == 'hog':
        Xf = extract_hog_batch(X)
    elif method == 'deep':
        ext = DeepFeatureExtractor()
        Xf = ext.extract(X)
    else:
        raise ValueError('Unknown method')

    X_train, X_test, y_train, y_test = train_test_split(Xf, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # quick SVM baseline
    svc = SVC(probability=True)
    params = {'C':[0.1,1,10], 'kernel':['rbf','linear']}
    grid = GridSearchCV(svc, params, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print('Best:', grid.best_params_)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best, model_out)
    print('Saved', model_out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', default='hog', choices=['hog','deep'])
    args = parser.parse_args()
    main(args.data, method=args.method)