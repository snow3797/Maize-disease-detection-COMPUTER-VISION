# Maize Leaf Disease Detection — Project Blueprint + Code

This document contains a connected, runnable pipeline for detecting maize leaf diseases from images. It includes:

- Project structure and instructions
- Scripts for data loading, preprocessing, feature extraction (HOG/SIFT + deep features), classical ML training, CNN training / fine-tuning, evaluation, and deployment as a FastAPI REST endpoint.

> **Important**: Run this in a Python 3.9+ environment. Use a virtualenv. GPU recommended for CNN training but not required for classical models.

---

## Project structure

```
maize-disease-project/
├── data/                  # your images: organized by class subfolders
│   ├── Common_Rust/
│   ├── Leaf_Spot/
│   ├── Maize_Leaf_Blight/
│   └── Healthy/
├── models/
│   ├── cnn_model.h5
│   └── classical_model.joblib
├── artifacts/
│   └── label_encoder.joblib
├── notebooks/             # optional: jupyter notebooks
├── src/
│   ├── utils.py
│   ├── feature_extraction.py
│   ├── deep_features.py
│   ├── train_classical.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── api.py
├── requirements.txt
└── README.md
```

---

## `requirements.txt` (minimal)

```
fastapi
uvicorn[standard]
scikit-learn
numpy
pandas
opencv-python-headless
opencv-contrib-python-headless
scikit-image
joblib
matplotlib
tensorflow>=2.6
pillow
tqdm
```

> Note: `opencv-contrib-python-headless` is required if you want SIFT. If you cannot install contrib build on your platform, omit SIFT and use HOG / deep features.

---

## `src/utils.py`

```python
# src/utils.py
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib


def load_images_from_folder(folder, target_size=(224,224), max_per_class=None):
    X = []
        y = []
            classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))])
                for cls in classes:
                        cls_folder = os.path.join(folder, cls)
                                files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('jpg','jpeg','png'))]
                                        if max_per_class:
                                                    files = files[:max_per_class]
                                                            for f in files:
                                                                        try:
                                                                                        im = Image.open(f).convert('RGB')
                                                                                                        im = im.resize(target_size)
                                                                                                                        arr = np.array(im)
                                                                                                                                        X.append(arr)
                                                                                                                                                        y.append(cls)
                                                                                                                                                                    except Exception as e:
                                                                                                                                                                                    print(f"Skipped {f}: {e}")
                                                                                                                                                                                        X = np.array(X)
                                                                                                                                                                                            y = np.array(y)
                                                                                                                                                                                                return X,y


                                                                                                                                                                                                def save_label_encoder(y, path):
                                                                                                                                                                                                    le = LabelEncoder()
                                                                                                                                                                                                        le.fit(y)
                                                                                                                                                                                                            joblib.dump(le, path)
                                                                                                                                                                                                                return le


                                                                                                                                                                                                                def load_label_encoder(path):
                                                                                                                                                                                                                    return joblib.load(path)
```

---

## `src/feature_extraction.py` — HOG and SIFT

```python
# src/feature_extraction.py
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
```

---

## `src/deep_features.py` — pre-trained CNN feature extractor

```python
# src/deep_features.py
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
```

---

## `src/train_classical.py` — extract features + train SVM / RandomForest

```python
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
```

---

## `src/train_cnn.py` — build / fine-tune a Keras CNN

```python
# src/train_cnn.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_finetune_model(num_classes, input_shape=(224,224,3), base_trainable=False):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model


def main(data_dir, out_path='models/cnn_model.h5', batch_size=32, epochs=15):
    # Use ImageDataGenerator with validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=20, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=batch_size, subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), batch_size=batch_size, subset='validation')

    num_classes = train_gen.num_classes
    model = build_finetune_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(out_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    model.save(out_path)
    print('Saved', out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    main(args.data, epochs=args.epochs)
```

---

## `src/evaluate.py` — evaluate saved models

```python
# src/evaluate.py
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
```

---

## `src/api.py` — FastAPI for inference

```python
# src/api.py
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from utils import load_label_encoder
from deep_features import DeepFeatureExtractor

app = FastAPI(title='Maize Leaf Disease Detection')

# Choose which model to load: 'cnn' or 'classical'
MODEL_TYPE = 'cnn'  # or 'classical'

if MODEL_TYPE == 'cnn':
    cnn = load_model('models/cnn_model.h5')
else:
    clf = joblib.load('models/classical_model.joblib')

le = load_label_encoder('artifacts/label_encoder.joblib')

# If classical model uses deep features, we still need extractor
deep_ext = DeepFeatureExtractor()


def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = image.resize((224,224))
    return np.array(image)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)

    if MODEL_TYPE == 'cnn':
        x = img.astype('float32')/255.0
        x = np.expand_dims(x,0)
        preds = cnn.predict(x)
        proba = preds[0]
        idx = int(np.argmax(proba))
        label = le.inverse_transform([idx])[0]
        return JSONResponse({'label': label, 'probabilities': {c: float(p) for c,p in zip(le.classes_, proba)}})

    else:
        # assume classical model trained on deep features
        feat = deep_ext.extract(np.expand_dims(img,0))
        proba = clf.predict_proba(feat)[0]
        idx = int(np.argmax(proba))
        label = le.inverse_transform([idx])[0]
        return JSONResponse({'label': label, 'probabilities': {c: float(p) for c,p in zip(le.classes_, proba)}})


# Run: uvicorn src.api:app --reload --port 8000
```

---

## How the components flow (end-to-end)

1. Put images into `data/CLASS_NAME/*` folders.
2. Run `src/train_cnn.py --data data/` to train a CNN (or `src/train_classical.py --data data/ --method hog`).
3. Training scripts create `models/cnn_model.h5` or `models/classical_model.joblib` and `artifacts/label_encoder.joblib`.
4. Evaluate using `src/evaluate.py --model models/cnn_model.h5 --data data/ --type cnn`.
5. Start the API: `uvicorn src.api:app --reload --port 8000` and send images to `/predict`.


## Notes, tips & improvements

- **Data balancing**: if classes are imbalanced, use `class_weight` in model.fit or resampling.
- **Augmentation**: heavy augmentation helps generalize (rotate, flip, color jitter).
- **SIFT**: if you use SIFT, you might prefer a bag-of-visual-words (BoVW) pipeline (k-means on descriptors) instead of simple averaging.
- **Explainability**: add Grad-CAM for CNN to visualize where the model looks.
- **Monitoring**: log predictions in a DB or file for later analysis.
- **Productionizing**: dockerize the API, add auth, rate-limiting, and batch endpoints.


---

## Quick commands

- Install deps: `pip install -r requirements.txt`
- Train CNN: `python src/train_cnn.py --data data/ --epochs 10`
- Train classical (HOG + SVM): `python src/train_classical.py --data data/ --method hog`
- Evaluate: `python src/evaluate.py --model models/cnn_model.h5 --data data/ --type cnn`
- Run API: `uvicorn src.api:app --reload --port 8000`


---

If you'd like, I can:
- generate a Dockerfile and docker-compose for the API + model
- provide a sample HTML frontend to upload images
- add Grad-CAM implementation for interpretability

Tell me which of these you'd like next and I'll add the code.

---

## Additional parts — Docker, Frontend, Grad-CAM, CI, Tests, Deployment tips

### Dockerfile (API + model)

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# set workdir
WORKDIR /app

# copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts

# expose port
EXPOSE 8000

# run
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```


### docker-compose.yml (API + optional simple redis for caching)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./artifacts:/app/artifacts:ro
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
  # optional: add redis if you need caching or rate limiting
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```


### Simple HTML + JavaScript frontend (single-file)

```html
<!-- frontend.html -->
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Maize Leaf Classifier</title>
</head>
<body>
  <h2>Upload a maize leaf photo</h2>
  <input type="file" id="fileinput" accept="image/*" />
  <button id="send">Classify</button>
  <pre id="result"></pre>

<script>
  document.getElementById('send').onclick = async () => {
    const inp = document.getElementById('fileinput');
    if (!inp.files.length) return alert('Choose a file');
    const f = inp.files[0];
    const fd = new FormData();
    fd.append('file', f);
    const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: fd });
    const txt = await res.json();
    document.getElementById('result').textContent = JSON.stringify(txt, null, 2);
  }
</script>
</body>
</html>
```


### Grad-CAM implementation (for CNN interpretability)

```python
# src/gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # img_array: (1,H,W,3) scaled 0-1
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlaid = heatmap * alpha + img
    overlaid = np.uint8(overlaid / np.max(overlaid) * 255)
    return overlaid
```

Use the function by loading your model, getting the name of the last conv layer (e.g., `conv5_block3_out` for ResNet50), preparing an input image scaled to 0-1, and saving the overlayed result for visualization. This helps explain model predictions to farmers.


### Hyperparameter tuning & cross-validation tips

- For classical models (SVM, RF): use `GridSearchCV` or `RandomizedSearchCV` with stratified k-fold (e.g., `StratifiedKFold(n_splits=5)`), and evaluate on a held-out test set.
- For CNNs: use a validation split and callbacks (`ReduceLROnPlateau`, `EarlyStopping`). Optionally use K-fold by generating separate training/validation splits from filenames and training multiple runs; average metrics to estimate variance.
- Important hyperparameters: learning rate, optimizer (Adam/SGD+momentum), batch size, weight decay, dropout, augmentation ranges.


### Unit tests & model contract

Create lightweight unit tests that check:

- `utils.load_images_from_folder` returns correct shapes and labels for a small synthetic dataset.
- The API `/predict` accepts an image and returns JSON with `label` and `probabilities` keys.

Example using `pytest`:

```python
# tests/test_utils.py
from src.utils import load_images_from_folder
import numpy as np

def test_load_images(tmp_path):
    d = tmp_path / 'data'
    c = d / 'Healthy'
    c.mkdir(parents=True)
    # create a blank image
    import PIL.Image as Image
    img = Image.new('RGB', (224,224), color=(255,255,255))
    p = c / '1.png'
    img.save(p)
    X,y = load_images_from_folder(str(d))
    assert X.shape[0] == 1
    assert y[0] == 'Healthy'
```


### CI: GitHub Actions (basic)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest -q
```


### Monitoring & logging suggestions

- Log predictions (timestamp, image hash, predicted label, probabilities) to a file or lightweight DB (SQLite/Postgres) for later analysis.
- Add request rate limiting using API gateway or FastAPI middleware.
- Expose a health endpoint `/health` that returns model loaded and last reload timestamp.


### Quick curl example

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/leaf.jpg"
```


---

### What I added in this update

I appended the following to the project document:
- Dockerfile and docker-compose for containerized deployment
- Single-file HTML uploader to test the API locally
- Grad-CAM implementation and how to use it
- Hyperparameter tuning and cross-validation tips
- Unit test example using pytest
- GitHub Actions CI example
- Monitoring and logging suggestions
- Example `curl` command


---

If you'd like, I can now:
- produce a ready-to-run `docker build` + `docker-compose up` guide including sample small dataset, or
- implement Grad-CAM into the API to return heatmap images alongside predictions, or
- dockerize everything and prepare a deployable image for a cloud provider (AWS/GCP/Heroku) with step-by-step instructions.

Pick one and I'll add the code and steps.

