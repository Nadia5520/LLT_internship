"""
classifier_inference.py

Utilities to load a trained Keras classifier and predict on image crops.
Also contains functions to produce embeddings and run nearest-neighbor search
against a reference set of artwork images+metadata.

Example usage:
    from classifier_inference import Classifier
    clf = Classifier('models/margo_classifier.keras', feature_model_path='models/feature_extractor.keras')
    prob, label = clf.predict_crop(cv2.imread('crop.jpg'))
    embedding = clf.embed_crop(cv2.imread('crop.jpg'))
    neigh = clf.build_index('/path/to/reference_images', metadata_csv='meta.csv')  # optional
    match = clf.query_embedding(embedding, top_k=3)
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pickle

class Classifier:
    def __init__(self, model_path, feature_model_path=None, img_size=(224,224), class_names=None):
        self.img_size = img_size
        self.model = tf.keras.models.load_model(model_path)
        # determine classes: if binary, class_names should be provided or default
        self.class_names = class_names if class_names is not None else ['other', 'margo_veillon']
        # load or build a feature extractor
        if feature_model_path and os.path.exists(feature_model_path):
            self.feature_extractor = tf.keras.models.load_model(feature_model_path)
        else:
            # attempt to reuse model but strip last layers to get embeddings
            # find GlobalAveragePooling2D output index
            layer_idx = None
            for i, layer in enumerate(self.model.layers[::-1]):
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    layer_idx = len(self.model.layers) - i - 1
                    break
            if layer_idx is None:
                # fallback: use model until last-but-one layer
                self.feature_extractor = tf.keras.Model(self.model.input, self.model.layers[-2].output)
            else:
                self.feature_extractor = tf.keras.Model(self.model.input, self.model.layers[layer_idx].output)

        # nearest-neighbor index and metadata
        self._nn = None
        self._metadata = None

    def _prep(self, img):
        # img: BGR (OpenCV) image -> convert to RGB, resize, expand dims
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, self.img_size)
        arr = tf.keras.applications.resnet.preprocess_input(np.array(resized, dtype=np.float32))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict_crop(self, img):
        x = self._prep(img)
        preds = self.model.predict(x)
        if preds.shape[-1] == 1:
            prob = float(preds[0,0])
            label = self.class_names[1] if prob >= 0.5 else self.class_names[0]
            return prob, label
        else:
            idx = int(np.argmax(preds[0]))
            return float(preds[0,idx]), self.class_names[idx]

    def embed_crop(self, img):
        x = self._prep(img)
        emb = self.feature_extractor.predict(x)
        emb = emb.reshape(-1)
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-10
        return emb / norm

    def build_index(self, reference_folder, metadata=None, batch=32, use_faiss=False, index_path='index.pkl'):
        """
        Build a nearest-neighbor index over images in reference_folder.
        metadata: optional dict mapping filename -> metadata (title, year, etc).
        Saves index and metadata to index_path (pickle).
        """
        files = [f for f in os.listdir(reference_folder) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))]
        embeddings = []
        filenames = []
        for f in files:
            p = os.path.join(reference_folder, f)
            img = cv2.imread(p)
            if img is None:
                continue
            emb = self.embed_crop(img)
            embeddings.append(emb)
            filenames.append(f)
        embeddings = np.vstack(embeddings).astype(np.float32)
        # build sklearn nearest neighbor (fast enough for hundreds-thousands)
        nn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine').fit(embeddings)
        self._nn = nn
        self._filenames = filenames
        self._embeddings = embeddings
        self._metadata = metadata or {}
        # save to disk
        with open(index_path, 'wb') as f:
            pickle.dump({
                'filenames': filenames,
                'embeddings': embeddings,
                'metadata': self._metadata
            }, f)
        return index_path

    def load_index(self, index_path='index.pkl'):
        with open(index_path, 'rb') as f:
            d = pickle.load(f)
        self._filenames = d['filenames']
        self._embeddings = d['embeddings']
        self._metadata = d.get('metadata', {})
        self._nn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine').fit(self._embeddings)
        return True

    def query_embedding(self, emb, top_k=3):
        if self._nn is None:
            raise RuntimeError("Index not built or loaded")
        dists, idxs = self._nn.kneighbors(emb.reshape(1, -1), n_neighbors=top_k, return_distance=True)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            fname = self._filenames[int(idx)]
            meta = self._metadata.get(fname, {})
            results.append({'filename': fname, 'dist': float(dist), 'metadata': meta})
        return results