import os
import pickle
import importlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_FILENAME = "model.h5"
TOKENIZER_FILENAME = "tokenizer.pkl"

# -------------------------------------------------------
#  TOKENIZER COMPATIBILITY LOADER
# -------------------------------------------------------
def load_tokenizer_compat(tokenizer_path):
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Fix old Keras import paths in tokenizer
            if module.startswith("keras.src.preprocessing"):
                new_module = module.replace(
                    "keras.src.preprocessing", "tensorflow.keras.preprocessing"
                )
                try:
                    mod = importlib.import_module(new_module)
                    return getattr(mod, name)
                except:
                    pass

            if module.startswith("keras.preprocessing"):
                new_module = module.replace(
                    "keras.preprocessing", "tensorflow.keras.preprocessing"
                )
                try:
                    mod = importlib.import_module(new_module)
                    return getattr(mod, name)
                except:
                    pass

            return super().find_class(module, name)

    with open(tokenizer_path, "rb") as f:
        try:
            return CompatUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)

# -------------------------------------------------------
#  MODEL + TOKENIZER LOADER
# -------------------------------------------------------
def load_model_and_tokenizer(model_path=MODEL_FILENAME, tokenizer_path=TOKENIZER_FILENAME):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    # Fix for old models using LSTM(time_major=True)
    from tensorflow.keras.layers import LSTM
    class CompatibleLSTM(LSTM):
        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

    custom_objects = {"LSTM": CompatibleLSTM}

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    tokenizer = load_tokenizer_compat(tokenizer_path)

    return model, tokenizer

# -------------------------------------------------------
#  DETECT MAXLEN FROM MODEL
# -------------------------------------------------------
def detect_maxlen(model, fallback=200):
    try:
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            if shape[1] is not None:
                return int(shape[1])
    except:
        pass

    # fallback to first layer
    try:
        first_layer = model.layers[0]
        shp = first_layer.input_shape
        if isinstance(shp, (list, tuple)) and len(shp) >= 2:
            if shp[1] is not None:
                return int(shp[1])
    except:
        pass

    return fallback

# -------------------------------------------------------
#  TEXT PREPROCESSING
# -------------------------------------------------------
def preprocess_texts(tokenizer, texts, maxlen):
    seq = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return padded

# -------------------------------------------------------
#  INTERPRET MODEL OUTPUT
# -------------------------------------------------------
def interpret_prediction(pred):
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        label = "Positive" if prob >= 0.5 else "Negative"
        confidence = prob if prob >= 0.5 else 1 - prob
    else:
        idx = int(np.argmax(pred[0]))
        label = "Positive" if idx == 1 else "Negative"
        confidence = float(np.max(pred[0]))

    return label, confidence

# -------------------------------------------------------
#  PREDICT
# -------------------------------------------------------
def predict_review(model, tokenizer, review_text, maxlen=None):
    if maxlen is None:
        maxlen = detect_maxlen(model)

    x = preprocess_texts(tokenizer, [review_text], maxlen)
    pred = model.predict(x)
    label, confidence = interpret_prediction(pred)
    return label, confidence, pred

# -------------------------------------------------------
#  MANUAL TEST (optional)
# -------------------------------------------------------
def main():
    print("=== IMDB Sentiment Predictor (Final Compatible Version) ===\n")

    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print("Error loading model/tokenizer:", e)
        return

    maxlen = detect_maxlen(model)
    print(f"Detected/preset input length (maxlen) = {maxlen}\n")

    while True:
        review = input("Enter a movie review (or 'exit' to quit): ").strip()
        if review.lower() == "exit":
            print("Goodbye.")
            break

        label, conf, raw = predict_review(model, tokenizer, review, maxlen)
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {label} (confidence: {conf*100:.2f}%)\n")


if __name__ == "__main__":
    main()
