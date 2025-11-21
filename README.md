# IMDB Sentiment Project — Final Compatible Version

This version fixes all compatibility issues:
- ✅ Loads old models with `time_major` safely (LSTM fix)
- ✅ Loads old tokenizers referencing `keras.src.preprocessing`
- ✅ Compatible with Python 3.10+ and TensorFlow 2.10+
- ✅ Works on CPU-only laptops (Intel i3 / UHD Graphics)

## How to Use
1. Place `model.h5` and `tokenizer.pkl` in this folder.
2. Run the following:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   python run_sentiment.py
   ```
3. Enter a movie review when prompted.

## Notes
- Ignore warnings about input_length or optimizer state — they do not affect predictions.
- If your tokenizer file has a name like `tokenizer (1).pkl`, rename it to `tokenizer.pkl`.
