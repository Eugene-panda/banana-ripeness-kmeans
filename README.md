# Banana Ripeness Streamlit App

The Streamlit app uses the same proposed pipeline as the Colab experiment:

Upload → Resize → Grayscale → Median Filter 3×3 → Grayscale K-Means (K=3)
→ 9 segmentation-derived features → StandardScaler → RBF SVM → Prediction

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python packages
- `COLAB_EXPORT_MODEL_CELL.py` — paste this into the final cell of the completed Colab notebook

## Step 1 — export the trained model from Colab

Open the completed K-Means notebook. After `kmeans_search.fit(...)` and evaluation have run,
paste and run `COLAB_EXPORT_MODEL_CELL.py`.

It downloads:

`kmeans_banana_svm.joblib`

That file contains the fitted StandardScaler + SVM pipeline and the preprocessing settings.

## Step 2 — put the files together

```text
banana_kmeans_streamlit/
├── app.py
├── requirements.txt
└── kmeans_banana_svm.joblib
```

## Step 3 — install dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — run Streamlit

```bash
streamlit run app.py
```

The app will show the uploaded image, grayscale preprocessing, K-Means segmentation,
dark/medium/bright binary cluster masks, the nine K-Means features, predicted ripeness,
confidence, and the SVM class probabilities.

Do not retrain the model inside Streamlit. The Colab notebook is used for training;
Streamlit only performs inference using the saved model.
