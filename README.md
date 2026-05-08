# CS506 Big Data Project — Manual Steps Guide
## GitHub Repository Intelligence Analyzer
**Muhammad Abubakar Siddique | 2023-AG-10411 | UAF Section E1**

---

## 📁 Project Structure

```
github_analyzer/
├── app.py                  ← Streamlit frontend (run this last)
├── backend.py              ← Helper functions / routes
├── preprocess.py           ← Data cleaning pipeline
├── train_model.py          ← ML model training
├── requirements.txt        ← Python dependencies
├── data/
│   ├── github_dataset.csv  ← You download this from Kaggle
│   └── cleaned_data.csv    ← Auto-generated after preprocess.py
├── models/
│   ├── model.pkl           ← Auto-generated after train_model.py
│   ├── metrics.json        ← Auto-generated after train_model.py
│   └── feature_columns.json
└── notebook/
    └── eda_notebook.py     ← EDA script (run in Jupyter or Colab)
```

---

## ✅ STEP 1 — Install Python and VS Code

1. Download Python 3.11 from https://www.python.org/downloads/
2. During installation, **check the box: "Add Python to PATH"**
3. Download VS Code from https://code.visualstudio.com/
4. Open VS Code → Install the **Python extension** (by Microsoft)

---

## ✅ STEP 2 — Download the Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/nikhil25803/github-dataset
2. Create a free Kaggle account if you don't have one
3. Click the blue **Download** button
4. Extract the ZIP file
5. Find the file named **`github_dataset.csv`**
6. Place it inside your project folder at:
   ```
   github_analyzer/data/github_dataset.csv
   ```

---

## ✅ STEP 3 — Set Up the Project

1. Open **Terminal** (or Command Prompt on Windows)
2. Navigate to your project folder:
   ```bash
   cd path/to/github_analyzer
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
5. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```
   ⏳ This may take 3–5 minutes.

---

## ✅ STEP 4 — Run the EDA Notebook (Google Colab)

This is where you do your Exploratory Data Analysis (EDA) — required for submission.

1. Go to https://colab.research.google.com
2. Click **New Notebook**
3. Upload your `data/cleaned_data.csv` file to Colab (Files panel on left)
   - **Note:** Run `preprocess.py` first (Step 5 below), then come back here
4. Open `notebook/eda_notebook.py` in VS Code
5. Copy each block marked `# CELL N` into a separate Colab cell
6. Run all cells (Runtime → Run All)
7. Download the generated charts from the `notebook/` folder
8. Take **screenshots** of the charts — you'll need them for your report

---

## ✅ STEP 5 — Clean and Preprocess the Dataset

In your terminal (with virtual environment active):
```bash
python preprocess.py
```

**Expected output:**
```
=======================================================
  GitHub Repository Intelligence Analyzer
  Data Preprocessing Pipeline
=======================================================
[1/6] Loading dataset from: data/github_dataset.csv
      Rows: 1,052  |  Columns: [...]
[2/6] Renaming and selecting relevant columns...
[3/6] Cleaning data (nulls, duplicates, types)...
      Rows removed: 12  |  Remaining: 1,040
[4/6] Engineering features...
      Target median stars threshold: 27
[5/6] Encoding categorical variables...
[6/6] Saving cleaned data to: data/cleaned_data.csv

✅  Preprocessing complete! File saved: data/cleaned_data.csv
```

---

## ✅ STEP 6 — Train the ML Model

In your terminal:
```bash
python train_model.py
```

**Expected output (approximate):**
```
[4/7] Training Random Forest classifier...
      CV Accuracy: 0.8921 ± 0.0134
[5/7] Training Gradient Boosting classifier...
      CV Accuracy: 0.8845 ± 0.0159
[6/7] Evaluating both models on test set...
      ✅  Best model: Random Forest (AUC = 0.94)
[7/7] Saving model and artifacts...
      Model saved: models/model.pkl
🎉  Training complete!
```

⏳ Training may take 1–3 minutes.

---

## ✅ STEP 7 — Launch the Streamlit Frontend

In your terminal:
```bash
streamlit run app.py
```

Your browser will automatically open at: **http://localhost:8501**

You should see a dark-themed dashboard with:
- 📊 Dashboard — KPI cards, charts, and distributions
- 🔮 Predict Popularity — enter any repo's stats and get a prediction
- 📈 Language Insights — popularity rates per language
- 🧪 Model Performance — accuracy, AUC-ROC, confusion matrix

**To stop the app:** Press `Ctrl + C` in the terminal.

---

## ✅ STEP 8 — Test the Prediction Feature

1. In the Streamlit app, click **"🔮 Predict Popularity"**
2. Enter these values to test:
   - Stars: `4500`
   - Forks: `850`
   - Issues: `120`
   - Pull Requests: `210`
   - Contributors: `35`
3. Click **"🚀 Analyze Repository"**
4. You should see: **"🔥 Likely Popular"** with a confidence score

---

## 📸 Screenshots to Take (for Submission)

Take screenshots of all of these for your project report:

- [ ] Terminal output of `preprocess.py` running successfully
- [ ] Terminal output of `train_model.py` — showing model accuracy
- [ ] Streamlit Dashboard page with all 4 charts visible
- [ ] Prediction result page (a popular + a non-popular example)
- [ ] Language Insights page
- [ ] Model Performance page (confusion matrix)
- [ ] EDA plots from Colab (correlation heatmap, stars distribution, etc.)

---

## 🐛 Common Issues & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: github_dataset.csv` | Check file is in `data/` folder |
| `FileNotFoundError: cleaned_data.csv` | Run `preprocess.py` first |
| `FileNotFoundError: model.pkl` | Run `train_model.py` first |
| Streamlit doesn't open | Manually go to http://localhost:8501 |
| Port already in use | Run `streamlit run app.py --server.port 8502` |

---

## 📋 Submission Checklist (by 20 May)

- [ ] `preprocess.py` — runs without errors
- [ ] `train_model.py` — model accuracy ≥ 80%
- [ ] `backend.py` — all route functions working
- [ ] `app.py` — Streamlit app runs and all 4 pages work
- [ ] `notebook/eda_notebook.py` — EDA completed in Colab
- [ ] All screenshots taken
- [ ] ZIP entire `github_analyzer/` folder for submission

---

*Project by Muhammad Abubakar Siddique — CS506 Big Data Analytics — UAF 2026*
