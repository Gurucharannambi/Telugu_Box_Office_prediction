# 🎬 Telugu Box Office Predictor

A machine learning web application that predicts worldwide box office collections for Telugu movies using Gradient Boosting Regression.

---

## 📁 Project Structure

```
telugu_boxoffice/
├── app.py                 # Main Streamlit application
├── telugu_movies.csv      # Telugu movies dataset (90+ movies, 2009–2025)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Features

| Page | Description |
|------|-------------|
| 🔮 Predict Box Office | Input movie details → get predicted worldwide collection + verdict |
| 📊 Movie Trends | Year-wise, hero, director, genre, season trend charts |
| 🤖 Model Performance | R², MAE, RMSE, feature importance, residual distribution |
| 🔍 Actual vs Predicted | Compare ML predictions vs real-world box office |

---

## 🧠 ML Model

- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Features:** Year, Budget, Screens, Hero, Director, Genre, Season, Is Sequel, Big Music
- **Target:** Worldwide Box Office Collection (₹ Crores)
- **Train/Test Split:** 80/20

---

## 💻 Run Locally

### 1. Clone / Download the project

```bash
cd telugu_boxoffice
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free Hosting)

### Step 1 — Push to GitHub

1. Create a new GitHub repository (e.g., `telugu-boxoffice-predictor`)
2. Upload all project files:
   - `app.py`
   - `telugu_movies.csv`
   - `requirements.txt`
   - `README.md`

```bash
git init
git add .
git commit -m "Telugu box office predictor app"
git remote add origin https://github.com/YOUR_USERNAME/telugu-boxoffice-predictor.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `telugu-boxoffice-predictor`
5. Set **Main file path** to: `app.py`
6. Click **"Deploy"**

✅ Your app will be live at:  
`https://your-username-telugu-boxoffice-predictor-app-XXXXXX.streamlit.app`

---

## 📊 Dataset Info

- **90+ Telugu movies** from 2009 to 2025
- **Features:** Movie name, Year, Hero, Director, Genre, Budget, Screens, Release Season, Sequel flag, Music flag
- **Target:** Worldwide collection in ₹ Crores
- **Verdicts:** Blockbuster / Hit / Average / Flop

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Model | Scikit-learn (GradientBoostingRegressor) |
| Visualizations | Plotly |
| Data Processing | Pandas, NumPy |
| Hosting | Streamlit Cloud (Free) |

---

## 📌 Notes

- The model is trained on a curated dataset. Predictions are estimates, not exact figures.
- For best accuracy, real-world data from sources like Box Office India / Sacnilk can be added.
- The model can be retrained by adding more rows to `telugu_movies.csv`.

---

*Built for academic/faculty presentation purposes.*
