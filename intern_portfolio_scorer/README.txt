========================================================
  GRAPHURA — INTERN PORTFOLIO READINESS SCORER
  Flask + Logistic Regression Web Application
========================================================

FOLDER STRUCTURE
----------------
intern_portfolio_scorer/
  app.py                          <- Flask web server (run this)
  generate_data.py                <- Generates the dataset (run once)
  train_model.py                  <- Trains the ML model (run once)
  requirements.txt                <- Python dependencies
  final_intern_readiness_dataset.xlsx  <- Dataset (auto-generated)
  best_intern_readiness_model.pkl      <- Trained model (auto-generated)
  label_encoder.pkl                    <- Label encoder (auto-generated)
  templates/
    base.html                     <- Shared navbar/layout
    index.html                    <- Dashboard page
    predict.html                  <- Score Intern page
    leaderboard.html              <- Leaderboard page

HOW TO RUN (first time)
-----------------------
1. Install dependencies:
      pip install -r requirements.txt

2. Generate the dataset:
      python generate_data.py

3. Train the model:
      python train_model.py

4. Start the website:
      python app.py

5. Open browser at:
      http://127.0.0.1:5000

HOW TO RUN (after first time)
------------------------------
Just run:
      python app.py

The .pkl and .xlsx files are already saved.

PAGES
-----
/              -> Dashboard (KPIs, charts, top performers)
/predict       -> Score an intern with sliders
/leaderboard   -> Filter & rank all interns

MODEL
-----
Algorithm      : Logistic Regression (L2, balanced class weights)
Test Accuracy  : 96.14%
Balanced Acc   : 96.32%
CV Accuracy    : 96.91%
Classes        : Job Ready / Almost Ready / Needs Improvement
Features       : 14 engineered features

========================================================
  Graphura India Private Limited  |  2025-26
========================================================
