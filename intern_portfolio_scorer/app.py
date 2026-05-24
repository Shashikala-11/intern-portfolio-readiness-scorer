"""
app.py  —  Graphura Intern Portfolio Readiness Scorer
Flask web application with Logistic Regression backend.
"""
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# ── App setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# ── Load model artefacts ─────────────────────────────────────
model         = joblib.load(os.path.join(BASE_DIR, 'best_intern_readiness_model.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

# ── Load dataset ─────────────────────────────────────────────
df_raw = pd.read_excel(os.path.join(BASE_DIR, 'final_intern_readiness_dataset.xlsx'))

ROLES = [
    "Full-Stack Developer", "Software Intern", "Frontend Engineer",
    "Backend Engineer", "Data Analyst", "UI/UX Designer",
    "Data Science & Analytics", "MERN Stack Developer",
    "Digital Marketing", "Content Writing",
    "HR Department", "Sales & Marketing",
    "Graphic Design", "Video Editing", "Social Media Manager",
    "General", "DevOps Engineer",
]

# ── Feature engineering (must match train_model.py) ──────────
def engineer_features(data: dict) -> pd.DataFrame:
    tools_used = data.get('tools_used', 'Python')
    tool_count = len(str(tools_used).split(','))

    no_of_projects       = float(data['no_of_projects'])
    business_relevance   = float(data['business_relevance'])
    documentation_quality= float(data['documentation_quality'])
    avg_project_score    = float(data['avg_project_score'])
    presentation_score   = float(data['presentation_score'])
    task_consistency     = float(data['task_consistency'])

    row = {
        'role':                      data['role'],
        'tools_used':                tools_used,
        'no_of_projects':            no_of_projects,
        'business_relevance':        business_relevance,
        'presentation_score':        presentation_score,
        'documentation_quality':     documentation_quality,
        'avg_project_score':         avg_project_score,
        'task_consistency':          task_consistency,
        'Tool_Count':                tool_count,
        'Projects_Per_Tool':         no_of_projects / (tool_count + 1),
        'Documentation_Per_Project': documentation_quality / (no_of_projects + 1),
        'Business_Impact_Per_Project': business_relevance / (no_of_projects + 1),
        'Combined_Quality_Score':    (avg_project_score + documentation_quality + presentation_score) / 3,
        'Consistency_Index':         task_consistency * 0.7 + business_relevance * 0.3,
    }
    return pd.DataFrame([row])


def compute_readiness_score(proba: list) -> float:
    """Map class probabilities → 0-100 readiness score.
    Classes are alphabetically sorted by LabelEncoder:
      0 = Almost Ready, 1 = Job Ready, 2 = Needs Improvement
    """
    score = proba[0] * 45 + proba[1] * 90 + proba[2] * 15
    return round(float(score), 1)


# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    total           = len(df_raw)
    job_ready       = int((df_raw['Employability_Label'] == 'Job Ready').sum())
    almost_ready    = int((df_raw['Employability_Label'] == 'Almost Ready').sum())
    needs_improvement = int((df_raw['Employability_Label'] == 'Needs Improvement').sum())
    avg_score       = round(float(df_raw['Readiness_Score'].mean()), 1)

    role_counts = df_raw['role'].value_counts().to_dict()

    top10 = df_raw.nlargest(10, 'Readiness_Score')[
        ['name', 'role', 'Readiness_Score', 'Employability_Label']
    ].to_dict(orient='records')

    bins   = [0, 40, 55, 70, 85, 100]
    labels = ['0–40', '40–55', '55–70', '70–85', '85–100']
    df_tmp = df_raw.copy()
    df_tmp['bucket'] = pd.cut(df_tmp['Readiness_Score'], bins=bins, labels=labels, include_lowest=True)
    score_dist = df_tmp['bucket'].value_counts().sort_index().to_dict()

    return render_template('index.html',
        total=total,
        job_ready=job_ready,
        almost_ready=almost_ready,
        needs_improvement=needs_improvement,
        avg_score=avg_score,
        role_counts=role_counts,
        top10=top10,
        score_dist=score_dist,
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    form_data = {}
    if request.method == 'POST':
        form_data = request.form.to_dict()
        try:
            input_df = engineer_features(form_data)
            proba    = model.predict_proba(input_df)[0]
            pred_cls = model.predict(input_df)[0]
            pred_label = label_encoder.inverse_transform([pred_cls])[0]
            readiness  = compute_readiness_score(list(proba))

            classes    = label_encoder.classes_
            class_probs = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

            result = {
                'name':        form_data.get('name', 'Candidate'),
                'label':       pred_label,
                'score':       readiness,
                'class_probs': class_probs,
            }
        except Exception as e:
            result = {'error': str(e)}

    return render_template('predict.html', result=result, roles=ROLES, form_data=form_data)


@app.route('/leaderboard')
def leaderboard():
    role_filter  = request.args.get('role', 'All')
    label_filter = request.args.get('label', 'All')

    df_f = df_raw.copy()
    if role_filter  != 'All': df_f = df_f[df_f['role'] == role_filter]
    if label_filter != 'All': df_f = df_f[df_f['Employability_Label'] == label_filter]

    df_f = df_f.sort_values('Readiness_Score', ascending=False)
    records = df_f[
        ['name', 'role', 'no_of_projects', 'avg_project_score',
         'Readiness_Score', 'Employability_Label']
    ].head(100).to_dict(orient='records')

    roles  = ['All'] + sorted(df_raw['role'].unique().tolist())
    labels = ['All', 'Job Ready', 'Almost Ready', 'Needs Improvement']

    return render_template('leaderboard.html',
        records=records, roles=roles, labels=labels,
        selected_role=role_filter, selected_label=label_filter,
    )


@app.route('/api/stats')
def api_stats():
    role_avg  = df_raw.groupby('role')['Readiness_Score'].mean().round(1).to_dict()
    label_dist = df_raw['Employability_Label'].value_counts().to_dict()
    return jsonify({'role_avg': role_avg, 'label_dist': label_dist})


if __name__ == '__main__':
    app.run(debug=True)
#updated