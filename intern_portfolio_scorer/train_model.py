"""
train_model.py
Trains Logistic Regression on the intern readiness dataset.
Saves best_intern_readiness_model.pkl and label_encoder.pkl.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# ── 1. Load dataset ──────────────────────────────────────────
df = pd.read_excel("final_intern_readiness_dataset.xlsx")
df = df.copy()

# ── 2. Handle missing values ─────────────────────────────────
for col in df.columns:
    if df[col].dtype.kind in ('O', 'S', 'U'):   # object / string types
        df[col] = df[col].fillna('Unknown')
    elif df[col].dtype.kind in ('i', 'f', 'u'): # numeric types
        df[col] = df[col].fillna(df[col].median())

# ── 3. Feature engineering ───────────────────────────────────
df['Tool_Count'] = df['tools_used'].apply(lambda x: len(str(x).split(',')))
df['Projects_Per_Tool'] = df['no_of_projects'] / (df['Tool_Count'] + 1)
df['Documentation_Per_Project'] = df['documentation_quality'] / (df['no_of_projects'] + 1)
df['Business_Impact_Per_Project'] = df['business_relevance'] / (df['no_of_projects'] + 1)
df['Combined_Quality_Score'] = (
    df['avg_project_score'] + df['documentation_quality'] + df['presentation_score']
) / 3
df['Consistency_Index'] = df['task_consistency'] * 0.7 + df['business_relevance'] * 0.3

# ── 4. Encode target ─────────────────────────────────────────
label_encoder = LabelEncoder()
df['Target_Encoded'] = label_encoder.fit_transform(df['Employability_Label'])
print("Classes:", label_encoder.classes_)

# ── 5. Features / target ─────────────────────────────────────
remove_cols = ['Readiness_Score', 'Employability_Label', 'Target_Encoded', 'name']
X = df.drop(columns=[c for c in remove_cols if c in df.columns])
y = df['Target_Encoded']

numeric_features    = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# ── 6. Preprocessing pipeline ────────────────────────────────
numeric_pipeline     = Pipeline(steps=[('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
])

# ── 7. Logistic Regression model ─────────────────────────────
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        penalty='l2',
        C=1.0,
        random_state=42,
    ))
])

# ── 8. Train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# ── 9. Evaluate ──────────────────────────────────────────────
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

print(f"\nTrain Accuracy : {accuracy_score(y_train, train_pred):.4f}")
print(f"Test  Accuracy : {accuracy_score(y_test,  test_pred):.4f}")
print(f"Balanced Acc   : {balanced_accuracy_score(y_test, test_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

# ── 10. Save ─────────────────────────────────────────────────
joblib.dump(model,         'best_intern_readiness_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\nModel and encoder saved successfully.")
