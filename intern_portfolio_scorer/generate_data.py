"""
generate_data.py
Generates a synthetic final_intern_readiness_dataset.xlsx
matching the exact schema from the ML pipeline PDF.
Run once before train_model.py.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1555

ROLES = [
    "Full-Stack Developer", "Software Intern", "Frontend Engineer",
    "Backend Engineer", "Data Analyst", "UI/UX Designer",
    "Data Science & Analytics", "MERN Stack Developer",
    "Digital Marketing", "Content Writing",
    "HR Department", "Sales & Marketing",
    "Graphic Design", "Video Editing", "Social Media Manager",
    "General", "DevOps Engineer",
]

TOOLS_BY_ROLE = {
    "Full-Stack Developer":       ["React, Node.js, MongoDB", "Vue.js, Express, PostgreSQL", "Angular, Django, MySQL", "Next.js, FastAPI, Redis"],
    "Software Intern":            ["Python, Git, Linux", "Java, Spring Boot, Maven", "C++, CMake, Docker", "Python, Flask, SQLite"],
    "Frontend Engineer":          ["React, TypeScript, Tailwind", "Vue.js, SCSS, Webpack", "HTML, CSS, JavaScript", "React, Redux, Jest"],
    "Backend Engineer":           ["Node.js, Express, MongoDB", "Python, Django, PostgreSQL", "Java, Spring, Hibernate", "Go, Gin, Redis"],
    "Data Analyst":               ["Python, Pandas, Tableau", "SQL, Power BI, Excel", "R, ggplot2, Shiny", "Python, Matplotlib, Seaborn"],
    "UI/UX Designer":             ["Figma, Adobe XD, Sketch", "Figma, Zeplin, InVision", "Adobe XD, Photoshop, Illustrator"],
    "Data Science & Analytics":   ["Python, Scikit-learn, TensorFlow", "Python, PyTorch, Pandas", "R, Caret, ggplot2"],
    "MERN Stack Developer":       ["MongoDB, Express, React, Node.js", "MongoDB, React, Redux, Node.js"],
    "Digital Marketing":          ["Google Analytics, SEMrush, HubSpot", "Facebook Ads, Google Ads, Canva"],
    "Content Writing":            ["WordPress, Grammarly, SEMrush", "Medium, Notion, Google Docs"],
    "HR Department":              ["Excel, HRMS, Zoho People", "SAP HR, MS Office, Slack"],
    "Sales & Marketing":          ["Salesforce, HubSpot, Excel", "Zoho CRM, Google Sheets, Canva"],
    "Graphic Design":             ["Adobe Illustrator, Photoshop, InDesign", "Canva, Figma, After Effects"],
    "Video Editing":              ["Adobe Premiere Pro, After Effects", "DaVinci Resolve, Final Cut Pro"],
    "Social Media Manager":       ["Hootsuite, Canva, Buffer", "Sprout Social, Adobe Express"],
    "General":                    ["MS Office, Google Workspace, Slack", "Notion, Trello, Zoom"],
    "DevOps Engineer":            ["Docker, Kubernetes, Jenkins", "Terraform, Ansible, AWS"],
}

FIRST_NAMES = [
    "Aarav","Aditi","Akash","Ananya","Arjun","Bhavya","Chirag","Deepika",
    "Divya","Gaurav","Harsha","Ishaan","Jaya","Karan","Kavya","Lakshmi",
    "Manish","Meera","Neeraj","Neha","Nikhil","Pooja","Priya","Rahul",
    "Riya","Rohan","Sakshi","Sanjay","Shreya","Siddharth","Sneha","Suresh",
    "Tanvi","Uday","Varun","Vidya","Vikram","Yash","Zara","Aisha",
    "Amit","Anjali","Arun","Bharat","Chetan","Disha","Esha","Farhan",
    "Geeta","Hemant","Isha","Jayesh","Komal","Lalit","Madhuri","Nandini",
    "Om","Pankaj","Qasim","Rajesh","Sunita","Tarun","Uma","Vijay",
]
LAST_NAMES = [
    "Sharma","Verma","Singh","Kumar","Patel","Gupta","Joshi","Mehta",
    "Nair","Reddy","Iyer","Pillai","Rao","Mishra","Tiwari","Pandey",
    "Chauhan","Yadav","Malhotra","Kapoor","Bose","Das","Ghosh","Sen",
    "Chatterjee","Mukherjee","Banerjee","Roy","Dutta","Sinha",
]

names = [f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}" for _ in range(N)]
roles = np.random.choice(ROLES, size=N, p=[
    0.10, 0.10, 0.10, 0.08, 0.08,
    0.06, 0.06, 0.06,
    0.05, 0.05,
    0.04, 0.04,
    0.04, 0.04, 0.04,
    0.03, 0.03,
])

tools_used = [np.random.choice(TOOLS_BY_ROLE[r]) for r in roles]

# Generate scores with realistic distributions per label
# We'll assign labels first, then generate scores accordingly
label_probs = [0.53, 0.36, 0.11]  # Job Ready, Almost Ready, Needs Improvement
labels_raw = np.random.choice(["Job Ready", "Almost Ready", "Needs Improvement"], size=N, p=label_probs)

no_of_projects = np.where(
    labels_raw == "Job Ready",
    np.random.randint(10, 45, N),
    np.where(labels_raw == "Almost Ready",
             np.random.randint(4, 20, N),
             np.random.randint(1, 8, N))
)

def score_for_label(label, high_mean, high_std, mid_mean, mid_std, low_mean, low_std):
    out = np.zeros(N)
    for i, l in enumerate(labels_raw):
        if l == "Job Ready":
            out[i] = np.clip(np.random.normal(high_mean, high_std), 0, 10)
        elif l == "Almost Ready":
            out[i] = np.clip(np.random.normal(mid_mean, mid_std), 0, 10)
        else:
            out[i] = np.clip(np.random.normal(low_mean, low_std), 0, 10)
    return np.round(out, 2)

business_relevance   = score_for_label(labels_raw, 8.5, 1.0, 5.5, 1.5, 2.5, 1.5)
presentation_score   = score_for_label(labels_raw, 8.0, 1.2, 5.0, 1.5, 2.0, 1.5)
documentation_quality= score_for_label(labels_raw, 8.2, 1.0, 5.2, 1.5, 2.2, 1.5)
avg_project_score    = score_for_label(labels_raw, 8.5, 0.8, 5.5, 1.2, 2.5, 1.2)
task_consistency     = score_for_label(labels_raw, 8.8, 0.8, 5.8, 1.3, 2.8, 1.3)

# Compute readiness score (0-100) based on weighted features
def compute_readiness(br, ps, dq, aps, tc, nop):
    raw = (aps * 0.25 + dq * 0.15 + br * 0.15 + ps * 0.10 + tc * 0.10) / 10 * 100
    project_bonus = np.clip(nop / 50 * 20, 0, 20)
    return np.round(np.clip(raw + project_bonus, 0, 100), 2)

readiness_scores = compute_readiness(
    business_relevance, presentation_score, documentation_quality,
    avg_project_score, task_consistency, no_of_projects
)

# Re-assign labels based on computed score for consistency
employability_label = np.where(
    readiness_scores >= 70, "Job Ready",
    np.where(readiness_scores >= 45, "Almost Ready", "Needs Improvement")
)

df = pd.DataFrame({
    "name": names,
    "role": roles,
    "no_of_projects": no_of_projects,
    "business_relevance": business_relevance,
    "presentation_score": presentation_score,
    "documentation_quality": documentation_quality,
    "tools_used": tools_used,
    "avg_project_score": avg_project_score,
    "task_consistency": task_consistency,
    "Readiness_Score": readiness_scores,
    "Employability_Label": employability_label,
})

df.to_excel("final_intern_readiness_dataset.xlsx", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df["Employability_Label"].value_counts())
print(df.head(3))
