"""
app.py  —  Graphura Intern Portfolio Readiness Scorer
Flask web application with Logistic Regression backend.
"""
import io
import os
import re
import difflib
import pandas as pd
import numpy as np
import joblib
import pdfplumber
from flask import Flask, render_template, request, jsonify

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None

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
MASTER_SKILL_DICTIONARIES = {
    'Languages': ['python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'ruby', 'scala', 'go', 'rust', 'sql', 'bash', 'matlab'],
    'Frameworks': ['react', 'vue', 'angular', 'next.js', 'nextjs', 'svelte', 'django', 'flask', 'fastapi', 'spring', 'express', 'ruby on rails', 'asp.net', 'tensorflow', 'keras', 'pytorch', 'spark'],
    'Tools': ['git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'jira', 'figma', 'photoshop', 'illustrator', 'postman', 'vscode', 'powerbi', 'tableau', 'notion', 'confluence'],
    'Databases': ['mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'oracle', 'sql server', 'cassandra', 'dynamodb', 'cockroachdb'],
    'Platforms': ['aws', 'azure', 'gcp', 'google cloud', 'firebase', 'heroku', 'digitalocean', 'vercel', 'netlify'],
    'AI/ML': ['machine learning', 'deep learning', 'artificial intelligence', 'data science', 'nlp', 'computer vision', 'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm'],
    'Cloud': ['aws', 'azure', 'gcp', 'google cloud', 'firebase', 'heroku'],
    'DevOps': ['docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'circleci', 'terraform', 'ansible'],
    'Data Analytics': ['power bi', 'tableau', 'excel', 'pandas', 'numpy', 'sql', 'matplotlib', 'seaborn', 'lookml'],
    'Frontend': ['html', 'css', 'javascript', 'typescript', 'react', 'vue', 'angular', 'bootstrap', 'tailwind', 'next.js'],
    'Backend': ['node', 'express', 'django', 'flask', 'fastapi', 'spring', 'ruby on rails', 'asp.net', 'java', 'python'],
    'Mobile': ['react native', 'flutter', 'swift', 'kotlin', 'android', 'ios'],
    'Cybersecurity': ['security', 'penetration testing', 'vulnerability assessment', 'network security', 'encryption', 'identity access management'],
    'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'adaptability', 'time management', 'collaboration', 'creativity', 'decision making'],
}
SKILL_CATEGORIES = MASTER_SKILL_DICTIONARIES.copy()

SECTION_HEADERS = {
    'skills': ['skills', 'technical skills', 'core competencies', 'key skills', 'skillset'],
    'experience': ['experience', 'work experience', 'professional experience', 'internships', 'projects'],
    'projects': ['projects', 'academic projects', 'portfolio', 'project experience'],
    'education': ['education', 'academic details', 'qualifications'],
    'certifications': ['certifications', 'certificates', 'training'],
    'achievements': ['achievements', 'awards', 'honors'],
    'leadership': ['leadership', 'positions of responsibility', 'activities'],
    'summary': ['summary', 'profile', 'about me', 'career objective'],
}

ACTION_VERBS = [
    'designed', 'built', 'delivered', 'implemented', 'optimized', 'launched', 'improved',
    'analyzed', 'automated', 'presented', 'collaborated', 'mentored', 'led', 'owned'
]

MODEL_FILES = {
    'random_forest': 'random_forest_model.pkl',
    'xgboost': 'xgb_model.pkl',
    'lightgbm': 'lgbm_model.pkl',
}

FRAMEWORK_WEIGHTS = {
    'project_quality': 25,
    'documentation_quality': 15,
    'presentation_quality': 10,
    'business_impact': 15,
    'technical_skills': 10,
    'consistency_score': 10,
    'project_diversity': 5,
    'commit_frequency': 5,
    'readme_quality': 3,
    'language_diversity': 2,
}

PROJECT_TYPE_KEYWORDS = [
    'api', 'web', 'mobile', 'dashboard', 'automation', 'machine learning', 'ml', 'analytics',
    'e-commerce', 'social', 'chatbot', 'iot', 'blockchain', 'game', 'security', 'cloud', 'data',
]

README_KEYWORDS = ['readme', 'documentation', 'docs', 'wiki', 'guide', 'installation']
COMMIT_KEYWORDS = ['commit', 'pushed', 'git', 'repository', 'branch', 'merge', 'pull request', 'pull request']
RESUME_KEYWORDS = [
    'algorithm', 'optimization', 'deployment', 'performance', 'automation', 'analysis',
    'machine learning', 'data', 'api', 'cloud', 'docker', 'git', 'database',
    'frontend', 'backend', 'testing', 'security', 'design', 'documentation',
    'communication', 'team', 'project', 'internship', 'research', 'product'
]


def soft_vote_models():
    models = {'logistic': model}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                continue
    return models

ENSEMBLE_MODELS = soft_vote_models()


def engineer_features(data: dict) -> pd.DataFrame:
    if data.get('parsed_features'):
        ml_input = data['parsed_features']
    else:
        skills = extract_skills(' '.join([data.get('tools_used', ''), data.get('role', '')]))
        projects = extract_projects(data.get('tools_used', ''))
        experience = extract_experience(data.get('tools_used', ''))
        ats_details = {'format_quality': 5}
        parsed = {
            'role': data.get('role', 'General'),
            'skills': skills,
            'projects': projects,
            'experience': experience,
            'ats_details': ats_details,
        }
        ml_input = generate_ml_features(parsed)

    return pd.DataFrame([ml_input])


def compute_readiness_score(proba: list) -> float:
    """Map class probabilities → 0-100 readiness score.
    Classes are alphabetically sorted by LabelEncoder:
      0 = Almost Ready, 1 = Job Ready, 2 = Needs Improvement
    """
    score = proba[0] * 45 + proba[1] * 90 + proba[2] * 15
    return round(float(score), 1)


def _normalize_score(value: float, max_value: float = 10.0) -> float:
    return max(0.0, min(100.0, (float(value) / float(max_value)) * 100.0))


def compute_framework_breakdown(parsed: dict) -> tuple[dict, float]:
    text = parsed.get('raw_text', '').lower()
    ml = parsed.get('ml_features', {})
    skills = parsed['skills'].get('all_skills', [])

    project_quality = _normalize_score(ml.get('avg_project_score', 5))
    documentation_quality = _normalize_score(ml.get('documentation_quality', 5))
    presentation_quality = _normalize_score(ml.get('presentation_score', 5))
    business_impact = _normalize_score(ml.get('business_relevance', 5))
    technical_skills = min(100.0, len(skills) * 8.0)
    consistency_score = _normalize_score(ml.get('task_consistency', 5))

    project_diversity = min(100.0, len({
        kw for kw in PROJECT_TYPE_KEYWORDS if kw in text
    }) * 20.0)

    commit_signal = 0
    if parsed['links'].get('github'):
        commit_signal += 50.0
    commit_signal += min(50.0, len([kw for kw in COMMIT_KEYWORDS if kw in text]) * 10.0)
    commit_frequency = min(100.0, commit_signal)

    readme_quality = 100.0 if any(term in text for term in README_KEYWORDS) else 35.0
    if 'readme' in text or 'documentation' in text:
        readme_quality = 100.0
    elif 'docs' in text or 'guide' in text or 'wiki' in text:
        readme_quality = 70.0

    language_diversity = min(100.0, len([lang for lang in SKILL_CATEGORIES['Languages'] if re.search(r'\b' + re.escape(lang) + r'\b', text)]) * 12.5)

    breakdown = {
        'project_quality': round(project_quality, 1),
        'documentation_quality': round(documentation_quality, 1),
        'presentation_quality': round(presentation_quality, 1),
        'business_impact': round(business_impact, 1),
        'technical_skills': round(technical_skills, 1),
        'consistency_score': round(consistency_score, 1),
        'project_diversity': round(project_diversity, 1),
        'commit_frequency': round(commit_frequency, 1),
        'readme_quality': round(readme_quality, 1),
        'language_diversity': round(language_diversity, 1),
    }

    weighted = sum(
        breakdown[key] * (FRAMEWORK_WEIGHTS[key] / 100.0)
        for key in FRAMEWORK_WEIGHTS
    )
    return breakdown, round(weighted, 1)


def ensemble_predict(features_df: pd.DataFrame) -> list:
    probabilities = []
    for mdl in ENSEMBLE_MODELS.values():
        if hasattr(mdl, 'predict_proba'):
            probabilities.append(mdl.predict_proba(features_df)[0])
    if not probabilities:
        raise RuntimeError('No valid models available for prediction.')
    return np.mean(np.vstack(probabilities), axis=0).tolist()


def build_insights(parsed: dict, proba: list) -> dict:
    label = label_encoder.inverse_transform([np.argmax(proba)])[0]
    missing_skills = []
    if 'docker' not in parsed['skills']['all_skills']:
        missing_skills.append('Docker')
    if not parsed['links']['github']:
        missing_skills.append('GitHub portfolio')
    if 'aws' not in parsed['skills']['all_skills'] and 'azure' not in parsed['skills']['all_skills']:
        missing_skills.append('Cloud deployment skills')
    strengths = []
    weaknesses = []
    if parsed['projects']['deployment_count'] > 0:
        strengths.append('Deployed projects highlighted')
    else:
        weaknesses.append('No deployment link detected')
    if parsed['experience']['leadership'] > 0:
        strengths.append('Leadership or coordination experience')
    if parsed['experience']['internships'] == 0:
        weaknesses.append('Internship or work experience references')
    suggestions = [
        'Add measurable achievements and metrics to each project.',
        'Include GitHub or live deployment links for technical credibility.',
        'List action verbs and targeted keywords in experience descriptions.',
    ]
    if not parsed['education']['degree']:
        suggestions.append('Clarify academic credentials and graduation year.')
    radar = {
        'frontend': min(10, len(parsed['skills']['Frontend']) * 2),
        'backend': min(10, len(parsed['skills']['Backend']) * 2),
        'ml': min(10, len(parsed['skills']['AI/ML']) * 2),
        'deployment': min(10, parsed['projects']['deployment_count'] * 2),
        'communication': min(10, 3 + parsed['experience']['leadership']),
    }
    framework_breakdown, framework_score = compute_framework_breakdown(parsed)
    return {
        'label': label,
        'confidence': round(float(max(proba) * 100), 1),
        'readiness': compute_readiness_score(proba),
        'framework_score': framework_score,
        'framework_breakdown': framework_breakdown,
        'ats_score': parsed['ats_score'],
        'strengths': strengths,
        'weaknesses': weaknesses,
        'missing_skills': missing_skills,
        'suggestions': suggestions,
        'skill_radar': radar,
    }


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def extract_resume_text(file_stream) -> str:
    try:
        with pdfplumber.open(file_stream) as pdf:
            pages = [page.extract_text() or '' for page in pdf.pages]
        return '\n'.join(pages)
    except Exception:
        return ''


def extract_sections(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = {'header': []}
    current = 'header'
    normalized_headers = {header: section for section, values in SECTION_HEADERS.items() for header in values}
    for line in lines:
        key = line.lower().strip(':').strip()
        if key in normalized_headers:
            current = normalized_headers[key]
            sections[current] = []
            continue
        sections.setdefault(current, []).append(line)
    return {section: '\n'.join(lines).strip() for section, lines in sections.items()}


def extract_personal_info(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone_match = re.search(r'(?:\+?\d{1,3}[\s-])?\d{10,15}', text)
    location_match = re.search(r'\b(hyderabad|bangalore|mumbai|delhi|pune|chennai|kolkata|remote|india|bangalore|gurgaon|noida|bengaluru)\b', text, re.I)
    name = None
    if lines:
        first = lines[0]
        if '@' not in first and not re.search(r'\b\d{2,4}\b', first):
            name = first
    if not name and lines:
        name = lines[0]
    return {
        'name': name or 'Candidate',
        'email': email_match.group(0) if email_match else None,
        'phone': phone_match.group(0) if phone_match else None,
        'location': location_match.group(0).title() if location_match else None,
    }


def normalize_term(value: str) -> str:
    return re.sub(r'[^a-z0-9+.#\- ]', '', value.lower()).strip()


def fuzzy_find_terms(text: str, terms: list[str], cutoff: float = 0.88) -> list[str]:
    normalized_text = text.lower()
    found = set()
    words = set(re.findall(r'[a-z0-9+.#\-]+', normalized_text))
    for term in terms:
        term_norm = normalize_term(term)
        if not term_norm:
            continue
        if re.search(r'\b' + re.escape(term_norm) + r'\b', normalized_text):
            found.add(term)
            continue
        if ' ' in term_norm and term_norm in normalized_text:
            found.add(term)
            continue
        match = difflib.get_close_matches(term_norm, words, n=1, cutoff=cutoff)
        if match:
            found.add(term)
    return sorted(found)


def extract_skills(text: str) -> dict:
    sections = extract_sections(text)
    search_text = ' '.join([sections.get('skills', ''), sections.get('technical skills', ''), sections.get('header', ''), sections.get('summary', ''), text])
    skills = {'all_skills': []}
    skills_by_category = {}
    for category, terms in MASTER_SKILL_DICTIONARIES.items():
        matches = fuzzy_find_terms(search_text, terms)
        if category == 'Soft Skills':
            matches = [term for term in matches if term.lower() in search_text]
        skills_by_category[category] = sorted(set(matches))
        skills['all_skills'].extend(matches)
    skills['all_skills'] = sorted(set(skills['all_skills']))
    skills.update(skills_by_category)
    return skills


def extract_projects(text: str) -> dict:
    sections = extract_sections(text)
    project_text = sections.get('projects', text)
    lines = [line.strip() for line in project_text.splitlines() if line.strip()]
    project_titles = []
    project_tech = set()
    for line in lines:
        if re.search(r'\b(project|project name|built|developed|engineered)\b', line, re.I):
            project_titles.append(line)
        elif re.search(r'\busing\b|\bwith\b|\bon\b', line, re.I) and len(line.split()) > 4:
            project_tech.update(re.findall(r'[A-Za-z#+\.\-]+', line.lower()))
    if not project_titles:
        project_titles = [line for line in lines if len(line) > 40][:3]
    deployed = len(re.findall(r'\bdeploy(ed|ment)?\b|\bhosted\b|\blive\b|\bproduction\b|\baws\b|\bazure\b|\bgcp\b|\bvercel\b|\bnetlify\b', project_text.lower()))
    complexity = min(10, max(1, len(re.findall(r'\b(api|microservice|ml|machine learning|automation|system design|distributed|data pipeline|analytics|dashboard|security)\b', project_text.lower())) + 2))
    github_links = len(re.findall(r'https?://(?:www\.)?github\.com/[\w\-]+', text))
    raw_project_count = max(1, len(project_titles))
    return {
        'project_count': raw_project_count,
        'deployment_count': min(5, deployed),
        'complexity_score': complexity,
        'github_links': github_links,
        'project_titles': project_titles,
        'technologies': sorted({term for term in project_tech if len(term) > 1}),
    }


def extract_experience(text: str) -> dict:
    sections = extract_sections(text)
    exp_text = sections.get('experience', text)
    lower = exp_text.lower()
    internships = len(re.findall(r'\bintern(ship)?\b', lower))
    freelance = len(re.findall(r'\bfreelance\b|\bcontract\b', lower))
    open_source = len(re.findall(r'\bopen source\b|\bcontribution\b|\bgithub\b', lower))
    leadership = len(re.findall(r'\blead(er|ership)?\b|\bmanaged\b|\bcoordinat(ed|or)\b|\bmentor(ed)?\b', lower))
    positions = re.findall(r'\b(?:as|at|with)\s+([A-Z][A-Za-z0-9 &\-]+)', exp_text)
    durations = re.findall(r'\b(\d+\s+(?:months?|years?))\b', exp_text, re.I)
    experience_score = min(10, 2 + internships + freelance + int(bool(durations)) + min(3, leadership))
    return {
        'internships': internships,
        'freelance': freelance,
        'open_source': open_source,
        'leadership': leadership,
        'company_count': len(sorted(set(positions))),
        'duration_phrases': durations,
        'experience_score': experience_score,
    }


def extract_links(text: str) -> dict:
    text_lower = text.lower()
    github = re.search(r'https?://(?:www\.)?github\.com/[\w\-]+', text_lower)
    linkedin = re.search(r'https?://(?:www\.)?linkedin\.com/in/[\w\-]+', text_lower)
    portfolio = re.search(r'https?://(?:www\.)?(?:www\.)?(?:portfolio|behance|dribbble|medium|dev\.to|codepen|codesandbox)\.[^\s]+', text_lower)
    website = re.search(r'https?://(?:www\.)?(?!github|linkedin|leetcode|hackerrank|portfolio|behance|dribbble|medium|dev\.to|codepen|codesandbox)[\w\.-]+\.[a-z]{2,6}/?', text_lower)
    leetcode = re.search(r'https?://(?:www\.)?leetcode\.com/[\w\-]+', text_lower)
    hackerrank = re.search(r'https?://(?:www\.)?hackerrank\.com/[\w\-]+', text_lower)
    return {
        'github': github.group(0) if github else None,
        'linkedin': linkedin.group(0) if linkedin else None,
        'portfolio': portfolio.group(0) if portfolio else None,
        'website': website.group(0) if website else None,
        'leetcode': leetcode.group(0) if leetcode else None,
        'hackerrank': hackerrank.group(0) if hackerrank else None,
    }


def extract_education(text: str) -> dict:
    lower = text.lower()
    degree = None
    for keyword in ['bachelor', 'master', 'm.s.', 'mtech', 'b.tech', 'ph.d', 'phd', 'mba', 'bba', 'bca', 'msc', 'bsc']:
        if keyword in lower:
            degree = keyword.title().replace('M.S.', 'MS').replace('B.Tech', 'B.Tech').replace('Ph.D', 'PhD')
            break
    grad_year = None
    year_match = re.search(r'20\d{2}|19\d{2}', text)
    if year_match:
        grad_year = year_match.group(0)
    institution = None
    sections = extract_sections(text)
    education_text = sections.get('education', text)
    lines = [line.strip() for line in education_text.splitlines() if line.strip()]
    if lines:
        institution = lines[0]
    return {
        'degree': degree,
        'institution': institution,
        'graduation_year': grad_year,
    }


def extract_certifications(text: str) -> list:
    matches = re.findall(r'(?:Certified|Certification|Certificate in|Certified in|Certified by)\s+[A-Za-z0-9 &/\-]+', text, re.I)
    return sorted(set(match.strip() for match in matches))


def calculate_ats_score(parsed: dict) -> tuple[int, dict]:
    text = parsed['raw_text'].lower()
    skills_count = len(parsed['skills']['all_skills'])
    project_score = min(100, parsed['projects']['project_count'] * 12 + parsed['projects']['deployment_count'] * 15 + parsed['projects']['complexity_score'] * 4)
    experience_score = min(100, parsed['experience']['experience_score'] * 10 + parsed['experience']['leadership'] * 5 + parsed['experience']['open_source'] * 3)
    formatting_score = min(100, 45 + parsed['ats_details'].get('format_quality', 5) * 5 + int(bool(re.search(r'\bexperience\b|\beducation\b|\bprojects?\b|\bskills?\b', text))) * 10)
    keyword_hits = len([kw for kw in RESUME_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text)])
    keywords_score = min(100, keyword_hits * 8 + len([verb for verb in ACTION_VERBS if re.search(r'\b' + verb + r'\b', text)]) * 3)
    links_score = 100 if parsed['links']['github'] and parsed['links']['linkedin'] else 70 if parsed['links']['github'] or parsed['links']['linkedin'] else 20

    categories = {
        'skills_section': round(min(100, skills_count * 10), 1),
        'projects': round(project_score, 1),
        'experience': round(experience_score, 1),
        'ats_formatting': round(formatting_score, 1),
        'keywords': round(keywords_score, 1),
        'links': round(links_score, 1),
    }

    weighted = (
        categories['skills_section'] * 0.20 +
        categories['projects'] * 0.20 +
        categories['experience'] * 0.20 +
        categories['ats_formatting'] * 0.15 +
        categories['keywords'] * 0.15 +
        categories['links'] * 0.10
    )

    return min(100, int(round(weighted, 0))), categories


def generate_ml_features(parsed: dict) -> dict:
    tools_used = ', '.join(parsed['skills']['all_skills'][:6]) or 'Python'
    project_count = max(1, parsed['projects']['project_count'])
    business_relevance = min(10, max(4, 4 + len(parsed['skills']['Databases']) + len(parsed['skills']['Cloud'])))
    documentation_quality = min(10, max(4, 4 + parsed['ats_details']['format_quality']))
    avg_project_score = min(10, max(4, parsed['projects']['complexity_score']))
    presentation_score = min(10, max(4, 5 + len(parsed['skills']['Frontend'])//2))
    task_consistency = min(10, max(4, 4 + parsed['experience']['experience_score']//2))
    tool_count = len(tools_used.split(','))
    return {
        'role': parsed.get('role', 'General'),
        'tools_used': tools_used,
        'no_of_projects': float(project_count),
        'business_relevance': float(business_relevance),
        'presentation_score': float(presentation_score),
        'documentation_quality': float(documentation_quality),
        'avg_project_score': float(avg_project_score),
        'task_consistency': float(task_consistency),
        'Tool_Count': float(tool_count),
        'Projects_Per_Tool': float(project_count) / (tool_count + 1),
        'Documentation_Per_Project': float(documentation_quality) / (project_count + 1),
        'Business_Impact_Per_Project': float(business_relevance) / (project_count + 1),
        'Combined_Quality_Score': float((avg_project_score + documentation_quality + presentation_score) / 3),
        'Consistency_Index': float(task_consistency * 0.7 + business_relevance * 0.3),
    }


def build_resume_stub(data: dict) -> dict:
    skills = extract_skills(' '.join([data.get('tools_used', ''), data.get('role', '' )]))
    projects = extract_projects(data.get('tools_used', ''))
    experience = extract_experience(data.get('tools_used', ''))
    education = extract_education(' '.join([data.get('role', ''), data.get('tools_used', '')]))
    raw_text = ' '.join([data.get('role', ''), data.get('tools_used', '')])
    parsed = {
        'skills': skills,
        'links': extract_links(raw_text),
        'projects': projects,
        'experience': experience,
        'education': education,
        'personal': {
            'name': data.get('name', 'Candidate'),
            'email': None,
            'phone': None,
            'location': None,
        },
        'raw_text': raw_text,
        'ats_details': {'format_quality': 5},
    }
    ats_score, ats_analysis = calculate_ats_score(parsed)
    parsed['ats_score'] = ats_score
    parsed['ats_analysis'] = ats_analysis
    parsed['feedback'] = generate_resume_feedback(parsed)
    parsed['ml_features'] = generate_ml_features(parsed)
    return parsed
    parsed['ats_details'] = {'format_quality': 5}
    parsed['ml_features'] = generate_ml_features(parsed)
    return parsed


def extract_resume_data(raw_text: str) -> dict:
    text = normalize_text(raw_text)
    resume = extract_personal_info(text)
    links = extract_links(text)
    skills = extract_skills(text)
    projects = extract_projects(text)
    experience = extract_experience(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    parsed = {
        'raw_text': text,
        'role': resume['name'] or 'General',
        'personal': resume,
        'links': links,
        'skills': skills,
        'projects': projects,
        'experience': experience,
        'education': education,
        'certifications': certifications,
        'ats_score': 0,
        'ats_details': {},
    }
    parsed['ats_details'] = {
        'format_quality': min(10, 4 + len(re.findall(r'\bresume\b|\bcv\b|\bbeen\b', text)) // 5),
        'sections': 1 + int(bool(skills['all_skills'])) + int(bool(experience['internships'] or experience.get('open_source', 0))) + int(bool(education['degree'])),
    }
    parsed['ats_score'], parsed['ats_analysis'] = calculate_ats_score(parsed)
    parsed['feedback'] = generate_resume_feedback(parsed)
    parsed['ml_features'] = generate_ml_features(parsed)
    parsed['summary'] = {
        'skill_count': len(skills['all_skills']),
        'project_count': projects['project_count'],
        'internship_count': experience['internships'],
        'education_level': education['degree'],
        'github': bool(links['github']),
        'linkedin': bool(links['linkedin']),
    }
    return parsed


def generate_resume_feedback(parsed: dict) -> dict:
    text = parsed['raw_text'].lower()
    missing = []
    suggestions = []
    strengths = []
    weaknesses = []

    if not parsed['personal'].get('email') or not parsed['personal'].get('phone'):
        missing.append('Contact information')
        weaknesses.append('Missing phone number or email address in the header.')
    else:
        strengths.append('Clear contact section found.')

    if not parsed['links']['github']:
        missing.append('GitHub link')
        weaknesses.append('No GitHub portfolio detected.')
    else:
        strengths.append('GitHub presence detected.')

    if not parsed['links']['linkedin']:
        missing.append('LinkedIn profile')
    else:
        strengths.append('LinkedIn profile detected.')

    if parsed['projects']['deployment_count'] == 0:
        missing.append('Deployed or live project links')
        weaknesses.append('No deployment or hosting evidence found for projects.')
    else:
        strengths.append('Deployed project experience found.')

    if parsed['experience']['internships'] == 0 and parsed['experience']['open_source'] == 0:
        missing.append('Internship or open source experience')
        weaknesses.append('Limited concrete experience entries.')
    else:
        strengths.append('Relevant experience or contributions detected.')

    action_verbs = len([verb for verb in ACTION_VERBS if re.search(r'\b' + verb + r'\b', text)])
    if action_verbs < 3:
        weaknesses.append('Consider adding more action-oriented verbs in project descriptions.')
    else:
        strengths.append('Strong use of action verbs in accomplishments.')

    if not parsed['certifications']:
        missing.append('Certifications or training')
    else:
        strengths.append('Professional certifications detected.')

    if parsed['projects']['project_count'] < 2:
        missing.append('More project examples')
        suggestions.append('Add at least two strong projects with measurable outcomes.')
    else:
        strengths.append('Multiple projects referenced.')

    if parsed['education']['degree'] is None:
        missing.append('Education details')
        suggestions.append('Add degree or academic qualification information clearly.')

    suggestions.extend([
        'Add quantified achievements for every project and experience bullet.',
        'Include deployment links or GitHub references for technical work.',
        'Use clear section headings and bullet points for strong ATS parsing.',
        'Add technical keywords relevant to your target role.',
    ])

    return {
        'strengths': sorted(set(strengths)),
        'weaknesses': sorted(set(weaknesses)),
        'suggestions': sorted(set(suggestions)),
        'missing_items': missing,
        'keyword_hits': len([kw for kw in RESUME_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text)]),
    }


def get_request_data():
    if request.content_type and 'application/json' in request.content_type:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


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


@app.route('/api/parse_resume', methods=['POST'])
def api_parse_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume uploaded.'}), 400

    uploaded_file = request.files['resume']
    if uploaded_file.filename == '':
        return jsonify({'error': 'Empty file name.'}), 400

    raw_text = extract_resume_text(uploaded_file.stream)
    if not raw_text.strip():
        return jsonify({'error': 'Could not extract text from PDF. Please upload a valid resume.'}), 400

    parsed = extract_resume_data(raw_text)
    parsed_response = {
        'personal': parsed['personal'],
        'links': parsed['links'],
        'skills': parsed['skills'],
        'projects': parsed['projects'],
        'experience': parsed['experience'],
        'education': parsed['education'],
        'certifications': parsed['certifications'],
        'ats_score': parsed['ats_score'],
        'ats_analysis': parsed['ats_analysis'],
        'feedback': parsed['feedback'],
        'summary': parsed['summary'],
        'parsed_features': parsed['ml_features'],
        'ml_features': parsed['ml_features'],
        'resume_details': {
            'name': parsed['personal']['name'],
            'email': parsed['personal']['email'],
            'phone': parsed['personal']['phone'],
            'location': parsed['personal']['location'],
            'skills': parsed['skills']['all_skills'],
            'projects': parsed['projects']['project_count'],
            'experience': parsed['experience']['experience_score'],
            'education': parsed['education']['degree'],
            'certifications': parsed['certifications'],
            'github': parsed['links']['github'],
            'linkedin': parsed['links']['linkedin'],
        },
        'debug': {
            'length': len(raw_text),
            'section_headers': list(extract_sections(raw_text).keys()),
        },
    }
    return jsonify(parsed_response)


@app.route('/api/sample_resume', methods=['GET'])
def api_sample_resume():
    sample_text = '''
Rahul Sharma
Email: rahul@example.com | Phone: +91 98765 43210 | Bangalore, India

Full-Stack Developer with internship experience and multiple deployed projects.

Skills: Python, React, Django, PostgreSQL, Docker, AWS, Git, REST APIs, SQL, Tableau.

Experience
- Internship at TechLabs: built an AI-enabled task manager with React and Django.
- Freelance developer for a fintech dashboard using AWS, Docker and PostgreSQL.

Projects
- CampusConnect: deployed social networking app on AWS.
- ResumeInsight: automated resume parsing with PDF and NLP.

Education
Bachelor of Technology in Computer Science, XYZ University, 2024.
    '''
    parsed = extract_resume_data(sample_text)
    parsed['parsed_features'] = parsed['ml_features']
    return jsonify(parsed)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = get_request_data()
    if not data:
        return jsonify({'error': 'No data provided.'}), 400

    try:
        input_df = engineer_features(data)
        proba = ensemble_predict(input_df)
        pred_cls = np.argmax(proba)
        pred_label = label_encoder.inverse_transform([pred_cls])[0]
        class_probs = {cls: round(float(p) * 100, 1) for cls, p in zip(label_encoder.classes_, proba)}

        parsed = data.get('parsed') or build_resume_stub(data)
        insights = build_insights(parsed, proba)
        feedback = parsed.get('feedback') or generate_resume_feedback(parsed)

        return jsonify({
            'name': data.get('name', parsed.get('personal', {}).get('name', 'Candidate')),
            'label': insights['label'],
            'score': insights['readiness'],
            'framework_score': insights['framework_score'],
            'framework_breakdown': insights['framework_breakdown'],
            'confidence': insights['confidence'],
            'class_probs': class_probs,
            'ats_score': parsed.get('ats_score', insights['ats_score']),
            'ats_analysis': parsed.get('ats_analysis', {}),
            'skill_radar': insights['skill_radar'],
            'strengths': feedback['strengths'],
            'weaknesses': feedback['weaknesses'],
            'missing_skills': feedback['missing_items'],
            'suggestions': feedback['suggestions'],
            'resume_summary': parsed.get('summary', {}),
            'resume_details': {
                'name': parsed['personal'].get('name'),
                'email': parsed['personal'].get('email'),
                'phone': parsed['personal'].get('phone'),
                'location': parsed['personal'].get('location'),
                'skills': parsed['skills']['all_skills'],
                'projects': parsed['projects']['project_count'],
                'experience': parsed['experience']['experience_score'],
                'education': parsed['education']['degree'],
                'certifications': parsed.get('certifications', []),
                'github': parsed['links'].get('github'),
                'linkedin': parsed['links'].get('linkedin'),
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


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