# Intern Portfolio Readiness Scorer

### Ingestion Production Repository
https://github.com/Shashikala-11/intern-portfolio-readiness-scorer

---

## Project Overview

Evaluating technical portfolios manually is slow, inconsistent, and highly subjective. At Graphura Private Limited, assessing an intern's readiness for permanent industry placement requires an objective system that analyzes real, public code footprints rather than static, self-declared resumes. 

The Intern Portfolio Readiness Scorer is an automated microservice data pipeline that connects directly to version-control APIs, parses raw profile counts, transforms them using non-linear mathematical models into a standardized 1-to-10 metric matrix, and uses an optimized Logistic Regression classifier to predict overall employability. The system instantly outputs a specialized continuous Readiness Score between 0 and 100, alongside a categorical placement group: Job Ready, Almost Ready, or Needs Improvement.

---

## Core System Architecture


[User Ingestion Link] ➔ [API Data Harvester] ➔ [Feature Math Matrix] ➔ [Pickle Model Scoring] ➔ [UI & PBI Dashboard]


### 1. Data Ingestion Core
Connects to public endpoints using token-authenticated headers to fetch live profile configurations.

### 2. Feature Logic Matrix
Computes weighted, log-scaled values to eliminate popularity bias and isolate true engineering depth.

### 3. Machine Learning Array
Runs serialized model vectors to evaluate data splits and map classification targets.

### 4. Presentation Edge
Feeds data to interactive corporate analytics dashboards and live web forms for macro-level workforce visibility.

---

## Technical Stack

| Infrastructure Track | Technologies Employed |
| :--- | :--- |
| Data Engineering | Python, Pandas, NumPy, JSON API Parsers |
| Predictive Modeling | Scikit-Learn, Pickle, Multi-class Logit Vectors |
| Web Microservice | Flask Web Engine, HTML5 Ingestion Layout, CSS |
| Business Analytics | Power BI Professional Ecosystem |

---

## Feature Engineering and Rubric Formulae

Raw metadata arrays are converted into normalized 1-to-10 criteria scores using deterministic equations to ensure absolute evaluation objectivity:

### Business Relevance Score
Forks indicate real structural reuse by other developers and are given double the weight of stars. The total is processed with a natural logarithm to prevent viral repositories from skewing prediction boundaries:

Business Relevance Score = clip(ln(1 + Total Forks * 2 + Total Stars) * 1.1 + 1.0, 1.0, 10.0)

### Presentation Tracking
Measures user profile organization. External blog links provide an immediate baseline boost, while active web deployments via GitHub Pages add incremental points up to a maximum cap of 10.0.

### Documentation Proxy
Uses average repository disk footprints as a fast structural indicator of layout completeness, repository health, and onboarding code depth.

---

## Predictive Modeling Performance

The underlying data pipeline was trained on a dataset of 1,555 unique, real software developer profiles with a strict 80-20 training-to-validation split matrix:

### Logistic Regression
- Validation Accuracy: 96.14 percent
- F1-Score (Weighted): 0.96
- Operational Status: Selected Production Engine (smooth probability boundaries and rapid microservice execution)

### Random Forest
- Validation Accuracy: 94.86 percent
- F1-Score (Weighted): 0.94
- Operational Status: Benchmarked Baseline

### Decision Tree
- Validation Accuracy: 94.21 percent
- F1-Score (Weighted): 0.94
- Operational Status: Benchmarked Baseline

---

## Media and Deployments

### Project Video Walkthrough
[YouTube Link Here]


---

## Installation and Local Setup

To run the production Flask application on a local server network, execute the commands below in your terminal environment:

bash
git clone [https://github.com/Shashikala-11/intern-portfolio-readiness-scorer.git](https://github.com/Shashikala-11/intern-portfolio-readiness-scorer.git)
cd intern-portfolio-readiness-scorer
pip install -r requirements.txt
python app.py

---

## Comprehensive Contribution Matrix

### Shashikala Gupta

* Core Title: Project Team Lead and Ingestion Architecture Co-Lead
* Contribution Delivery: Managed the end-to-end engineering roadmap and checkpoints. Co-designed and optimized the high-speed data harvester script. Led the log-scaled feature matrix math configurations and coordinated the final microservice integrated product rollout.

### Ayush and Shivangi

* Core Title: Data Engineering and Extraction Track
* Contribution Delivery: Collaborated on the core data harvesting script setup for the 1,555 rows. Partnered on defining evaluation matrix logic rules, parameter scales, and training data matrix preparation.

### Vedant

* Core Title: Business Intelligence Track
* Contribution Delivery: Designed and built the enterprise Power BI dashboards and advanced business analytics reports. Modeled the metrics distributions, handled cross-filtering reports for management, and delivered actionable insights on candidate placement distributions.

### Faraz

* Core Title: Machine Learning Track
* Contribution Delivery: Built, trained, and tuned the machine learning algorithms. Conducted cross-validation and hyperparameter optimization to achieve a peak validation accuracy of 96.14 percent with the Logistic Regression model, and built comparative baselines for the Random Forest and Decision Tree models.

### Saksham and Rajlakshmi

* Core Title: Full-Stack Deployment Track
* Contribution Delivery: Developed the production web app infrastructure. Wrapped the serialized model binaries into a scalable Flask microservice backend and designed the front-end user interface form layout to handle live queries seamlessly.

---

## Acknowledgments

We express our sincere gratitude to the engineering board, project mentors, and the leadership team at Graphura Private Limited for providing the operational guidance, resources, and deployment ecosystem necessary to bring this machine learning pipeline to life.

```

```
