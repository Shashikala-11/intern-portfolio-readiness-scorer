import os
import re
import pandas as pd
from pdfminer.high_level import extract_text

# Path to the folder where you downloaded the resumes
RESUME_FOLDER = "path/to/your/resume/folder"

def extract_info_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    
    # 1. Extract Email (Intern_ID)
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    intern_id = email.group(0) if email else os.path.basename(pdf_path)

    # 2. Extract Skills (Technical_Skills)
    # Add common skills you expect to find
    skill_keywords = ['Python', 'SQL', 'Java', 'React', 'Tableau', 'Power BI', 'Machine Learning', 'AWS']
    found_skills = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    
    # 3. Count Projects (Total_Projects)
    # Looks for common headers like "Projects" or "Professional Experience"
    project_sections = re.findall(r'(Project|Experience|Assignment)', text, re.IGNORECASE)
    total_projects = len(project_sections)

    return {
        "Intern_ID": intern_id,
        "Technical_Skills_List": ", ".join(found_skills),
        "Technical_Skills_Score": min(len(found_skills), 10), # Simple score based on count
        "Total_Projects": total_projects,
        "Full_Text": text # Keep this to help you manually score the rest
    }

# Process all resumes
data = []
for filename in os.listdir(RESUME_FOLDER):
    if filename.endswith(".pdf"):
        file_path = os.path.join(RESUME_FOLDER, filename)
        data.append(extract_info_from_pdf(file_path))

# Create DataFrame with your specific columns
df = pd.DataFrame(data)

# Add placeholder columns for manual scoring as per your CSV structure 
cols_to_add = [
    'Domain', 'Project_Quality', 'Documentation Score', 'Presentation Quality', 
    'Business Impact', 'Consistency Score', 'Commit Frequency', 
    'README Word Count', 'Project Diversity'
]
for col in cols_to_add:
    df[col] = "" # Leave empty for your manual audit

# Save to the CSV format you provided 
df.to_csv("Intern_Portfolio_Readiness_Scorer_Dataset.csv", index=False)
print("Extraction complete. Check the CSV file!")