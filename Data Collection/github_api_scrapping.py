import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Configure your token and headers
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
BASE_URL = "https://api.github.com"

def get_candidate_metrics(username):
    try:
        # 1. Fetch User Profile Data (Consistency base)
        user_res = requests.get(f"{BASE_URL}/users/{username}", headers=HEADERS)
        if user_res.status_code != 200:
            return None
        user_data = user_res.json()
        
        public_repos = user_data.get('public_repos', 0)
        followers = user_data.get('followers', 0)
        has_portfolio_site = 1 if user_data.get('blog') else 0 # Presentation signal
        
        # 2. Fetch User Repositories
        repos_res = requests.get(f"{BASE_URL}/users/{username}/repos?per_page=10&sort=updated", headers=HEADERS)
        repos_data = repos_res.json() if repos_res.status_code == 200 else []
        
        total_stars = 0
        total_forks = 0
        readme_sizes = []
        languages_set = set()
        has_pages_count = 0
        
        for repo in repos_data:
            total_stars += repo.get('stargazers_count', 0)
            total_forks += repo.get('forks_count', 0)
            if repo.get('has_pages'):
                has_pages_count += 1
                
            # Fetch Languages per Repo
            lang_url = repo.get('languages_url')
            lang_res = requests.get(lang_url, headers=HEADERS)
            if lang_res.status_code == 200:
                languages_set.update(lang_res.json().keys())
                
            # Fetch README size/presence as a proxy for documentation
            readme_res = requests.get(f"{BASE_URL}/repos/{username}/{repo['name']}/readme", headers=HEADERS)
            if readme_res.status_code == 200:
                readme_sizes.append(readme_res.json().get('size', 0))
        
        # Calculate Averages/Aggregates
        avg_readme_size = sum(readme_sizes) / len(readme_sizes) if readme_sizes else 0
        unique_languages = len(languages_set)
        
        # Construct the raw feature row
        return {
            "Username": username,
            "Num_Projects": public_repos,                # Data field [cite: 17, 30]
            "Total_Stars": total_stars,                  # Proxy for Business Impact [cite: 13]
            "Total_Forks": total_forks,                  # Proxy for Business Impact [cite: 13]
            "Unique_Languages": unique_languages,        # Tools Used count [cite: 17, 30]
            "Avg_Readme_Size": avg_readme_size,          # Documentation proxy [cite: 17, 30]
            "Has_Portfolio_Site": has_portfolio_site,    # Presentation feature [cite: 13, 17]
            "Has_GitHub_Pages": has_pages_count          # Presentation feature [cite: 13]
        }
        
    except Exception as e:
        print(f"Error processing {username}: {e}")
        return None

# Example usage with a seed list of candidates
candidates = ["octocat", "torvalds"] # Replace with real developer handles
dataset = []

for idx, candidate in enumerate(candidates):
    print(f"Fetching data for {candidate}...")
    metrics = get_candidate_metrics(candidate)
    if metrics:
        dataset.append(metrics)
    time.sleep(1) # Polite API backoff delay

df = pd.DataFrame(dataset)
df.to_csv("raw_portfolio_data.csv", index=False)
print("Data collection complete. Saved to raw_portfolio_data.csv")