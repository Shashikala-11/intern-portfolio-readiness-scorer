import requests
import pandas as pd
import time
from datetime import datetime

import os
from dotenv import load_dotenv
# Configuration
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
BASE_URL = "https://api.github.com"

def fetch_bulk_usernames(target_count=150):
    """Discovers real active developers using the GitHub Search Users API."""
    usernames = []
    page = 1
    
    # Query: Users with > 5 repositories, focusing on active profiles
    query = "type:user+repos:>5"
    
    while len(usernames) < target_count:
        url = f"{BASE_URL}/search/users?q={query}&per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"Search API limit reached or error code: {response.status_code}")
            break
            
        items = response.json().get('items', [])
        if not items:
            break
            
        for item in items:
            usernames.append(item['login'])
            if len(usernames) >= target_count:
                break
        
        print(f"Discovered {len(usernames)} user handles so far...")
        page += 1
        time.sleep(2)  # Search rate limit protection
        
    return usernames

def extract_deep_metrics(username):
    """Queries deep profile and repository metrics for an individual user."""
    try:
        # 1. Pull Comprehensive User Profile Details
        user_res = requests.get(f"{BASE_URL}/users/{username}", headers=HEADERS)
        if user_res.status_code != 200:
            return None
        u_data = user_res.json()
        
        # Calculate profile age
        created_at = datetime.strptime(u_data.get('created_at', '2020-01-01T00:00:00Z'), "%Y-%m-%dT%H:%M:%SZ")
        age_days = (datetime.now() - created_at).days
        
        # 2. Pull Repositories (Up to 100 repositories for deep aggregation)
        repos_res = requests.get(f"{BASE_URL}/users/{username}/repos?per_page=100&sort=updated", headers=HEADERS)
        repos_data = repos_res.json() if repos_res.status_code == 200 else []
        
        # Initializing aggregation metrics
        total_stars = 0
        total_forks = 0
        total_watchers = 0
        total_open_issues = 0
        total_size = 0
        has_pages_count = 0
        licenses_count = 0
        readmes_count = 0
        total_readme_size = 0
        languages_set = set()
        topics_count = 0
        primary_languages = {}
        
        repo_count = len(repos_data)
        
        for repo in repos_data:
            total_stars += repo.get('stargazers_count', 0)
            total_forks += repo.get('forks_count', 0)
            total_watchers += repo.get('watchers_count', 0)
            total_open_issues += repo.get('open_issues_count', 0)
            total_size += repo.get('size', 0)
            
            if repo.get('has_pages'):
                has_pages_count += 1
            if repo.get('license'):
                licenses_count += 1
                
            lang = repo.get('language')
            if lang:
                languages_set.add(lang)
                primary_languages[lang] = primary_languages.get(lang, 0) + 1
                
            # Count topics attached to the repositories
            topics_count += len(repo.get('topics', []))
            
            # Sub-query for README metrics
            readme_res = requests.get(f"{BASE_URL}/repos/{username}/{repo['name']}/readme", headers=HEADERS)
            if readme_res.status_code == 200:
                readmes_count += 1
                total_readme_size += readme_res.json().get('size', 0)
        
        # Computed descriptive attributes
        avg_repo_size = total_size / repo_count if repo_count > 0 else 0
        readme_presence_rate = (readmes_count / repo_count) if repo_count > 0 else 0
        avg_readme_size = (total_readme_size / readmes_count) if readmes_count > 0 else 0
        license_rate = (licenses_count / repo_count) if repo_count > 0 else 0
        top_lang = max(primary_languages, key=primary_languages.get) if primary_languages else "None"
        
        # Final expanded dictionary mapping directly to the data schema
        return {
            "Username": username,
            "Full_Name": u_data.get('name'),               # REAL FULL NAME
            "Bio": u_data.get('bio'),
            "Location": u_data.get('location'),
            "Company": u_data.get('company'),
            "Public_Repos_Count": u_data.get('public_repos', 0),
            "Public_Gists_Count": u_data.get('public_gists', 0),
            "Followers_Count": u_data.get('followers', 0),
            "Following_Count": u_data.get('following', 0),
            "Account_Age_Days": age_days,
            "Total_Stars": total_stars,
            "Total_Forks": total_forks,
            "Total_Watchers": total_watchers,
            "Open_Issues_Count": total_open_issues,
            "Avg_Repo_Size_KB": round(avg_repo_size, 2),
            "Unique_Languages_Count": len(languages_set),
            "Primary_Language": top_lang,
            "Total_Topics_Count": topics_count,
            "Readme_Presence_Rate": round(readme_presence_rate, 2),
            "Avg_Readme_Size_Bytes": round(avg_readme_size, 2),
            "License_Adoption_Rate": round(license_rate, 2),
            "Has_Portfolio_Link": 1 if u_data.get('blog') else 0,
            "GitHub_Pages_Count": has_pages_count
        }
    except Exception as e:
        print(f"Skipping profile {username} due to unexpected exception: {e}")
        return None

# Execution Flow
print("Step 1: Discovering user handles globally via Search API...")
target_users = fetch_bulk_usernames(target_count=100) # Set higher (e.g., 500) for large-scale training

print("\nStep 2: Performing deep data extraction per user...")
bulk_dataset = []
for idx, user in enumerate(target_users):
    print(f"[{idx+1}/{len(target_users)}] Scraping full profile for: {user}")
    profile_data = extract_deep_metrics(user)
    if profile_data:
        bulk_dataset.append(profile_data)
    time.sleep(1.2)  # Steady back-off to easily safely bypass Core API limits

# Convert to structured dataset and output
final_df = pd.DataFrame(bulk_dataset)
final_df.to_csv("comprehensive_bulk_portfolios.csv", index=False)
print(f"\nExecution Complete! Saved {len(final_df)} highly detailed profiles to 'comprehensive_bulk_portfolios.csv'.")