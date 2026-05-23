import requests
import pandas as pd
import time

import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")
HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

def fetch_optimized_dataset(target_count=1550):
    bulk_data = []
    # Using multiple location queries to bypass the 1,000 search result limit
    hubs = ["India", "United States", "United Kingdom", "Germany", "Canada", "Singapore"]
    
    for hub in hubs:
        if len(bulk_data) >= target_count:
            break
        print(f"Gathering data from hub: {hub}...")
        
        for page in range(1, 11): # 10 pages max per search
            if len(bulk_data) >= target_count:
                break
                
            search_url = f"https://api.github.com/search/users?q=location:{hub}+repos:5..50&per_page=100&page={page}"
            res = requests.get(search_url, headers=HEADERS).json()
            items = res.get('items', [])
            if not items:
                break
                
            for item in items:
                username = item['login']
                
                # Request 1: Full profile details
                u_res = requests.get(f"https://api.github.com/users/{username}", headers=HEADERS).json()
                full_name = u_res.get('name')
                if not full_name: 
                    continue # Skip accounts without a real name
                
                # Request 2: Get all repos at once to aggregate stats without looping individual readme files
                r_res = requests.get(f"https://api.github.com/users/{username}/repos?per_page=50", headers=HEADERS).json()
                
                stars = sum(r.get('stargazers_count', 0) for r in r_res if isinstance(r, dict))
                forks = sum(r.get('forks_count', 0) for r in r_res if isinstance(r, dict))
                has_pages = sum(1 for r in r_res if isinstance(r, dict) and r.get('has_pages'))
                
                bulk_data.append({
                    "Username": username,
                    "Full_Name": full_name,
                    "Location": u_res.get('location'),
                    "Public_Repos_Count": u_res.get('public_repos', 0),
                    "Followers_Count": u_res.get('followers', 0),
                    "Total_Stars": stars,
                    "Total_Forks": forks,
                    "Has_Portfolio_Site": 1 if u_res.get('blog') else 0,
                    "GitHub_Pages_Count": has_pages
                })
                
                if len(bulk_data) % 50 == 0:
                    print(f"Collected {len(bulk_data)} real candidate records...")
                
                # Small pause to safely stay within the 5,000 hourly authenticated request limit
                time.sleep(0.2) 
                
    return pd.DataFrame(bulk_data)

df = fetch_optimized_dataset()
df.to_csv("comprehensive_1500_portfolios.csv", index=False)