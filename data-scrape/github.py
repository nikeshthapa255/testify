import requests
import os
import shutil
from git import Repo
import yaml
import json
import time

import setup

# Replace with your GitHub personal access token
GITHUB_TOKEN = setup.GIT_TOKEN

headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

def get_github_repos(language, topic, stars, per_page=10, page=3):
    try:
        url = f"https://api.github.com/search/repositories?q=language:{language}+topic:{topic}+stars:>{stars}&sort=stars&order=desc&per_page={per_page}&page={page}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return {"items": []}

def clone_repo(repo_url, repo_name):
    repo_dir = os.path.join("repos", repo_name)
    try:
        if not os.path.exists(repo_dir):
            Repo.clone_from(repo_url, repo_dir)
    except Exception as e:
        print(f"Error cloning {repo_name}: {e}")
        return None

    # Try to restore the repository in case of checkout failure
    try:
        repo = Repo(repo_dir)
        repo.git.restore('--source=HEAD', ':')
    except Exception as e:
        print(f"Error checking out {repo_name}: {e}")
        return None

    return repo_dir

def extract_component_files(repo_dir):
    components = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".jsx") or file.endswith(".js") or file.endswith(".tsx") or file.endswith(".ts"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        components.append({
                            "path": os.path.join(root, file),
                            "content": f.read()
                        })
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
    return components

def extract_test_files(repo_dir):
    tests = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".test.js") or file.endswith(".test.jsx") or file.endswith(".test.ts") or file.endswith(".test.tsx"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        tests.append({
                            "path": os.path.join(root, file),
                            "content": f.read()
                        })
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
    return tests

def uses_react_testing_library(repo_dir):
    package_json_path = os.path.join(repo_dir, 'package.json')
    try:
        with open(package_json_path, 'r', encoding='utf-8', errors='ignore') as f:
            package_json = json.load(f)
            dependencies = package_json.get('dependencies', {})
            dev_dependencies = package_json.get('devDependencies', {})
            if 'react-testing-library' in dependencies or 'react-testing-library' in dev_dependencies or 'jest' in dependencies or 'jest' in dev_dependencies:
                return True
    except Exception as e:
        print(f"Error reading package.json: {e}")
    return False

def save_data(repo_name, components, tests):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        with open(os.path.join(data_dir, f"{repo_name}_components.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(components, f, allow_unicode=True)
    except Exception as e:
        print(f"Error saving components for {repo_name}: {e}")

    try:
        with open(os.path.join(data_dir, f"{repo_name}_tests.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(tests, f, allow_unicode=True)
    except Exception as e:
        print(f"Error saving tests for {repo_name}: {e}")


def remove_readonly(func, path, excinfo):
    os.chmod(path, 0o777)
    func(path)

def delete_repo(repo_dir):
    retry_count = 3
    for _ in range(retry_count):
        try:
            shutil.rmtree(repo_dir, onerror=remove_readonly)
            return True
        except Exception as e:
            print(f"Error deleting repository {repo_dir}: {e}. Retrying...")
            time.sleep(1)
    print(f"Failed to delete repository {repo_dir} after {retry_count} retries.")
    return False

def main():
    if not os.path.exists("repos"):
        os.makedirs("repos")
    
    repos = get_github_repos('TypeScript', 'react', 100)
    for repo in repos['items']:
        repo_name = repo['name']
        clone_url = repo['clone_url']
        
        print(f"Cloning {repo_name}...")
        repo_dir = clone_repo(clone_url, repo_name)
        
        if repo_dir is None:
            continue
        
        if not uses_react_testing_library(repo_dir):
            print(f"Skipping {repo_name} as it does not use React Testing Library or Jest.")
            # Clean up the repository directory if it exists
            if os.path.exists(repo_dir):
                delete_repo(repo_dir)
            continue
        
        print(f"Extracting components from {repo_name}...")
        components = extract_component_files(repo_dir)
        
        print(f"Extracting test cases from {repo_name}...")
        tests = extract_test_files(repo_dir)
        
        print(f"Saving data for {repo_name}...")
        save_data(repo_name, components, tests)
        print(f"Data saved for {repo_name}")
        
        # Delete the cloned repository after processing
        print(f"Deleting repository {repo_name} from local drive...")
        delete_repo(repo_dir)

if __name__ == "__main__":
    main()