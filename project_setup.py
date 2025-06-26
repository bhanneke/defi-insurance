#!/usr/bin/env python3
"""
Enhanced Project Setup Script for Academic Research Projects
Author: Your Name
Date: 2025

Usage:
1. Copy this script to your new project folder
2. Run: python project_setup.py
3. Script will either set up new project or activate existing environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class ProjectSetup:
    def __init__(self):
        self.project_dir = Path.cwd()
        self.project_name = self._get_project_name()
        self.venv_name = "venv"
        self.venv_path = self.project_dir / self.venv_name
        self.python_version = self._get_python_version()
        self.python_executable = self._get_python_executable()
        
    def _get_project_name(self):
        """Get project name from directory, replace spaces with underscores"""
        return self.project_dir.name.replace(" ", "_").replace("-", "_")
    
    def _get_python_version(self):
        """Get current Python version"""
        return f"{sys.version_info.major}.{sys.version_info.minor}"
    
    def _get_python_executable(self):
        """Get the correct Python executable for this system"""
        # Use the same Python that's running this script
        return sys.executable
    
    def _run_command(self, command, shell=True, check=True):
        """Run shell command with error handling"""
        try:
            result = subprocess.run(command, shell=shell, check=check, 
                                  capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {command}")
            print(f"Error: {e.stderr}")
            return None
    
    def check_existing_setup(self):
        """Check if project is already set up"""
        checks = {
            "Virtual Environment": self.venv_path.exists(),
            "Requirements File": (self.project_dir / "requirements.txt").exists(),
            "Git Repository": (self.project_dir / ".git").exists(),
            "Environment File": (self.project_dir / ".env").exists(),
            "Gitignore": (self.project_dir / ".gitignore").exists()
        }
        
        all_exist = all(checks.values())
        
        print(f"\n=== Project Setup Status for '{self.project_name}' ===")
        for item, exists in checks.items():
            status = "✓" if exists else "✗"
            print(f"{status} {item}")
        
        return all_exist
    
    def create_requirements_txt(self):
        """Create requirements.txt with common academic packages"""
        requirements_content = """# Data Analysis and Scientific Computing
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0

# Jupyter and Interactive Development
jupyter>=1.0.0
ipykernel>=6.20.0

# Data Visualization
plotly>=5.0.0

# File Handling
openpyxl>=3.1.0
xlsxwriter>=3.0.0

# HTTP Requests and APIs
requests>=2.28.0

# Environment Management
python-dotenv>=1.0.0

# Add your specific packages below:
# uncomment and modify as needed
# scikit-learn>=1.3.0
# beautifulsoup4>=4.12.0
# nltk>=3.8.0
# spacy>=3.6.0
"""
        
        requirements_path = self.project_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(requirements_content)
        print("✓ Created requirements.txt with common academic packages")
    
    def create_gitignore(self):
        """Create comprehensive .gitignore for research projects"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local

# Data files (uncomment if you don't want to track data)
# *.csv
# *.xlsx
# *.json
# *.parquet
# data/
# raw_data/

# Output files
*.log
*.out
*.err

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Academic/Research specific
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.synctex.gz
*.toc
"""
        
        gitignore_path = self.project_dir / ".gitignore"
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore for research projects")
    
    def create_env_file(self):
        """Create .env file template for API keys"""
        env_content = """# Environment Variables for API Keys and Configuration
# Copy this file to .env and fill in your actual values
# Never commit .env to version control!

# Example API Keys (uncomment and fill as needed)
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
# TWITTER_API_KEY=your_twitter_key_here
# DATABASE_URL=your_database_url_here

# Project Configuration
PROJECT_NAME={}
PYTHON_VERSION={}

# Data Sources
# DATA_DIR=./data
# OUTPUT_DIR=./output
""".format(self.project_name, self.python_version)
        
        env_path = self.project_dir / ".env.template"
        with open(env_path, "w") as f:
            f.write(env_content)
        
        # Create empty .env file if it doesn't exist
        actual_env = self.project_dir / ".env"
        if not actual_env.exists():
            actual_env.touch()
        
        print("✓ Created .env.template and .env files")
    
    def create_readme(self):
        """Create basic README.md"""
        readme_content = f"""# {self.project_name}

## Project Description
Brief description of your research project.

## Setup
This project was set up using the enhanced project setup script.

### Requirements
- Python {self.python_version}
- Virtual environment (included)

### Installation
1. Activate the virtual environment:
   - Windows: `venv\\Scripts\\activate`
   - macOS/Linux: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`

### Environment Variables
Copy `.env.template` to `.env` and fill in your API keys and configuration.

## Usage
Describe how to use your project/analysis.

## Data
Describe your data sources and structure.

## Results
Document your findings and outputs.
"""
        
        readme_path = self.project_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print("✓ Created README.md")
    
    def create_virtual_environment(self):
        """Create virtual environment"""
        print(f"Creating virtual environment with Python {self.python_version}...")
        
        # Use the same Python executable that's running this script
        result = self._run_command(f'"{self.python_executable}" -m venv {self.venv_name}')
        if result is None:
            print("Failed to create virtual environment")
            return False
        
        print("✓ Created virtual environment")
        return True
    
    def install_requirements(self):
        """Install packages from requirements.txt"""
        print("Installing requirements...")
        
        # Get the correct pip path based on OS
        if platform.system() == "Windows":
            pip_path = self.venv_path / "Scripts" / "pip"
        else:
            pip_path = self.venv_path / "bin" / "pip"
        
        # Upgrade pip first
        result = self._run_command(f'"{pip_path}" install --upgrade pip')
        if result is None:
            print("Warning: Could not upgrade pip")
        
        # Install requirements
        result = self._run_command(f'"{pip_path}" install -r requirements.txt')
        if result is None:
            print("Failed to install requirements")
            return False
        
        print("✓ Installed requirements")
        return True
    
    def setup_git(self):
        """Initialize git repository"""
        # Check if git is available
        result = self._run_command("git --version", check=False)
        if result is None or result.returncode != 0:
            print("Git not found - skipping git setup")
            return
        
        # Initialize git repo if not exists
        if not (self.project_dir / ".git").exists():
            self._run_command("git init")
            # Configure git if not already configured
            result = self._run_command("git config user.name", check=False)
            if result and not result.stdout.strip():
                print("Note: Git user not configured. You may want to run:")
                print("  git config --global user.name 'Your Name'")
                print("  git config --global user.email 'your.email@example.com'")
            
            self._run_command("git add .")
            self._run_command(f'git commit -m "Initial commit for {self.project_name}"', check=False)
            print("✓ Initialized git repository")
        else:
            print("✓ Git repository already exists")
    
    def activate_environment_message(self):
        """Show activation message with Mac-specific instructions"""
        activate_cmd = f"source {self.venv_name}/bin/activate"
        
        print(f"\n=== Environment Ready! ===")
        print(f"To activate the virtual environment, run:")
        print(f"  {activate_cmd}")
        print(f"\nMac-specific VS Code setup:")
        print(f"1. Open VS Code in this directory: code .")
        print(f"2. Select Python interpreter: Cmd+Shift+P -> 'Python: Select Interpreter'")
        print(f"3. Choose: {self.venv_path}/bin/python")
        print(f"\nOr create a VS Code shortcut:")
        print(f"  echo 'source venv/bin/activate && code .' > start_project.sh")
        print(f"  chmod +x start_project.sh")
        print(f"  ./start_project.sh")
    
    def run_setup(self):
        """Main setup process"""
        print(f"Enhanced Project Setup Script")
        print(f"Working Directory: {self.project_dir}")
        print(f"Python Version: {self.python_version}")
        
        # Check if already set up
        if self.check_existing_setup():
            print(f"\n✓ Project '{self.project_name}' is already set up!")
            self.activate_environment_message()
            return
        
        print(f"\nSetting up new project '{self.project_name}'...")
        
        # Create all components
        if not self.venv_path.exists():
            if not self.create_virtual_environment():
                return
        
        if not (self.project_dir / "requirements.txt").exists():
            self.create_requirements_txt()
        
        if not (self.project_dir / ".gitignore").exists():
            self.create_gitignore()
        
        if not (self.project_dir / ".env").exists():
            self.create_env_file()
        
        if not (self.project_dir / "README.md").exists():
            self.create_readme()
        
        # Install requirements
        self.install_requirements()
        
        # Setup git
        self.setup_git()
        
        print(f"\n✅ Project '{self.project_name}' setup complete!")
        self.activate_environment_message()


if __name__ == "__main__":
    setup = ProjectSetup()
    setup.run_setup()