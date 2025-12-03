#!/bin/bash
# GitHub Repository Setup Script

echo "Blueprint Analysis AI - GitHub Setup"
echo "======================================"

# Initialize git repository
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files..."
git add .

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Blueprint Analysis AI with SAM2, OCR, and Fine-tuned Classification"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Copy the repository URL"
echo "3. Run: git remote add origin <your-repo-url>"
echo "4. Run: git push -u origin main"
echo ""
echo "Optional: Add badges to README.md:"
echo "- ![Python](https://img.shields.io/badge/Python-3.11-blue)"
echo "- ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)"
echo "- ![SAM2](https://img.shields.io/badge/SAM2-Meta-green)"
