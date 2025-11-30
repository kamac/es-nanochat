#!/bin/bash
# Zip the nanochat repository for easy upload to Jupyter/Colab
# Excludes unnecessary files to keep size small

echo "Creating nanochat.zip..."

# Get the repo directory name
REPO_DIR=$(basename "$PWD")

# Create zip excluding unnecessary files
cd ..
zip -r nanochat.zip "$REPO_DIR" \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*.pyo" \
    -x "*/.pytest_cache/*" \
    -x "*/venv/*" \
    -x "*/.venv/*" \
    -x "*/env/*" \
    -x "*/.env/*" \
    -x "*/node_modules/*" \
    -x "*/.DS_Store" \
    -x "*/tokenized_data/*" \
    -x "*/base_checkpoints/*" \
    -x "*.egg-info/*" \
    -x "*/dist/*" \
    -x "*/build/*" \
    -x "*/.ipynb_checkpoints/*" \
    -q

cd "$REPO_DIR"

# Move zip to current directory
mv ../nanochat.zip .

# Get file size
if [[ "$OSTYPE" == "darwin"* ]]; then
    SIZE=$(du -h nanochat.zip | cut -f1)
else
    SIZE=$(du -h nanochat.zip | cut -f1)
fi

echo "✓ Created nanochat.zip ($SIZE)"
echo ""
echo "To upload to Jupyter:"
echo "1. Upload nanochat.zip to your Jupyter environment"
echo "2. Run: !unzip -q nanochat.zip"
echo "3. Run: %cd nanochat"
echo ""
echo "Or use the unzip_in_jupyter.py script for easier setup"

