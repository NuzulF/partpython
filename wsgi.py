import sys
import os

project_path = "/var/www/html/repopythonv2"

sys.path.insert(0, project_path)
os.chdir(project_path)

# NLTK path
os.environ["NLTK_DATA"] = f"{project_path}/nltk_data"

# HuggingFace cache
os.environ["TRANSFORMERS_CACHE"] = "/var/www/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/var/www/.cache/huggingface/datasets"
os.environ["HF_HOME"] = "/var/www/.cache/huggingface"

from run import app
application = app