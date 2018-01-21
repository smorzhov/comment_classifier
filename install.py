"""
It installs some dependencies

Usage: python install.py
"""

import nltk

# download set of high-frequency words (like the, to) and words, which have little lexical content
nltk.download('stopwords')

nltk.download('punkt')