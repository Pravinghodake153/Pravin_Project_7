"""
============================================================
config.py — Configuration Management
============================================================
PURPOSE:
    Loads all settings from the .env file so we never
    hardcode secrets or file paths in our code.

HOW IT WORKS:
    1. Reads the .env file from the project root
    2. Stores each setting in a simple variable
    3. Other files import these variables instead of
       reading .env themselves
============================================================
"""

import os
from dotenv import load_dotenv

# ----------------------------------------------------------
# Load .env file
# ----------------------------------------------------------
# This reads the .env file and puts all key=value pairs
# into the operating system's environment variables.
# After this call, os.getenv("CHUNK_SIZE") will return "500"
# ----------------------------------------------------------
load_dotenv()

# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------
# os.path.dirname(__file__) gives us the folder where THIS
# file (config.py) lives. We use it to build absolute paths
# so the project works from any working directory.
# ----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Where the PDF file is stored
PDF_PATH = os.path.join(
    PROJECT_ROOT,
    os.getenv("PDF_PATH", "data/transformer_rag_research_paper_50p.pdf")
)

# Where ChromaDB will save its database files
CHROMA_PERSIST_DIR = os.path.join(
    PROJECT_ROOT,
    os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
)

# ----------------------------------------------------------
# Embedding model
# ----------------------------------------------------------
# This is a small, free model from Hugging Face that runs
# on your CPU. It converts text into 384-dimensional vectors
# (lists of numbers) that represent the meaning of the text.
# ----------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ----------------------------------------------------------
# Chunking settings
# ----------------------------------------------------------
# CHUNK_SIZE: How many characters each text piece should be.
#   - 500 chars is roughly 1-2 paragraphs
#   - Too small = loses context, Too big = too vague
#
# CHUNK_OVERLAP: How many characters overlap between chunks.
#   - 100 chars means consecutive chunks share some text
#   - This prevents important info from being split across
#     two chunks and lost at the boundary
# ----------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ----------------------------------------------------------
# ChromaDB collection name
# ----------------------------------------------------------
# A "collection" in ChromaDB is like a table in a database.
# All our PDF chunks will be stored in this one collection.
# ----------------------------------------------------------
COLLECTION_NAME = "rag_research_paper"

# ----------------------------------------------------------
# Retrieval settings
# ----------------------------------------------------------
# How many chunks to retrieve when answering a question.
# More chunks = more context but also more noise.
# 5 is a good balance for a 50-page document.
# ----------------------------------------------------------
TOP_K_RESULTS = 5
