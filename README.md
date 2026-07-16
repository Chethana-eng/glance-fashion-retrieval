# Semantic Fashion Image Retrieval using OpenCLIP and FAISS

Semantic fashion image retrieval system that understands natural language queries using OpenCLIP and retrieves visually similar fashion images through FAISS vector search. The project incorporates an Intelligent Query Processing module that extracts fashion-specific attributes before retrieval.

## Overview

The objective of this project is to retrieve relevant fashion images from the Fashionpedia dataset using descriptive natural language queries instead of keyword matching. The system combines OpenCLIP embeddings with FAISS similarity search and introduces a query processing pipeline to improve retrieval quality.

---

## Features

- Semantic image retrieval using natural language
- OpenCLIP-based image and text embeddings
- FAISS vector similarity search
- Intelligent query parsing
- Weighted prompt generation
- Top-K image retrieval
- Modular architecture
- Indexed approximately 48,823 Fashionpedia images

---

## System Architecture/ Workflow

Fashionpedia Images
        │
Dataset Loader
        │
OpenCLIP Image Encoder
        │
Image Embeddings
        │
FAISS Index

────────────────────────────

User Query
        │
Query Parser
        │
Weighted Query Builder
        │
OpenCLIP Text Encoder
        │
Query Embedding
        │
FAISS Search
        │
Top-K Retrieved Images

---

## Technology Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Framework | PyTorch |
| Vision-Language Model | OpenCLIP |
| Vector Database | FAISS |
| Dataset | Fashionpedia |

---

## Sample Queries

- A person wearing a blue shirt in a modern office
- A woman wearing a red dress
- A person in a bright yellow raincoat
- A blue puffer jacket

---

## Installation

git clone https://github.com/Chethana-eng/glance-fashion-retrieval.git
cd glance-fashion-retrieval
pip install -r requirements.txt
python main.py


