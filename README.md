# HackLLM - Hallucination Detection System

A comprehensive hallucination detection system for evaluating outputs from Large Language Models (LLMs), developed for the SHROOM task at SemEval 2024.

## Overview

This project implements a multi-dimensional approach to detect hallucinations in NLP model outputs across various tasks including Machine Translation (MT), Definition Modeling (DM), and Paraphrase Generation (PG). The system analyzes four distinct types of hallucinations to provide robust detection capabilities.

## What is SHROOM?

SHROOM (Shared-task on Hallucinations and Observable Overgeneration Mistakes) is a task at SemEval 2024 focused on binary classification of model outputs. The goal is to determine whether a given production from an NLP model constitutes a hallucination.

Participants are evaluated on:
- **Accuracy**: Correctness of hallucination vs non-hallucination classification
- **Probability Correlation**: How well predicted probabilities correlate with empirical annotator probabilities

## Hallucination Detection Approach

The system evaluates four distinct dimensions of hallucinations:

### 1. Linguistic Hallucination (20% weight)
- Measures language quality and fluency
- Uses perplexity scoring via GPT-2
- Grammar error detection with LanguageTool
- Lexical overlap analysis (ROUGE-L, Jaccard similarity)

### 2. Logical Hallucination (35% weight)
- Detects contradictions and logical inconsistencies
- Utilizes RoBERTa-large-mnli for Natural Language Inference
- Computes entailment/contradiction probabilities

### 3. Factual Hallucination (30% weight)
- Verifies factual accuracy of generated content
- Named Entity Recognition via spaCy
- Entity support ratio calculation
- Cross-references entities with source/target references

### 4. Contextual Hallucination (15% weight)
- Measures semantic drift from context
- Sentence embeddings via all-MiniLM-L6-v2
- Cosine similarity for semantic alignment

## Models & Technologies

- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic embeddings
- **NLI Model**: `roberta-large-mnli` for entailment detection
- **Perplexity Model**: `gpt2` for fluency scoring
- **NER**: spaCy `en_core_web_sm` for entity extraction
- **Grammar Checker**: LanguageTool for linguistic quality
- **Libraries**: transformers, sentence-transformers, spaCy, NLTK, pandas, scikit-learn

## Dataset

The system works with the SHROOM dataset containing:

### Tasks
- **MT (Machine Translation)**: Detects hallucinations in translations
- **DM (Definition Modeling)**: Validates word definitions in context
- **PG (Paraphrase Generation)**: Checks paraphrase accuracy

### Data Fields
- `hyp`: Model-generated hypothesis
- `src`: Source input text
- `tgt`: Target reference text
- `ref`: Reference indicator (src/tgt/either)
- `task`: Task type (MT/DM/PG)
- `label`: Ground truth annotation
- `p(Hallucination)`: Annotator probability

### Dataset Files
```
data/
├── train.model-agnostic.json     # Training data (30,000 examples)
├── train.model-aware.v2.json     # Training data with model info
├── val.model-agnostic.json       # Validation data
├── val.model-aware.v2.json       # Validation with model info
├── test.model-agnostic.json      # Test data
├── test.model-aware.json         # Test with model info
├── train_scored.csv              # Processed training results
└── README-v2.txt                 # Dataset documentation
```

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
```

### Setup
```bash
# Clone the repository
git clone https://github.com/108nitish/HackLLM-Stuff
cd HackLLM-Stuff

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm
```

### Required Packages
```
transformers
sentence-transformers
torch
spacy
language-tool-python
nltk
pandas
numpy
scikit-learn
scipy
rouge-score
tqdm
jupyter
```

## Usage

### Jupyter Notebook
Open and run `test.ipynb` which contains the complete implementation pipeline:
```bash
jupyter notebook test.ipynb
```

### Basic Pipeline
```python
from hallucination_detector import run_pipeline

# Process dataset
df_results = run_pipeline(
    input_path="./data/train.model-agnostic.json",
    output_path="./data/train_scored.csv",
    evaluate=True
)
```

### Custom Configuration
```python
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector(
    sim_threshold=0.75,    # Similarity threshold
    ppl_hi=60.0,          # Perplexity upper bound
    w_ling=0.20,          # Linguistic weight
    w_log=0.35,           # Logical weight
    w_fact=0.30,          # Factual weight
    w_ctx=0.15            # Contextual weight
)

# Classify single example
result = detector.classify(row_data)
```

### Command Line Usage
```bash
python -m hallucination_detector \
    --input ./data/train.model-agnostic.json \
    --output ./data/results.csv \
    --device cuda:0 \
    --evaluate \
    --w_ling 0.20 \
    --w_log 0.35 \
    --w_fact 0.30 \
    --w_ctx 0.15 \
    --sim_threshold 0.70 \
    --ppl_hi 80.0
```

## Implementation Details

### Core Classes

#### `Providers`
Manages lazy loading of ML models:
- Sentence-BERT embeddings
- NLI models (RoBERTa)
- Language models (GPT-2)
- LanguageTool checker
- spaCy NLP pipeline

#### `SimilarityScorer`
Computes semantic similarity using sentence embeddings with cosine similarity.

#### `NLIScorer`
Natural Language Inference scoring:
- Entailment probability
- Neutral probability
- Contradiction probability

#### `LinguisticScorer`
Evaluates language quality:
- Perplexity calculation
- Grammar error counting
- Lexical overlap metrics

#### `FactualScorer`
Validates factual accuracy:
- Entity extraction
- Entity support ratio
- NLI-based fact checking

#### `ContextualScorer`
Measures contextual alignment using similarity and entailment scores.

### Reference Selection Strategy

The system intelligently chooses references based on the `ref` flag:
- `"src"`: Use source as reference
- `"tgt"`: Use target as reference
- `"either"`: Automatically select the reference with higher similarity to hypothesis

## Output Format

The system generates CSV output with the following columns:

| Column | Description |
|--------|-------------|
| `task` | Source task (MT/DM/PG) |
| `ref_flag` | Reference type specified |
| `ref_used` | Actual reference chosen |
| `sim` | Semantic similarity score (0-1) |
| `p_entail` | Entailment probability (0-1) |
| `p_neutral` | Neutral probability (0-1) |
| `p_contra` | Contradiction probability (0-1) |
| `linguistic_p` | Linguistic hallucination score (0-1) |
| `logical_p` | Logical hallucination score (0-1) |
| `factual_p` | Factual hallucination score (0-1) |
| `contextual_p` | Contextual hallucination score (0-1) |
| `hallucination_p` | Overall hallucination probability (0-1) |
| `pred_label` | Predicted label (Hallucination/Not Hallucination) |
| `gold_label` | Ground truth label (if available) |
| `gold_p` | Annotator probability (if available) |

## Evaluation Metrics

The system computes:
- **Accuracy**: Binary classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Pearson Correlation**: Linear correlation with gold probabilities
- **Spearman Correlation**: Rank correlation with gold probabilities

## Project Structure

```
HackLLM-Stuff/
├── data/                          # Dataset files
│   ├── train.model-agnostic.json  # Training data (30K examples)
│   ├── train.model-aware.v2.json  # Training with model info
│   ├── val.model-agnostic.json    # Validation data
│   ├── val.model-aware.v2.json    # Validation with model info
│   ├── test.model-agnostic.json   # Test data
│   ├── test.model-aware.json      # Test with model info
│   ├── train_scored.csv           # Processed results
│   └── README-v2.txt              # Dataset documentation
├── .ipynb_checkpoints/            # Jupyter checkpoints
├── .venv/                         # Virtual environment
├── test.ipynb                     # Main implementation notebook
└── README.md                      # This file
```

## Key Features

- **Multi-dimensional Analysis**: Combines 4 types of hallucination detection
- **Flexible Reference Selection**: Automatic or manual reference choice
- **Configurable Weights**: Adjustable importance for each dimension
- **Graceful Fallbacks**: Works even when some models are unavailable
- **Batch Processing**: Efficient processing with progress tracking (tqdm)
- **Comprehensive Metrics**: Multiple evaluation metrics for thorough assessment
- **GPU Support**: Optional CUDA acceleration for faster inference

## Algorithm Workflow

1. **Input Processing**: Load hypothesis, source, target, and reference flag
2. **Reference Selection**: Choose appropriate reference based on similarity
3. **Feature Extraction**:
   - Compute semantic embeddings
   - Calculate NLI probabilities
   - Extract named entities
   - Measure perplexity and grammar
4. **Score Computation**:
   - Linguistic: perplexity + grammar + overlap
   - Logical: contradiction probability
   - Factual: entity support + NLI
   - Contextual: semantic similarity + entailment
5. **Weighted Aggregation**: Combine scores using configurable weights
6. **Classification**: Threshold at 0.5 for binary label

## Results

Example results on training data (30,000 examples):
- Processes all three task types: MT, DM, PG
- Generates probability scores for nuanced evaluation
- Supports both model-agnostic and model-aware scenarios

## Tuning & Optimization

### Adjustable Parameters

```python
# Weights (must sum to 1.0 after normalization)
w_ling=0.20    # Linguistic hallucination weight
w_log=0.35     # Logical hallucination weight
w_fact=0.30    # Factual hallucination weight
w_ctx=0.15     # Contextual hallucination weight

# Thresholds
sim_threshold=0.70   # Similarity threshold for contextual scoring
ppl_hi=80.0          # Upper bound for perplexity normalization
```

### Performance Tips

- Use GPU (`--device cuda:0`) for faster inference
- Batch similar examples together
- Cache model outputs when processing multiple times
- Consider lighter models for faster prototyping

## Troubleshooting

### Common Issues

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**NLTK data missing:**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
```

**Out of memory:**
- Use CPU instead of GPU
- Process smaller batches
- Use quantized models

## Contributing

Contributions are welcome! Areas for improvement:
- Additional hallucination types
- More sophisticated entity matching
- Multilingual support
- Improved reference selection strategies
- Fine-tuned models for specific tasks

## License

See repository for license information.

## References

### Papers
- [SemEval 2024 SHROOM Task](https://helsinki-nlp.github.io/shroom/)
- [Noraset et al (2017) - Definition Modeling](https://dl.acm.org/doi/10.5555/3298023.3298042)
- [Bevilacqua et al (2020) - Definition Modeling Framework](https://aclanthology.org/2020.emnlp-main.585/)
- [Giulianelli et al (2023) - Model-Aware Definition Modeling](https://aclanthology.org/2023.acl-long.176)

### Models
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [roberta-large-mnli](https://huggingface.co/roberta-large-mnli)
- [gpt2](https://huggingface.co/gpt2)
- [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm)

## Citation

If you use this code, please cite the SHROOM task:
```bibtex
@inproceedings{shroom2024,
  title={SHROOM: Shared-task on Hallucinations and Observable Overgeneration Mistakes},
  booktitle={Proceedings of SemEval 2024},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub: [https://github.com/108nitish/HackLLM-Stuff](https://github.com/108nitish/HackLLM-Stuff)

---

**Developed for HackLLM - SemEval 2024 SHROOM Task**
