# Knowledge Triple Extraction from Text

This project fine-tunes small language models to extract structured knowledge triples from natural language text, comparing performance across different model sizes.

## Goal

Extract knowledge triples from text in the format `Subject | Relation | Object`, enabling structured information retrieval from unstructured text. The project evaluates how well different small language models can learn this task through fine-tuning.

## Data

**Source**: WebNLG dataset from Hugging Face (`GEM/web_nlg`) 
- Training data: 10% random sample of the full training set
- Test data: 300 samples from 10% of the test set
- Format: Text descriptions with corresponding knowledge triples

The WebNLG dataset contains structured data-to-text generation examples, which we reverse-engineer to create a text-to-knowledge extraction task.

## Models Fine-tuned

Five small language models were fine-tuned and compared:
- **Gemma 270M** (unsloth/gemma-3-270m-it)
- **Gemma 2B** (unsloth/gemma-3-2b-it)
- **Gemma 4B** (unsloth/gemma-3-4b-it-unsloth-bnb-4bit)
- **Llama 1B** (unsloth/Llama-3.2-1B-Instruct-bnb-4bit)
- **Llama 3B** (unsloth/Llama-3.2-3B-Instruct-bnb-4bit)

## Fine-tuning Setup

**Framework**: Unsloth for efficient fine-tuning
**Technique**: LoRA (Low-Rank Adaptation)
- LoRA rank: 128
- LoRA alpha: 128
- Target modules: All attention and MLP layers
- Training epochs: 4
- Max steps: 50
- Learning rate: 7e-5
- Batch size: 4 (with gradient accumulation)

**Optimisations**:
- 4-bit quantisation for memory efficiency
- Gradient checkpointing
- Response-only training (only train on model outputs)

## Prompt Format

The models were trained with a detailed system prompt specifying the exact formatting rules:

### Complete System Prompt
```
You are tasked with extracting knowledge triples from text. Each triple must follow this exact structure: 'Subject | Relation | Object'

Rules for formatting triples:
Subject:
A single entity name.
Replace spaces with underscores.
Can be a person, place, organization, or other entity.
Relation:
Always in camelCase.
Represents the property or relationship between subject and object.
Object:
Dates → Normalize to ISO format "YYYY-MM-DD".
Degree or institution → Format as "Institution, Degree Year".
Places (cities, states, countries, regions, airports) → Replace spaces with underscores, preserve commas and accents, no quotes.
Entities (people, organizations, missions, artists, genres, events, equipment/vehicles) → Replace spaces with underscores, preserve hyphens and special characters, no quotes.
Entities with qualifiers or disambiguation → Keep parentheses and underscores, no quotes.
Titles, roles, or string values → Always in double quotes.
Lists of multiple entities → Combine into a single, comma-separated string wrapped in double quotes.
Numeric values → Include as-is, no quotes.
Output:
Return all triples as a Python list of strings.
Each triple must strictly follow the Subject | Relation | Object format.
Ensure consistency: underscores for multi-word names, quotes where required, ISO dates, numeric values as-is.
```

## Evaluation Method

**Fuzzy Matching Evaluation**: Since exact string matching is too strict for this task, we implemented component-wise fuzzy matching:

### Matching Thresholds
- **Entity matching**: 90% similarity (strict)
- **Relation matching**: 80% similarity (moderate)
- **Value matching**: 80% similarity (moderate)

### Scoring Method
1. Parse each triple into entity, relation, and value components
2. Use fuzzy string matching (Levenshtein distance) for each component
3. Handle numeric values with tolerance-based comparison
4. Calculate precision, recall, and F1 scores
5. Match ground truth triples to predicted triples using best-match assignment

## Results

### Performance Comparison

| Model | Avg F1 Score | Avg Precision | Avg Recall |
|-------|-------------|---------------|------------|
| Gemma 270M | 0.153 | 0.159 | 0.164 |
| Gemma 2B | 0.316 | 0.325 | 0.319 |
| Gemma 4B | 0.481 | 0.483 | 0.489 |
| Llama 1B | 0.407 | 0.421 | 0.403 |
| Llama 3B | 0.464 | 0.465 | 0.479 |

### Key Findings

- **Gemma 4B achieved the highest performance** with F1 score of 0.481, precision of 0.483, and recall of 0.489
- **Clear scaling effect**: Larger models within the same family (Gemma) show consistent improvement
- **Interesting cross-family comparison**: Llama 1B (F1: 0.407) outperforms Gemma 2B (F1: 0.316), suggesting model architecture and training quality matter beyond just parameter count
- **Balanced performance**: Most models achieve similar precision and recall scores
- The fuzzy matching approach provides more realistic evaluation than exact string matching
- Fine-tuning small models can achieve reasonable performance on knowledge extraction

## Project Structure

```
├── evaluateSmallModels.ipynb  # Evaluation script with fuzzy matching
├── finetuneSmallModels.ipynb  # Fine-tuning script
├── data/                      # Test results for each model
│   ├── gemma270m-test_results.csv
│   ├── gemma2b-test_results.csv
│   ├── gemma4b-test_results.csv
│   ├── llama1b-test_results.csv
│   └── llama3b-test_results.csv
└── README.md
```



## Dependencies

- unsloth
- transformers
- datasets
- pandas
- numpy
- fuzzywuzzy
- matplotlib
- torch

## Future Improvements

### Hyperparameter Tuning
- **LoRA configuration**: Experiment with different rank values (256, 512) and alpha ratios
- **Learning rates**: Test different schedules and rates 
- **Training duration**: Longer training with more epochs and careful monitoring for overfitting
- **Batch sizes**: Larger effective batch sizes through gradient accumulation

### Model Scaling
- **Larger models**: Test 7B and 13B parameter models (Llama 3.1-7B, Gemma 2-9B, Gemma 2-27B)
- **Model architecture comparison**: Compare different families (Qwen, Mistral, Phi)
- **Full fine-tuning vs LoRA**: Evaluate whether full parameter training improves results

### Dataset Improvements
- **Cleaner preprocessing**: Better handling of malformed triples and edge cases
- **Balanced sampling**: Ensure equal representation across different triple types and domains
- **Numeric value handling**: Create specific training examples for various numeric formats (percentages, measurements, currencies)

### Enhanced Instructions
- **Error-specific guidance**: Add rules for common mistakes identified during evaluation
- **Context-aware formatting**: Instructions for handling ambiguous entities and relations
- **Validation steps**: Teach models to self-check output format before finalising


Combined, these improvements could potentially achieve F1 scores of 0.7+ on this knowledge extraction task.