# RSDA Pipeline - Restoring Stale Data Affinity

## Architecture

```
Input Dataset
    ↓
[Phase 1: Evaluation]
    ├─ Local Entropy Calculation (Qwen Model)
    ├─ LLM Quality Scoring (GPT-4)
    └─ Potential Entropy Computation
    ↓
[Dual-Threshold Segmentation]
    ├─ High Noise → Discard
    ├─ Renovation Zone → Phase 2
    └─ Low Potential → Selective Reserve
    ↓
[Phase 2: Renovation]
    ├─ Strategy Selection
    ├─ LLM-based Rewriting
    └─ Quality Enhancement
    ↓
Final Dataset (Renovated + Reserved)
```

## Installation

### Requirements
```bash
pip install torch transformers openai numpy asyncio
```

## Configuration

### File Paths
```python
INPUT_DATA_FILE = pathtoinput.json
FINAL_DATASET_FILE = output_dataset.json
STATISTICS_FILE = pipeline_stats.json
```

### Strategy Configuration
Edit `strategy_prompts.json` to customize renovation strategies:
```json
{
  "instruction": {
    "0": "Keep unchanged",
    "1": "Rewrite with positive tone..."
  },
  "input": {
    "0": "Keep unchanged",
    "1": "Add rich context...",
    "2": "Perform domain migration..."
  },
  "output": {
    "0": "Add Chain-of-Thought...",
    "1": "Provide diverse solutions...",
    "2": "Make more concise...",
    "3": "Enrich background..."
  }
}
```

## Input Data Format

```json
[
  {
    "instruction": "Explain the concept...",
    "input": "Given context...",
    "output": "The explanation is..."
  }
]
```

## Output Files

### 1. **Final Dataset** (`dolly_u.json`)
Combined renovated and reserved high-quality samples with metadata:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "meta": {
    "original_id": 0,
    "potential_entropy": 0.456,
    "strategy_mark": [1, 2, 0],
    "zone": "renovation"
  }
}
```

### 2. **Evaluation Report** (`dolly_evaluation.json`)
Detailed analysis for each sample including scores, gaps, and entropy metrics.

### 3. **Statistics** (`dolly_pipeline.json`)
Pipeline performance metrics:
```json
{
  "total_samples": 1000,
  "segmentation": {
    "high_noise_zone": {"count": 100, "percentage": 10.0},
    "renovation_zone": {"count": 700, "renovated": 680},
    "low_potential_zone": {"reserved_high_quality": 100}
  },
  "final_dataset_composition": {
    "total_final_samples": 780,
    "retention_rate": 78.0
  }
}
```

## Usage

### Basic Execution
```bash
python RSDA.py
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
python uldp_pipeline.py
```

### Custom Configuration
Modify the configuration section in the script:
```python
API_KEY = "your-api-key"
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o"
LOCAL_MODEL_ID = "path/to/local/model"
```
