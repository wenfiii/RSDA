\# RSDA Pipeline - Restoring Stale Data Affinity



\## Architecture



```

Input Dataset

&nbsp;   ↓

\[Phase 1: Evaluation]

&nbsp;   ├─ Local Entropy Calcul# RSDA Pipeline - Restoring Stale Data Affinity



\## Architecture



```

Input Dataset

&nbsp;   ↓

\[Phase 1: Evaluation]

&nbsp;   ├─ Local Entropy Calculation

&nbsp;   ├─ LLM Quality Scoring

&nbsp;   └─ Potential Entropy Computation

&nbsp;   ↓

\[Dual-Threshold Segmentation]

&nbsp;   ├─ High Noise → Discard

&nbsp;   ├─ Renovation Zone → Phase 2

&nbsp;   └─ Low Potential → Selective Reserve

&nbsp;   ↓

\[Phase 2: Renovation]

&nbsp;   ├─ Strategy Selection

&nbsp;   ├─ LLM-based Rewriting

&nbsp;   └─ Quality Enhancement

&nbsp;   ↓

Final Dataset (Renovated + Reserved)

```

\## Installation



\### Requirements

```bash

pip install torch transformers openai numpy asyncio

```



\## Configuration



\### File Paths

```python

INPUT\_DATA\_FILE = "path/to/input.json"

FINAL\_DATASET\_FILE = "output\_dataset.json"

STATISTICS\_FILE = "pipeline\_stats.json"

```



\### Strategy Configuration

Edit `strategy\_prompts.json` to customize renovation strategies:

```json

{

&nbsp; "instruction": {

&nbsp;   "0": "Keep unchanged",

&nbsp;   "1": "Rewrite with positive tone..."

&nbsp; },

&nbsp; "input": {

&nbsp;   "0": "Keep unchanged",

&nbsp;   "1": "Add rich context...",

&nbsp;   "2": "Perform domain migration..."

&nbsp; },

&nbsp; "output": {

&nbsp;   "0": "Add Chain-of-Thought...",

&nbsp;   "1": "Provide diverse solutions...",

&nbsp;   "2": "Make more concise...",

&nbsp;   "3": "Enrich background..."

&nbsp; }

}

```

\## Input Data Format



```json

\[

&nbsp; {

&nbsp;   "instruction": "Explain the concept...",

&nbsp;   "input": "Given context...",

&nbsp;   "output": "The explanation is..."

&nbsp; }

]

```



\## Output Files



\### 1. \*\*Final Dataset\*\* (`dolly\_u.json`)

Combined renovated and reserved high-quality samples with metadata:

```json

{

&nbsp; "instruction": "...",

&nbsp; "input": "...",

&nbsp; "output": "...",

&nbsp; "meta": {

&nbsp;   "original\_id": 0,

&nbsp;   "potential\_entropy": 0.456,

&nbsp;   "strategy\_mark": \[1, 2, 0],

&nbsp;   "zone": "renovation"

&nbsp; }

}

```



\### 2. \*\*Evaluation Report\*\* (`dolly\_evaluation.json`)

Detailed analysis for each sample including scores, gaps, and entropy metrics.



\### 3. \*\*Statistics\*\* (`dolly\_pipeline.json`)

Pipeline performance metrics:

```json

{

&nbsp; "total\_samples": 1000,

&nbsp; "segmentation": {

&nbsp;   "high\_noise\_zone": {"count": 100, "percentage": 10.0},

&nbsp;   "renovation\_zone": {"count": 700, "renovated": 680},

&nbsp;   "low\_potential\_zone": {"reserved\_high\_quality": 100}

&nbsp; },

&nbsp; "final\_dataset\_composition": {

&nbsp;   "total\_final\_samples": 780,

&nbsp;   "retention\_rate": 78.0

&nbsp; }

}

```



\## Usage



\### Basic Execution

```bash

python uldp\_pipeline.py

```



\### Environment Variables

```bash

export OPENAI\_API\_KEY="your-api-key"

python uldp\_pipeline.py

```



\### Custom Configuration

Modify the configuration section in the script:

```python

API\_KEY = "your-api-key"

BASE\_URL = "https://api.openai.com/v1"

MODEL\_NAME = "gpt-4o"

LOCAL\_MODEL\_ID = "path/to/local/model"

```

ation

&nbsp;   ├─ LLM Quality Scoring

&nbsp;   └─ Potential Entropy Computation

&nbsp;   ↓

\[Dual-Threshold Segmentation]

&nbsp;   ├─ High Noise → Discard

&nbsp;   ├─ Renovation Zone → Phase 2

&nbsp;   └─ Low Potential → Selective Reserve

&nbsp;   ↓

\[Phase 2: Renovation]

&nbsp;   ├─ Strategy Selection

&nbsp;   ├─ LLM-based Rewriting

&nbsp;   └─ Quality Enhancement

&nbsp;   ↓

Final Dataset (Renovated + Reserved)

```

\## Installation



\### Requirements

```bash

pip install torch transformers openai numpy asyncio

```



\## Configuration



\### File Paths

```python

INPUT\_DATA\_FILE = "path/to/input.json"

FINAL\_DATASET\_FILE = "output\_dataset.json"

STATISTICS\_FILE = "pipeline\_stats.json"

```



\### Strategy Configuration

Edit `strategy\_prompts.json` to customize renovation strategies:

```json

{

&nbsp; "instruction": {

&nbsp;   "0": "Keep unchanged",

&nbsp;   "1": "Rewrite with positive tone..."

&nbsp; },

&nbsp; "input": {

&nbsp;   "0": "Keep unchanged",

&nbsp;   "1": "Add rich context...",

&nbsp;   "2": "Perform domain migration..."

&nbsp; },

&nbsp; "output": {

&nbsp;   "0": "Add Chain-of-Thought...",

&nbsp;   "1": "Provide diverse solutions...",

&nbsp;   "2": "Make more concise...",

&nbsp;   "3": "Enrich background..."

&nbsp; }

}

```

\## Input Data Format



```json

\[

&nbsp; {

&nbsp;   "instruction": "Explain the concept...",

&nbsp;   "input": "Given context...",

&nbsp;   "output": "The explanation is..."

&nbsp; }

]

```



\## Output Files



\### 1. \*\*Final Dataset\*\* (`dolly\_u.json`)

Combined renovated and reserved high-quality samples with metadata:

```json

{

&nbsp; "instruction": "...",

&nbsp; "input": "...",

&nbsp; "output": "...",

&nbsp; "meta": {

&nbsp;   "original\_id": 0,

&nbsp;   "potential\_entropy": 0.456,

&nbsp;   "strategy\_mark": \[1, 2, 0],

&nbsp;   "zone": "renovation"

&nbsp; }

}

```



\### 2. \*\*Evaluation Report\*\* (`dolly\_evaluation.json`)

Detailed analysis for each sample including scores, gaps, and entropy metrics.



\### 3. \*\*Statistics\*\* (`dolly\_pipeline.json`)

Pipeline performance metrics:

```json

{

&nbsp; "total\_samples": 1000,

&nbsp; "segmentation": {

&nbsp;   "high\_noise\_zone": {"count": 100, "percentage": 10.0},

&nbsp;   "renovation\_zone": {"count": 700, "renovated": 680},

&nbsp;   "low\_potential\_zone": {"reserved\_high\_quality": 100}

&nbsp; },

&nbsp; "final\_dataset\_composition": {

&nbsp;   "total\_final\_samples": 780,

&nbsp;   "retention\_rate": 78.0

&nbsp; }

}

```



\## Usage



\### Basic Execution

```bash

python uldp\_pipeline.py

```



\### Environment Variables

```bash

export OPENAI\_API\_KEY="your-api-key"

python uldp\_pipeline.py

```



\### Custom Configuration

Modify the configuration section in the script:

```python

API\_KEY = "your-api-key"

BASE\_URL = "https://api.openai.com/v1"

MODEL\_NAME = "gpt-4o"

LOCAL\_MODEL\_ID = "path/to/local/model"

```



