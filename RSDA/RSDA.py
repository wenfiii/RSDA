import asyncio
import os
import json
import torch
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ================= Configuration =================
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o"

# Local entropy model configuration
LOCAL_MODEL_ID = "path/to/local/model"

# Data file path configuration
INPUT_DATA_FILE = "input_data.json"
RESULT_FILE = "evaluation_results.json"
FINAL_DATASET_FILE = "final_dataset.json"
STATISTICS_FILE = "pipeline_statistics.json"

# Paper-defined parameters
COMPONENT_WEIGHTS = {
    'ins_weight': 0.15,
    'inp_weight': 0.35,
    'out_weight': 0.50
}

ALPHA = 0.4  # Cognitive uncertainty weight
BETA = 0.6  # Strategic weakness weight

# Dual threshold configuration
TAU_HIGH_PERCENTILE = 90  # Top 10% discard (high entropy noise)
TAU_LOW_PERCENTILE = 20  # Bottom 20% selective reserve

# Sensitivity thresholds
DELTA = {
    'ins': 0.1,
    'inp': 0.2,
    'out': 0.1
}


# ================= Local Entropy Calculator =================
class LocalEntropyCalculator:
    def __init__(self, model_name=LOCAL_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading local entropy model: {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).eval()
            print("[SUCCESS] Local Entropy Model Loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load local model: {e}")
            self.model = None

    def compute_entropy(self, text):
        """Calculate text entropy (Loss)"""
        if not self.model or not text:
            return 0.0
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss.item()
        return loss

    @staticmethod
    def normalize(val, min_v=0.50, max_v=4.00):
        """Normalize entropy value to [0, 1]"""
        return max(0.0, min(1.0, (val - min_v) / (max_v - min_v)))


# ================= Potential Entropy Calculator =================
class PotentialEntropyCalculator:
    def __init__(self, alpha=ALPHA, beta=BETA, component_weights=None):
        self.alpha = alpha
        self.beta = beta
        self.weights = component_weights or COMPONENT_WEIGHTS

    def calculate_strategy_score(self, scores):
        """Calculate strategy evaluation score"""
        gap_ins = 1 - scores.get('s_ins_tone', 0.0)
        gap_inp_depth = 1 - scores.get('s_inp_depth', 0.0)
        gap_inp_complex = 1 - scores.get('s_inp_complex', 0.0)
        gap_inp_avg = (gap_inp_depth + gap_inp_complex) / 2

        gap_out_cot = 1 - scores.get('s_out_cot', 0.0)
        gap_out_div = 1 - scores.get('s_out_div', 0.0)
        gap_out_dens = 1 - scores.get('s_out_dens', 0.0)
        gap_out_bg = 1 - scores.get('s_out_bg', 0.0)
        gap_out_avg = (gap_out_cot + gap_out_div + gap_out_dens + gap_out_bg) / 4

        ins_component = self.weights['ins_weight'] * gap_ins
        inp_component = self.weights['inp_weight'] * gap_inp_avg
        out_component = self.weights['out_weight'] * gap_out_avg

        strategy_score = ins_component + inp_component + out_component

        return {
            'strategy_score': strategy_score,
            'components': {
                'instruction': ins_component,
                'input': inp_component,
                'output': out_component
            },
            'gaps': {
                'ins_tone': gap_ins,
                'inp_depth': gap_inp_depth,
                'inp_complex': gap_inp_complex,
                'inp_avg': gap_inp_avg,
                'out_cot': gap_out_cot,
                'out_div': gap_out_div,
                'out_dens': gap_out_dens,
                'out_bg': gap_out_bg,
                'out_avg': gap_out_avg
            }
        }

    def calculate_potential_entropy(self, normalized_local_entropy, scores):
        """Calculate final potential entropy value"""
        strategy_result = self.calculate_strategy_score(scores)

        potential_entropy = (
                self.alpha * normalized_local_entropy +
                self.beta * strategy_result['strategy_score']
        )

        return {
            'potential_entropy': potential_entropy,
            'epistemic_component': self.alpha * normalized_local_entropy,
            'strategic_component': self.beta * strategy_result['strategy_score'],
            'strategy_details': strategy_result,
            'coefficients': {
                'alpha': self.alpha,
                'beta': self.beta,
                'weights': self.weights
            }
        }


# ================= Strategy Library Configuration =================
STRATEGY_PROMPTS_FILE = "strategy_prompts.json"


def load_strategy_prompts(filepath=STRATEGY_PROMPTS_FILE):
    """Load strategy library from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        strategy_prompts = {}
        for component in ['instruction', 'input', 'output']:
            if component in data:
                strategy_prompts[component] = {
                    int(k): v for k, v in data[component].items()
                }
            else:
                print(f"[WARN] Component '{component}' not found in strategy file")
                strategy_prompts[component] = {0: f"Keep {component} unchanged."}

        print(f"[SUCCESS] Loaded strategy prompts from {filepath}")
        return strategy_prompts

    except FileNotFoundError:
        print(f"[WARN] Strategy file not found: {filepath}")
        print("[INFO] Using default strategy prompts")
        return get_default_strategy_prompts()
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in strategy file: {e}")
        print("[INFO] Using default strategy prompts")
        return get_default_strategy_prompts()


def get_default_strategy_prompts():
    """Return default strategy library"""
    return {
        "instruction": {
            0: "Keep the instruction unchanged.",
            1: "Rewrite the instruction with a more encouraging, positive, and polite tone. Consider adding a helpful persona."
        },
        "input": {
            0: "Keep the input unchanged.",
            1: "Transform the input into a rich, story-driven scenario with concrete details and real-world context.",
            2: "Perform field migration: move the problem to a different domain to increase complexity."
        },
        "output": {
            0: "Rewrite output with step-by-step Chain-of-Thought reasoning. Make implicit steps explicit.",
            1: "Rewrite output with diverse solution methods. Provide at least 2 distinct approaches.",
            2: "Rewrite output to be more concise. Extract core principles and remove redundancy.",
            3: "Rewrite output with richer background. Add transitional semantics and clarify causal relationships."
        }
    }


STRATEGY_PROMPTS = None


def calculate_strategy_mark(scores, gaps, delta=DELTA):
    """Calculate strategy mark based on gaps and thresholds"""
    mark = [0, 0, 0]

    # Instruction
    if gaps['ins_tone'] > delta['ins']:
        mark[0] = 1

    # Input
    if gaps['inp_avg'] > delta['inp']:
        if gaps['inp_complex'] >= gaps['inp_depth']:
            mark[1] = 2
        else:
            mark[1] = 1

    # Output
    out_gaps_map = {
        0: gaps['out_cot'],
        1: gaps['out_div'],
        2: gaps['out_dens'],
        3: gaps['out_bg']
    }
    best_out_idx = max(out_gaps_map, key=out_gaps_map.get)
    if out_gaps_map[best_out_idx] > delta['out']:
        mark[2] = best_out_idx

    return mark


def construct_renovation_prompt(sample, mark):
    """Construct renovation prompt"""
    m_ins, m_inp, m_out = mark
    directives = []

    if m_ins != 0:
        directives.append(f"- INSTRUCTION: {STRATEGY_PROMPTS['instruction'][m_ins]}")
    if m_inp != 0:
        directives.append(f"- INPUT: {STRATEGY_PROMPTS['input'][m_inp]}")
    directives.append(f"- OUTPUT: {STRATEGY_PROMPTS['output'][m_out]}")

    directive_text = "\n".join(directives)

    prompt = f"""You are an expert Data Refurbisher. Rewrite the sample to improve quality based on directives.

CRITICAL RULES:
1. Return ONLY valid JSON: {{"instruction": "...", "input": "...", "output": "..."}}
2. NO markdown blocks (```json). NO explanations outside JSON.
3. For multi-solution/multi-step outputs: combine into ONE string using \\n separators.
4. Escape all quotes inside strings.

[Original]
Instruction: {sample['instruction']}
Input: {sample.get('input', '')}
Output: {sample['output']}

[Directives]
{directive_text}

Generate renovated JSON now:"""

    return prompt


def load_input_data(filepath):
    """Load input data file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[SUCCESS] Loaded {len(data)} samples from {filepath}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format: {e}")
        return []


def save_results(data, filename):
    """Save results to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"[ERROR] Failed to save: {e}")


def calculate_tau_thresholds(potential_entropies, high_percentile=TAU_HIGH_PERCENTILE,
                             low_percentile=TAU_LOW_PERCENTILE):
    """
    Calculate dual thresholds: tau_high (90th) and tau_low (20th)

    Data segmentation:
    - PE > tau_high (top 10%): High entropy noise zone, discard
    - tau_low < PE <= tau_high (middle 70%): Renovation zone, enter renovation process
    - PE <= tau_low (bottom 20%): Low potential zone, selective reserve
    """
    if not potential_entropies:
        return 0.0, 0.0

    tau_high = np.percentile(potential_entropies, high_percentile)
    tau_low = np.percentile(potential_entropies, low_percentile)

    return tau_high, tau_low


def select_reserve_samples(eval_results, tau_low):
    """
    Select high-quality original data from low potential zone (PE <= tau_low)

    Strategy:
    1. Filter all samples with PE <= tau_low
    2. Calculate strategic component scores for these samples
    3. Select samples with strategic scores below median (better quality original data)

    Returns:
        tuple: (reserved samples list, discarded samples list)
    """
    low_pe_samples = [r for r in eval_results if r['potential_entropy'] <= tau_low]

    if not low_pe_samples:
        return [], []

    low_pe_samples.sort(key=lambda x: x['potential_details']['strategic_component'])

    strategy_scores = [r['potential_details']['strategic_component'] for r in low_pe_samples]
    median_score = np.median(strategy_scores)

    reserve_samples = [
        r for r in low_pe_samples
        if r['potential_details']['strategic_component'] <= median_score
    ]

    discarded_low_quality = [
        r for r in low_pe_samples
        if r['potential_details']['strategic_component'] > median_score
    ]

    print(f"\n{'=' * 20} Low-PE Zone Analysis {'=' * 20}")
    print(f"Total low-PE samples (PE <= tau_low): {len(low_pe_samples)}")
    print(f"Strategy score median: {median_score:.4f}")
    print(f"Reserved high-quality samples: {len(reserve_samples)}")
    print(f"Discarded low-quality samples: {len(discarded_low_quality)}")

    return reserve_samples, discarded_low_quality


# ================= Main Pipeline =================
async def run_uldp_pipeline():
    print("=" * 60)
    print("ULDP Pipeline (Optimized with Dual-Threshold Strategy)")
    print("=" * 60)

    # 0. Load strategy library
    global STRATEGY_PROMPTS
    STRATEGY_PROMPTS = load_strategy_prompts()

    # 1. Load data
    samples = load_input_data(INPUT_DATA_FILE)
    if not samples:
        print("[ERROR] No valid samples loaded. Exiting.")
        return

    # 2. Initialize components
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    entropy_calc = LocalEntropyCalculator(LOCAL_MODEL_ID)
    potential_calc = PotentialEntropyCalculator()

    all_eval_results = []
    all_entropy_records = []

    # 3. Phase 1: Evaluate all samples
    print(f"\n{'=' * 20} Phase 1: Evaluation {'=' * 20}")
    for i, sample in enumerate(samples):
        print(f"\n[{i + 1}/{len(samples)}] Evaluating sample {i}...")

        # 3.1 Calculate local entropy
        raw_entropy = entropy_calc.compute_entropy(sample['output'])
        norm_entropy = entropy_calc.normalize(raw_entropy)

        all_entropy_records.append({
            "sample_id": i,
            "raw_entropy": raw_entropy,
            "normalized_entropy": norm_entropy
        })

        # 3.2 LLM evaluation
        eval_prompt = f"""Act as a strict data quality analyst. Return flat JSON with scores (0.00-1.00).

Evaluate:
Instruction: {sample['instruction']}
Input: {sample.get('input', '')}
Output: {sample['output']}

Criteria: s_ins_tone, s_inp_depth, s_inp_complex, s_out_cot, s_out_div, s_out_dens, s_out_bg.
Output JSON only."""

        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw_json = json.loads(res.choices[0].message.content)

            # Clean scores
            target_keys = ['s_ins_tone', 's_inp_depth', 's_inp_complex',
                           's_out_cot', 's_out_div', 's_out_dens', 's_out_bg']
            scores = {}
            for k in target_keys:
                if k in raw_json:
                    scores[k] = float(raw_json[k])
                else:
                    scores[k] = 0.5
                    print(f"[WARN] Missing {k}, using default 0.5")

        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            continue

        # 3.3 Calculate potential entropy
        potential_result = potential_calc.calculate_potential_entropy(norm_entropy, scores)

        # 3.4 Calculate strategy mark
        gaps = potential_result['strategy_details']['gaps']
        mark = calculate_strategy_mark(scores, gaps)

        # Save evaluation results
        eval_record = {
            "sample_id": i,
            "original_data": sample,
            "evaluation_scores": scores,
            "gaps": gaps,
            "strategy_mark": mark,
            "entropy": {
                "raw": raw_entropy,
                "normalized": norm_entropy
            },
            "potential_entropy": potential_result['potential_entropy'],
            "potential_details": potential_result
        }
        all_eval_results.append(eval_record)

        print(f"  PE: {potential_result['potential_entropy']:.4f}, Mark: {mark}")

    # 4. Calculate dual thresholds
    potential_values = [r['potential_entropy'] for r in all_eval_results]
    tau_high, tau_low = calculate_tau_thresholds(potential_values)

    print(f"\n{'=' * 20} Threshold Calculation {'=' * 20}")
    print(f"tau_high (90th percentile): {tau_high:.4f}")
    print(f"tau_low (20th percentile): {tau_low:.4f}")
    print(f"\nData Segmentation:")
    print(f"  - High Noise Zone (PE > {tau_high:.4f}): Top 10%, DISCARD")
    print(f"  - Renovation Zone ({tau_low:.4f} < PE <= {tau_high:.4f}): Middle 70%, RENOVATE")
    print(f"  - Low Potential Zone (PE <= {tau_low:.4f}): Bottom 20%, SELECTIVE RESERVE")

    # 5. Data segmentation
    high_noise_samples = [r for r in all_eval_results if r['potential_entropy'] > tau_high]
    renovation_samples = [r for r in all_eval_results
                          if tau_low < r['potential_entropy'] <= tau_high]
    reserve_samples, discarded_low_samples = select_reserve_samples(all_eval_results, tau_low)

    # 6. Phase 2: Renovate middle zone samples
    print(f"\n{'=' * 20} Phase 2: Renovation {'=' * 20}")
    all_renovated_data = []

    for record in renovation_samples:
        pe = record['potential_entropy']
        sample_id = record['sample_id']

        print(f"\n[{sample_id}] Renovating (PE={pe:.4f})...")

        renovate_prompt = construct_renovation_prompt(
            record['original_data'],
            record['strategy_mark']
        )

        try:
            renov_res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Data Refurbisher. Return JSON only."},
                    {"role": "user", "content": renovate_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            renovated = json.loads(renov_res.choices[0].message.content)

            renovated['meta'] = {
                'original_id': sample_id,
                'potential_entropy': pe,
                'strategy_mark': record['strategy_mark'],
                'zone': 'renovation'
            }

            all_renovated_data.append(renovated)
            print(f"  Renovation complete")

        except Exception as e:
            print(f"  Renovation failed: {e}")

    # 7. Save all results
    print(f"\n{'=' * 20} Saving Results {'=' * 20}")

    # 7.1 Evaluation report
    if all_eval_results:
        save_results(all_eval_results, RESULT_FILE)

    # 7.2 Final dataset (sorted by PE descending)
    if all_renovated_data:
        all_renovated_data.sort(key=lambda x: x['meta']['potential_entropy'], reverse=True)

    final_dataset = []
    for item in all_renovated_data:
        final_dataset.append(item)

    if reserve_samples:
        for r in reserve_samples:
            data_copy = r['original_data'].copy()
            data_copy['meta'] = {
                'original_id': r['sample_id'],
                'potential_entropy': r['potential_entropy'],
                'strategic_component': r['potential_details']['strategic_component'],
                'zone': 'reserve_high_quality'
            }
            final_dataset.append(data_copy)

    if final_dataset:
        save_results(final_dataset, FINAL_DATASET_FILE)

    # 7.3 Statistics
    statistics = {
        "total_samples": len(samples),
        "evaluated_samples": len(all_eval_results),
        "thresholds": {
            "tau_high": tau_high,
            "tau_low": tau_low,
            "high_percentile": TAU_HIGH_PERCENTILE,
            "low_percentile": TAU_LOW_PERCENTILE
        },
        "segmentation": {
            "high_noise_zone": {
                "count": len(high_noise_samples),
                "percentage": len(high_noise_samples) / len(all_eval_results) * 100,
                "action": "discarded"
            },
            "renovation_zone": {
                "count": len(renovation_samples),
                "percentage": len(renovation_samples) / len(all_eval_results) * 100,
                "renovated": len(all_renovated_data),
                "action": "renovated"
            },
            "low_potential_zone": {
                "total_count": len([r for r in all_eval_results if r['potential_entropy'] <= tau_low]),
                "percentage": len([r for r in all_eval_results if r['potential_entropy'] <= tau_low]) / len(
                    all_eval_results) * 100,
                "reserved_high_quality": len(reserve_samples),
                "discarded_low_quality": len(discarded_low_samples),
                "action": "selective_reserve"
            }
        },
        "final_dataset_composition": {
            "renovated_samples": len(all_renovated_data),
            "reserved_original_samples": len(reserve_samples),
            "total_final_samples": len(all_renovated_data) + len(reserve_samples),
            "discarded_samples": len(high_noise_samples) + len(discarded_low_samples),
            "retention_rate": (len(all_renovated_data) + len(reserve_samples)) / len(all_eval_results) * 100
        }
    }
    save_results(statistics, STATISTICS_FILE)


if __name__ == "__main__":
    asyncio.run(run_uldp_pipeline())