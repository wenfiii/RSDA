import asyncio
import os
import json
import torch
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_URL = "https://"
MODEL_NAME = "gpt-4o"

LOCAL_MODEL_ID = "Qwen3-0.6B"

DELTA = {'ins': 0, 'inp': 0, 'out': 0}
RESULT_FILE = "evaluation_report.json"
RENOVATION_FILE = "uldp.json"
ENTROPY_FILE = "local_entropy_values.json"
POTENTIAL_WEIGHTS = {
    'local_entropy': 0.5,
    'strategy_score': 0.5,
    'ins_weight': 0.2,
    'inp_weight': 0.3,
    'out_weight': 0.5
}

TEST_SAMPLES = [
  {
    "instruction": "Name three tools that can be used to backup data.",
    "input": "",
    "output": "Three tools that can be used to backup data are cloud storage, external hard drives, and USB flash drives."
  }
]


class LocalEntropyCalculator:
    def __init__(self, model_name=LOCAL_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local entropy model: {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=self.device, trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).eval()
            print("Local Entropy Model Loaded.")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self.model = None

    def compute_entropy(self, text):
        if not self.model or not text:
            return 0.0
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss.item()
        return loss

    @staticmethod
    def normalize(val, min_v=0.50, max_v=4.00):
        return max(0.00, min(1.00, (val - min_v) / (max_v - min_v)))


class PotentialEntropyCalculator:

    def __init__(self, weights=None):
        self.weights = weights or POTENTIAL_WEIGHTS
    def calculate_strategy_score(self, scores):

        gap_ins_tone = 1 - scores.get('s_ins_tone', 0.0)
        gap_inp_depth = 1 - scores.get('s_inp_depth', 0.0)
        gap_inp_complex = 1 - scores.get('s_inp_complex', 0.0)
        gap_out_cot = 1 - scores.get('s_out_cot', 0.0)
        gap_out_div = 1 - scores.get('s_out_div', 0.0)
        gap_out_dens = 1 - scores.get('s_out_dens', 0.0)
        gap_out_bg = 1 - scores.get('s_out_bg', 0.0)

        ins_gap_score = gap_ins_tone * self.weights['ins_weight']

        inp_gap_score = (gap_inp_depth + gap_inp_complex) * self.weights['inp_weight']

        out_gap_score = (gap_out_cot + gap_out_div + gap_out_dens + gap_out_bg) * self.weights['out_weight']

        raw_strategy_score = ins_gap_score + inp_gap_score + out_gap_score

        return {
            'raw': raw_strategy_score,
            'normalized': raw_strategy_score,
            'components': {
                'instruction': ins_gap_score,
                'input': inp_gap_score,
                'output': out_gap_score
            },
            'gaps': {
                'ins_tone': gap_ins_tone,
                'inp_depth': gap_inp_depth,
                'inp_complex': gap_inp_complex,
                'out_cot': gap_out_cot,
                'out_div': gap_out_div,
                'out_dens': gap_out_dens,
                'out_bg': gap_out_bg
            }
        }

    def calculate_potential_entropy(self, normalized_local_entropy, scores):

        strategy_result = self.calculate_strategy_score(scores)
        potential_entropy = (
                self.weights['local_entropy'] * normalized_local_entropy +
                self.weights['strategy_score'] * strategy_result['normalized']
        )

        return {
            #'potential_entropy': potential_entropy,
            #'local_entropy_component': self.weights['local_entropy'] * normalized_local_entropy,
            'strategy_component': self.weights['strategy_score'] * strategy_result['normalized'],
            'strategy_details': strategy_result,
            'weights_used': self.weights
        }

STRATEGY_PROMPTS = {
    "instruction": {
        0: "This part of ...",
        1: "Rewrite the instruction to use a more encouraging, positive, and polite tone, possibly adding a persona. For the input section of this data, introduce a strong, positive sentiment."
    },
    "input": {
        0: "This part of the input remains unchanged without modification.",
        1: "Transform the input into a rich, story-driven scenario with concrete details. Supplement the background setting to focus more on real-world value.",
        2: "Perform a field migration operation. Move the problem background to other disciplines or social contexts (e.g., from generic math to physics or economics) to increase complexity."
    },
    "output": {
        0: "Rewrite the data output section, breaking down the problem-solving process into consecutive steps. Clearly identify key connections and avoid skipping steps or implicit reasoning.",
        1: "Rewrite the data output section to implement diversified solution methods. Expand the original single solution by providing at least two distinct approaches (e.g., algebraic vs. geometric), broadening the perspective.",
        2: "Rewrite the data output section to filter out irrelevant text. Extract the principles or formulas involved. Return to the essence of the problem, retaining only key data and logic.",
        3: "Rewrite the data output section to perform semantic completion. Ensure logical connections between sentences, supplement transitional semantics, and clarify causal relationships. Add precise details if context is vague."
    }
}


def calculate_mark_only(scores):
    gaps = {k: 1.0 - v for k, v in scores.items()}
    mark = [0, 0, 0]
    if gap_ins_tone > DELTA['ins']:
        mark[0] = 1

    gap_inp_depth = gaps.get('s_inp_depth', 0)
    gap_inp_complex = gaps.get('s_inp_complex', 0)
    max_inp_gap = max(gap_inp_depth, gap_inp_complex)
    if max_inp_gap > DELTA['inp']:
        if gap_inp_complex >= gap_inp_depth:
            mark[1] = 2
        else:
            mark[1] = 1

    out_gaps_map = {
        0: gap_out_cot,
        1: gap_out_div,
        2: gap_out_dens,
        3: gap_out_bg
    }
    best_strat_idx = max(out_gaps_map, key=out_gaps_map.get)
    if out_gaps_map[best_strat_idx] > DELTA['out']:
        mark[2] = best_strat_idx
    else:
        mark[2] = 0

    return mark, gaps


def construct_renovation_prompt(sample, mark):
    m_ins, m_inp, m_out = mark
    strat_ins = STRATEGY_PROMPTS["instruction"].get(m_ins, "")
    strat_inp = STRATEGY_PROMPTS["input"].get(m_inp, "")
    strat_out = STRATEGY_PROMPTS["output"].get(m_out, "")
    directives = []
    if m_ins != 0: directives.append(f"- INSTRUCTION: {strat_ins}")
    if m_inp != 0: directives.append(f"- INPUT: {strat_inp}")
    directives.append(f"- OUTPUT: {strat_out}")
    directive_text = "\n".join(directives)
    prompt = f"""You are an expert Data Refurbisher. Your goal is to rewrite the given data sample to improve its quality based on specific directives.

CRITICAL FORMATTING RULES:
1. Return ONLY a single valid JSON object. No markdown code blocks (```json), no explanations.
2. The JSON keys must be exactly: "instruction", "input", "output".
3. **Handling Multi-Turn/Multi-Solution Content**: Even if the renovation directive asks for multiple perspectives, diverse solutions, or distinct steps, you must combine them into a **SINGLE string** for the "output" field. like "Solution 1:\\nFirst approach...\\n\\nSolution 2:\\nSecond approach..."
   - Use `\\n` or `\\n\\n` to separate different sections or paragraphs.
   - DO NOT output a JSON list or array (e.g., [Solution1, Solution2] is FORBIDDEN).
   - Ensure all double quotes inside the text are escaped (e.g., \\").

[Original Data]
Instruction: {sample['instruction']}
Input: {sample.get('input', '')}
Output: {sample['output']}

[Renovation Directives]
{directive_text}

[Target Format Example]
{{
    "instruction": "...",
    "input": "...",
    "output":"
}}

Please generate the renovated data strictly adhering to the JSON format above.
"""
    return prompt


def save_evaluation_results(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n[Success]: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"\n[Error]: {e}")

async def run_hybrid_pipeline():
    print(f"ULDP Running with Potential Entropy Calculation...\n")
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    entropy_calc = LocalEntropyCalculator(LOCAL_MODEL_ID)
    potential_calc = PotentialEntropyCalculator()

    all_eval_results = []
    all_renovated_data = []
    all_entropy_values = []

    for i, sample in enumerate(TEST_SAMPLES):

        #print(f"\n{'=' * 20} Step 0:  {'=' * 20}")
        #raw_entropy = entropy_calc.compute_entropy(sample['output'])
        #norm_entropy = entropy_calc.normalize(raw_entropy)

        #print(f"(Loss): {raw_entropy:.4f}")
        #print(f" {norm_entropy:.4f}")

        #entropy_record = {
        #    "sample_id": i,
            # "instruction": sample['instruction'],
        #    "raw_entropy": raw_entropy,
        #    "normalized_entropy": norm_entropy,
        #    "output_length": len(sample['output'])
        #}
        #all_entropy_values.append(entropy_record)

        print(f"\n{'=' * 20} Step 1: LLM  {'=' * 20}")

        eval_prompt = f"""Act as a strict data quality analyst. Return JSON.
Evaluate: 
Ins: {sample['instruction']}
Inp: {sample['input']}
Out: {sample['output']}

Criteria (0.00-1.00): s_ins_tone, s_inp_depth, s_inp_complex, s_out_cot, s_out_div, s_out_dens, s_out_bg.
Output purely a flat JSON object mapping criteria to float scores.
"""

        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw_json = json.loads(res.choices[0].message.content)

            scores = {}
            target_keys = ['s_ins_tone', 's_inp_depth', 's_inp_complex',
                           's_out_cot', 's_out_div', 's_out_dens', 's_out_bg']

            for k in target_keys:
                if k in raw_json and isinstance(raw_json[k], (int, float)):
                    scores[k] = float(raw_json[k])

            if not scores:
                for v in raw_json.values():
                    if isinstance(v, dict):
                        for k in target_keys:
                            if k in v and isinstance(v[k], (int, float)):
                                scores[k] = float(v[k])

            for k in target_keys:
                if k not in scores:
                    scores[k] = 0.50

            print( json.dumps(scores, indent=2))

        except Exception as e:
            print(f"API Error: {e}")
            continue
        #potential_result = potential_calc.calculate_potential_entropy(norm_entropy, scores)
        #components = potential_result['strategy_details']['components']
        #gaps = potential_result['strategy_details']['gaps']

        print(f"\n{'=' * 20} Step 2:  (Mark) {'=' * 20}")
        mark, gaps = calculate_mark_only(scores)
        print(f"ark: {mark}")

        for k, v in gaps.items():
            print(f"  {k}: {v:.2f}")
        print(gap_ins_tone,gap_inp_depth,gap_inp_complex,gap_out_cot,gap_out_div,gap_out_dens,gap_out_bg)

        print(f"  Instruction: {mark[0]} ({'Polite' if mark[0] == 1 else 'Keep'})")
        print(f"  Input:       {mark[1]} ({'Story' if mark[1] == 1 else ('Complex' if mark[1] == 2 else 'Keep')})")
        print(f"  Output:      {mark[2]} (0=CoT, 1=Div, 2=Dens, 3=Bg)")

        result_record = {
            "sample_id": i,
            "original_data": sample,
            "evaluation_scores": scores,
            #"metric_gaps": gaps,
            "final_strategy_mark": mark,
            #"entropy": {
            #    "raw": raw_entropy,
            #    "normalized": norm_entropy
            #},
            #"potential_entropy": potential_result
        }
        all_eval_results.append(result_record)

        print(f"\n{'=' * 20} Step 3: renovation {'=' * 20}")
        renovate_prompt_content = construct_renovation_prompt(sample, mark)

        try:
            renov_res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Data Refurbisher. Return JSON only."},
                    {"role": "user", "content": renovate_prompt_content}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            final_data = json.loads(renov_res.choices[0].message.content)

            #final_data['original_id'] = i
            #final_data['potential_entropy'] = potential_result['potential_entropy']
            #final_data['strategy_mark'] = mark

            all_renovated_data.append(final_data)

            print(json.dumps(final_data, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"ERROR: {e}")

    if all_eval_results:
        save_evaluation_results(all_eval_results, RESULT_FILE)
        print(f"Done: {len(all_eval_results)} ")
    if all_renovated_data:
        all_renovated_data.sort(key=lambda x: x.get('potential_entropy', 0), reverse=True)
        save_evaluation_results(all_renovated_data, RENOVATION_FILE)
    if all_entropy_values:
        save_evaluation_results(all_entropy_values, ENTROPY_FILE)
    else:
        print("ERROR!")


if __name__ == "__main__":
    asyncio.run(run_hybrid_pipeline())