import json
import pandas as pd
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import os

# ================= 配置区域 =================
# API 配置
API_KEY = os.getenv("OPENAI_API_KEY", "sk-dwyhno2Lan773h3YPtOArk86AV8nPiMU443qWr97qXFN3afk")
BASE_URL = "https://api.n1n.ai/v1"
MODEL_NAME = "gpt-4o-mini"

# 文件路径配置
INPUT_FILE = "dolly.jsonl"  # 原始dolly数据
ALPACA_FULL_FILE = "dolly_alpaca.json"  # 转换后的完整alpaca格式
SCORED_FILE = "dolly_score_low.csv"  # 打分后的中间文件
OUTPUT_FILE = "dolly_bad.json"  # 最终筛选后的低质量数据

# 筛选参数
TOP_K = 1000  # 筛选前1000条低质量数据

# 并发控制
MAX_CONCURRENT = 10
RETRY_TIMES = 3


# ============================================


def load_dolly_data(filepath):
    """
    加载dolly的jsonl格式数据并转换为alpaca格式

    Dolly格式:
    {
        "instruction": "...",
        "context": "...",  # 可选
        "response": "...",
        "category": "..."
    }

    Alpaca格式:
    {
        "instruction": "...",
        "input": "...",  # 对应dolly的context
        "output": "..."  # 对应dolly的response
    }
    """
    print(f"📂 加载Dolly数据: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 转换为alpaca格式
                alpaca_item = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("context", ""),  # dolly的context映射到alpaca的input
                    "output": item.get("response", ""),  # dolly的response映射到alpaca的output
                }
                data.append(alpaca_item)

    print(f"✅ 加载了 {len(data)} 条数据\n")
    return data


def build_prompt(item):
    """构建评分 Prompt"""
    instruction = item.get('instruction', '')
    inp = item.get('input', '')
    output = item.get('output', '')

    prompt = f"""You are an expert judge evaluating the quality of an AI assistant's response.

Instruction: {instruction}
Input: {inp}
Response: {output}

Rate the Response on a scale of 1 to 5 (5 being best). 
Criteria: 
- Accuracy: Is the information correct?
- Helpfulness: Does it address the instruction?
- Clarity: Is it well-structured and easy to understand?

Output ONLY a single number between 1 and 5 (e.g., 5).
Do not include any explanation or additional text.

Score:"""

    return prompt


async def score_single_sample(client, item, semaphore, index):
    """
    对单个样本进行评分（带重试机制）
    """
    prompt = build_prompt(item)

    async with semaphore:
        for attempt in range(RETRY_TIMES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict data quality evaluator. Output only decimal numbers with 2 decimal places (e.g., 3.47)."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=10,
                    timeout=30.0
                )

                generated_text = response.choices[0].message.content.strip()

                try:
                    score = float(generated_text)
                    score = max(0.00, min(4.99, score))
                    score = round(score, 2)
                except ValueError:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', generated_text)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.00, min(4.99, score))
                        score = round(score, 2)
                    else:
                        score = 0.00

                return index, score

            except Exception as e:
                if attempt == RETRY_TIMES - 1:
                    print(f"\n样本 {index} 评分失败 (已重试 {RETRY_TIMES} 次): {e}")
                    return index, 1.0
                await asyncio.sleep(1)


async def score_all_samples(data):
    """
    批量评分所有样本
    """
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=2
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"开始评分 {len(data)} 个样本...")
    print(f"并发数: {MAX_CONCURRENT}, 模型: {MODEL_NAME}\n")

    tasks = [
        score_single_sample(client, item, semaphore, i)
        for i, item in enumerate(data)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="评分进度")

    results.sort(key=lambda x: x[0])
    scores = [score for _, score in results]

    return scores


def filter_and_save_low_quality(scored_file, output_file, top_k):
    """
    筛选低质量数据并保存
    """
    df = pd.read_csv(scored_file)

    print(f"\n{'=' * 50}")
    print(f"数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  平均分数: {df['quality_score'].mean():.2f}")
    print(f"  分数范围: {df['quality_score'].min():.2f} - {df['quality_score'].max():.2f}")

    # 统计各分数段分布
    print(f"\n  分数分布:")
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
    df['score_range'] = pd.cut(df['quality_score'], bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        count = len(df[df['score_range'] == label])
        print(f"    {label} 分: {count} 个 ({count / len(df) * 100:.1f}%)")
    print(f"{'=' * 50}\n")

    # 按分数升序排序(最低分在前)
    df_sorted = df.sort_values(by="quality_score", ascending=True)

    # 取前K个最差样本
    top_k_df = df_sorted.head(top_k)

    print(f"筛选后 {top_k} 个低质量样本:")
    print(f"  分数范围: {top_k_df['quality_score'].min():.2f} - {top_k_df['quality_score'].max():.2f}")
    print(f"  平均分数: {top_k_df['quality_score'].mean():.2f}\n")

    # 转换为标准alpaca格式
    result_data = []
    for _, row in top_k_df.iterrows():
        result_data.append({
            "instruction": row['instruction'],
            "input": str(row['input']) if pd.notna(row['input']) else "",
            "output": row['output']
        })

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存 {len(result_data)} 条低质量数据到: {output_file}")


async def main():
    """
    主流程：加载 → 转换 → 评分 → 筛选 → 保存
    """
    print("=" * 60)
    print("Dolly to Alpaca 数据处理 - API 版本")
    print("=" * 60)

    # Step 1: 加载dolly数据并转换为alpaca格式
    alpaca_data = load_dolly_data(INPUT_FILE)

    # Step 1.5: 保存完整的alpaca格式数据（母集）
    print(f"💾 保存完整alpaca格式数据: {ALPACA_FULL_FILE}")
    with open(ALPACA_FULL_FILE, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存完整母集 {len(alpaca_data)} 条数据\n")

    # Step 2: 批量评分
    print("📊 开始质量评分...")
    scores = await score_all_samples(alpaca_data)

    # Step 3: 保存评分结果
    print(f"\n💾 保存评分结果: {SCORED_FILE}")
    df = pd.DataFrame(alpaca_data)
    df['quality_score'] = scores

    # 确保目录存在
    os.makedirs(os.path.dirname(SCORED_FILE), exist_ok=True)
    df.to_csv(SCORED_FILE, index=False, encoding='utf-8')
    print(f"✅ 已保存评分结果\n")

    # Step 4: 筛选低质量数据
    print("🔍 筛选低质量数据...")
    filter_and_save_low_quality(SCORED_FILE, OUTPUT_FILE, TOP_K)

    print("\n" + "=" * 60)
    print("✨ 处理完成！")
    print(f"   - 完整母集(alpaca格式): {ALPACA_FULL_FILE}")
    print(f"   - 低质量子集(alpaca格式): {OUTPUT_FILE}")
    print(f"   - 评分详情: {SCORED_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())