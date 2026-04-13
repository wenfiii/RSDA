import json


def clean_to_alpaca_format(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    skipped = 0

    for idx, item in enumerate(data):

        if "instruction" not in item:
            print(f"ERROR: Item {idx} missing 'instruction' field, skipped")
            skipped += 1
            continue

        if "output" not in item:
            print(f"WARNING: Item {idx} missing 'output' field, skipped")
            print(f"  Preview: {str(item)[:100]}...")
            skipped += 1
            continue

        cleaned_item = {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }
        cleaned_data.append(cleaned_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully cleaned {len(cleaned_data)} items")
    if skipped > 0:
        print(f"Skipped {skipped} invalid items")
    print(f"Saved to: {output_file}")

    if cleaned_data:
        print("\nSample data:")
        print(json.dumps(cleaned_data[0], ensure_ascii=False, indent=2))


def clean_with_optional_input(input_file, output_file):
    """
    Clean data, omitting input field when empty (more standard Alpaca format)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    skipped = 0

    for idx, item in enumerate(data):
        if "instruction" not in item or "output" not in item:
            print(f"WARNING: Item {idx} missing required fields, skipped")
            skipped += 1
            continue

        cleaned_item = {"instruction": item["instruction"]}

        if item.get("input", "").strip():
            cleaned_item["input"] = item["input"]

        cleaned_item["output"] = item["output"]
        cleaned_data.append(cleaned_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully cleaned {len(cleaned_data)} items")
    if skipped > 0:
        print(f"Skipped {skipped} invalid items")
    print(f"Saved to: {output_file}")


def check_data_structure(input_file):
    """
    Check data structure and identify issues
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nData structure check:")
    print(f"  Total items: {len(data)}")

    has_instruction = sum(1 for item in data if "instruction" in item)
    has_input = sum(1 for item in data if "input" in item)
    has_output = sum(1 for item in data if "output" in item)
    has_meta = sum(1 for item in data if "meta" in item)

    print(f"  Contains 'instruction': {has_instruction}/{len(data)}")
    print(f"  Contains 'input': {has_input}/{len(data)}")
    print(f"  Contains 'output': {has_output}/{len(data)}")
    print(f"  Contains 'meta': {has_meta}/{len(data)}")

    for idx, item in enumerate(data):
        if "output" not in item:
            print(f"\nFirst item missing 'output' (index {idx}):")
            print(f"  Available fields: {list(item.keys())}")
            print(f"  Content preview: {json.dumps(item, ensure_ascii=False, indent=2)[:300]}...")
            break

    if data:
        print(f"\nFirst item fields: {list(data[0].keys())}")


def add_statistics(input_file):
    """
    Generate dataset statistics
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    with_input = sum(1 for item in data if item.get("input", "").strip())
    without_input = total - with_input

    print(f"\nDataset statistics:")
    print(f"  Total items: {total}")
    print(f"  With input: {with_input} ({with_input / total * 100:.1f}%)")
    print(f"  Without input: {without_input} ({without_input / total * 100:.1f}%)")


if __name__ == "__main__":
    input_file = "/home/disk2/dolly_u.json"
    output_file = "dolly_u.json"

    print("=" * 50)
    check_data_structure(input_file)
    print("=" * 50)

    clean_to_alpaca_format(input_file, output_file)

    # add_statistics(output_file)