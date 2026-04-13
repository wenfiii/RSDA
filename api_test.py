import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys


class AlpacaSampler:
    """
    Alpaca Dataset Random Sampling Tool
    Supports multiple sampling strategies and output formats
    """

    def __init__(self, data_path: str):
        """
        Initialize sampler

        Args:
            data_path: Path to Alpaca data file (JSON/JSONL format)
        """
        self.data_path = Path(data_path)
        self.dataset = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {self.data_path}")

        print(f"📂 Loading data: {self.data_path}")

        try:
            # Support JSON format
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # Support JSONL format
            elif self.data_path.suffix == '.jsonl':
                data = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

            print(f"✅ Successfully loaded {len(data)} samples")
            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def sample(
            self,
            n: int,
            strategy: str = "random",
            seed: Optional[int] = None,
            filter_empty_input: bool = False,
            min_output_length: int = 0
    ) -> List[Dict]:
        """
        Sample data

        Args:
            n: Number of samples
            strategy: Sampling strategy
                - "random": Completely random
                - "balanced": Balance samples with/without input
                - "longest": Select longest samples
                - "shortest": Select shortest samples
            seed: Random seed for reproducibility
            filter_empty_input: Whether to filter out samples with empty input
            min_output_length: Minimum output length requirement

        Returns:
            List of sampled data
        """
        # Set random seed
        if seed is not None:
            random.seed(seed)
            print(f"🎲 Random seed: {seed}")

        # Data preprocessing
        filtered_data = self._filter_data(
            self.dataset,
            filter_empty_input,
            min_output_length
        )

        if len(filtered_data) == 0:
            raise ValueError("No data meets the filtering criteria")

        # Check sample count
        if n > len(filtered_data):
            print(f"⚠️  Warning: Requested count ({n}) > Available data ({len(filtered_data)}), returning all data")
            n = len(filtered_data)

        # Execute sampling
        print(f"🎯 Sampling strategy: {strategy}")

        if strategy == "random":
            sampled = random.sample(filtered_data, n)

        elif strategy == "balanced":
            sampled = self._balanced_sample(filtered_data, n)

        elif strategy == "longest":
            sorted_data = sorted(
                filtered_data,
                key=lambda x: len(x.get('output', '')),
                reverse=True
            )
            sampled = sorted_data[:n]

        elif strategy == "shortest":
            sorted_data = sorted(
                filtered_data,
                key=lambda x: len(x.get('output', ''))
            )
            sampled = sorted_data[:n]

        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

        print(f"✅ Sampling completed: {len(sampled)} samples")
        return sampled

    def _filter_data(
            self,
            data: List[Dict],
            filter_empty_input: bool,
            min_output_length: int
    ) -> List[Dict]:
        """Filter data"""
        filtered = data.copy()
        original_count = len(filtered)

        # Filter empty input
        if filter_empty_input:
            filtered = [
                d for d in filtered
                if d.get('input', '').strip()
            ]
            print(f"🔍 Filtered empty input: {original_count} → {len(filtered)}")

        # Filter short output
        if min_output_length > 0:
            before = len(filtered)
            filtered = [
                d for d in filtered
                if len(d.get('output', '')) >= min_output_length
            ]
            print(f"🔍 Filtered short output (<{min_output_length} chars): {before} → {len(filtered)}")

        return filtered

    def _balanced_sample(self, data: List[Dict], n: int) -> List[Dict]:
        """Balanced sampling (with input vs without input)"""
        with_input = [d for d in data if d.get('input', '').strip()]
        without_input = [d for d in data if not d.get('input', '').strip()]

        # Calculate proportions
        ratio = len(with_input) / len(data)
        n_with = int(n * ratio)
        n_without = n - n_with

        # Sample separately
        sampled_with = random.sample(with_input, min(n_with, len(with_input)))
        sampled_without = random.sample(without_input, min(n_without, len(without_input)))

        sampled = sampled_with + sampled_without

        # If not enough, supplement from remaining data
        if len(sampled) < n:
            remaining = [d for d in data if d not in sampled]
            sampled += random.sample(remaining, n - len(sampled))

        print(f"   With input: {len(sampled_with)}, Without input: {len(sampled_without)}")
        return sampled

    def save(
            self,
            data: List[Dict],
            output_path: str,
            format: str = "json",
            indent: int = 2
    ):
        """
        Save sampling results

        Args:
            data: Data to save
            output_path: Output file path
            format: Output format (json/jsonl)
            indent: JSON indentation spaces
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving data: {output_path}")

        try:
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=indent)

            elif format == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

            else:
                raise ValueError(f"Unsupported output format: {format}")

            print(f"✅ Save successful: {len(data)} samples")

        except Exception as e:
            raise RuntimeError(f"Save failed: {e}")

    def print_statistics(self, data: List[Dict]):
        """Print data statistics"""
        print("\n" + "=" * 60)
        print("📊 Data Statistics")
        print("=" * 60)

        total = len(data)
        with_input = sum(1 for d in data if d.get('input', '').strip())
        without_input = total - with_input

        avg_inst_len = sum(len(d.get('instruction', '')) for d in data) / total
        avg_input_len = sum(len(d.get('input', '')) for d in data) / total
        avg_output_len = sum(len(d.get('output', '')) for d in data) / total

        print(f"Total count:    {total}")
        print(f"With input:     {with_input} ({with_input / total * 100:.1f}%)")
        print(f"Without input:  {without_input} ({without_input / total * 100:.1f}%)")
        print(f"\nAverage length:")
        print(f"  instruction:  {avg_inst_len:.0f} chars")
        print(f"  input:        {avg_input_len:.0f} chars")
        print(f"  output:       {avg_output_len:.0f} chars")
        print("=" * 60 + "\n")

    def print_samples(self, data: List[Dict], n: int = 3):
        """Print sample preview"""
        print(f"\n📝 Sample Preview (first {min(n, len(data))} samples):")
        print("-" * 60)

        for i, sample in enumerate(data[:n], 1):
            print(f"\nSample {i}:")
            print(f"  Instruction: {sample.get('instruction', '')[:100]}...")

            input_text = sample.get('input', '')
            if input_text.strip():
                print(f"  Input:       {input_text[:100]}...")
            else:
                print(f"  Input:       (empty)")

            print(f"  Output:      {sample.get('output', '')[:100]}...")

        print("-" * 60 + "\n")


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description="Alpaca Dataset Random Sampling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Randomly sample 100 data points
  python alpaca_sampler.py alpaca_data.json -n 100 -o sampled_100.json

  # Use fixed random seed (reproducible)
  python alpaca_sampler.py alpaca_data.json -n 50 -o sampled_50.json --seed 42

  # Balanced sampling (maintain ratio of with/without input)
  python alpaca_sampler.py alpaca_data.json -n 200 -o balanced_200.json -s balanced

  # Only sample data with input
  python alpaca_sampler.py alpaca_data.json -n 100 -o with_input.json --filter-empty-input

  # Sample longest 50 samples
  python alpaca_sampler.py alpaca_data.json -n 50 -o longest_50.json -s longest

  # Output as JSONL format
  python alpaca_sampler.py alpaca_data.json -n 100 -o sampled.jsonl --format jsonl
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Input Alpaca data file path'
    )

    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        required=True,
        help='Number of samples to extract'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='sampled_alpaca.json',
        help='Output file path (default: sampled_alpaca.json)'
    )

    parser.add_argument(
        '-s', '--strategy',
        type=str,
        choices=['random', 'balanced', 'longest', 'shortest'],
        default='random',
        help='Sampling strategy (default: random)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--filter-empty-input',
        action='store_true',
        help='Filter out samples with empty input'
    )

    parser.add_argument(
        '--min-output-length',
        type=int,
        default=0,
        help='Minimum output length requirement (characters)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl'],
        default='json',
        help='Output format (default: json)'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Print sample preview'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics'
    )

    args = parser.parse_args()

    try:
        # Initialize sampler
        sampler = AlpacaSampler(args.input_file)

        # Execute sampling
        sampled_data = sampler.sample(
            n=args.num_samples,
            strategy=args.strategy,
            seed=args.seed,
            filter_empty_input=args.filter_empty_input,
            min_output_length=args.min_output_length
        )

        # Save results
        sampler.save(
            sampled_data,
            args.output,
            format=args.format
        )

        # Print statistics
        if args.stats:
            sampler.print_statistics(sampled_data)

        # Print sample preview
        if args.preview:
            sampler.print_samples(sampled_data)

        print("🎉 Done!")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()