#!/usr/bin/env python3
"""
Dataset Validation Script for Failure-Aware ERNIE

This script validates the dataset structure, checks for required fields,
validates decision labels, and computes label distribution statistics.

Usage:
    python validate_dataset.py --data_dir ../data/failure_aware
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


# Valid decision types
VALID_DECISIONS = {"correct", "uncertain", "refuse"}
VALID_EVIDENCE_QUALITY = {"high", "medium", "low"}


def load_json_dataset(file_path: Path) -> List[Dict]:
    """Load a JSON dataset file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def validate_example(example: Dict, index: int, filename: str) -> Tuple[bool, List[str]]:
    """
    Validate a single training example.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ["instruction", "input", "output"]
    for field in required_fields:
        if field not in example:
            errors.append(f"{filename} example {index}: Missing required field '{field}'")
    
    if errors:
        return False, errors
    
    # Validate output is valid JSON string
    try:
        output_data = json.loads(example["output"])
    except json.JSONDecodeError:
        errors.append(f"{filename} example {index}: 'output' is not valid JSON")
        return False, errors
    
    # Check required output fields
    output_required = ["decision", "answer", "justification", "evidence_quality"]
    for field in output_required:
        if field not in output_data:
            errors.append(f"{filename} example {index}: Output missing '{field}' field")
    
    # Validate decision type
    if "decision" in output_data:
        decision = output_data["decision"]
        if decision not in VALID_DECISIONS:
            errors.append(
                f"{filename} example {index}: Invalid decision '{decision}'. "
                f"Must be one of {VALID_DECISIONS}"
            )
    
    # Validate evidence quality
    if "evidence_quality" in output_data:
        quality = output_data["evidence_quality"]
        if quality not in VALID_EVIDENCE_QUALITY:
            errors.append(
                f"{filename} example {index}: Invalid evidence_quality '{quality}'. "
                f"Must be one of {VALID_EVIDENCE_QUALITY}"
            )
    
    # Check that strings are non-empty
    for field in ["answer", "justification"]:
        if field in output_data:
            if not isinstance(output_data[field], str) or len(output_data[field].strip()) == 0:
                errors.append(f"{filename} example {index}: '{field}' must be a non-empty string")
    
    # Validate instruction and input are non-empty strings
    if not isinstance(example["instruction"], str) or len(example["instruction"].strip()) == 0:
        errors.append(f"{filename} example {index}: 'instruction' must be a non-empty string")
    
    if not isinstance(example["input"], str) or len(example["input"].strip()) == 0:
        errors.append(f"{filename} example {index}: 'input' must be a non-empty string")
    
    return len(errors) == 0, errors


def compute_statistics(data: List[Dict], filename: str) -> Dict:
    """Compute statistics for the dataset."""
    stats = {
        "total_examples": len(data),
        "decision_distribution": Counter(),
        "evidence_quality_distribution": Counter(),
        "decision_by_evidence": {},
    }
    
    for example in data:
        try:
            output = json.loads(example["output"])
            decision = output.get("decision", "unknown")
            evidence = output.get("evidence_quality", "unknown")
            
            stats["decision_distribution"][decision] += 1
            stats["evidence_quality_distribution"][evidence] += 1
            
            # Track decision by evidence quality
            key = f"{evidence}:{decision}"
            if key not in stats["decision_by_evidence"]:
                stats["decision_by_evidence"][key] = 0
            stats["decision_by_evidence"][key] += 1
            
        except (json.JSONDecodeError, KeyError):
            continue
    
    return stats


def validate_dataset(data_dir: Path) -> bool:
    """
    Validate all datasets in the directory.
    
    Returns:
        True if all validations pass, False otherwise.
    """
    all_valid = True
    all_errors = []
    
    # Expected dataset files
    dataset_files = ["train.json", "val.json", "test.json"]
    
    print("=" * 70)
    print("Failure-Aware ERNIE Dataset Validation")
    print("=" * 70)
    print()
    
    for filename in dataset_files:
        file_path = data_dir / filename
        
        print(f"Validating {filename}...")
        print("-" * 70)
        
        if not file_path.exists():
            print(f"[ERROR] {filename} not found at {file_path}")
            all_valid = False
            continue
        
        # Load dataset
        try:
            data = load_json_dataset(file_path)
        except ValueError as e:
            print(f"[ERROR] {e}")
            all_valid = False
            continue
        
        print(f"[OK] Loaded {len(data)} examples")
        
        # Validate each example
        file_errors = []
        for i, example in enumerate(data):
            is_valid, errors = validate_example(example, i, filename)
            if not is_valid:
                file_errors.extend(errors)
                all_valid = False
        
        if file_errors:
            print(f"[ERROR] Found {len(file_errors)} validation errors:")
            for error in file_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(file_errors) > 10:
                print(f"  ... and {len(file_errors) - 10} more errors")
            all_errors.extend(file_errors)
        else:
            print("[OK] All examples have valid structure")
        
        # Compute and display statistics
        stats = compute_statistics(data, filename)
        print(f"\nStatistics for {filename}:")
        print(f"  Total examples: {stats['total_examples']}")
        print(f"\n  Decision distribution:")
        for decision, count in sorted(stats["decision_distribution"].items()):
            percentage = (count / stats['total_examples']) * 100
            print(f"    {decision:12s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n  Evidence quality distribution:")
        for quality, count in sorted(stats["evidence_quality_distribution"].items()):
            percentage = (count / stats['total_examples']) * 100
            print(f"    {quality:12s}: {count:3d} ({percentage:5.1f}%)")
        
        print()
    
    print("=" * 70)
    if all_valid:
        print("[SUCCESS] All validations passed!")
        print("=" * 70)
        return True
    else:
        print(f"[ERROR] Validation failed with {len(all_errors)} total errors")
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate failure-aware dataset for ERNIE fine-tuning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/failure_aware",
        help="Path to the dataset directory"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return 1
    
    success = validate_dataset(data_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
