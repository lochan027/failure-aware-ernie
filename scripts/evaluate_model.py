#!/usr/bin/env python3
"""
Model Evaluation Script for Failure-Aware ERNIE

This script evaluates the fine-tuned model against the base model on the test set,
computing key metrics:
- Answer Accuracy (for 'correct' decisions)
- Refusal Rate
- False Confidence Rate (wrong answers labeled as correct)
- Hallucination Rate
- Calibration metrics

Usage:
    python evaluate_model.py --test_data ../data/failure_aware/test.json \
                              --base_model nghuyong/ernie-3.0-base-zh \
                              --finetuned_model ../output/ernie_failure_aware \
                              --output results.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


def load_test_data(file_path: Path) -> List[Dict]:
    """Load test dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_model_output(output_text: str) -> Dict:
    """
    Parse the model's output to extract decision, answer, justification, and evidence quality.
    Handles both JSON format and unstructured text.
    """
    # Try to extract JSON from the output
    try:
        # Look for JSON-like structure
        json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', output_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "decision": parsed.get("decision", "unknown"),
                "answer": parsed.get("answer", ""),
                "justification": parsed.get("justification", ""),
                "evidence_quality": parsed.get("evidence_quality", "unknown")
            }
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract decision from text
    decision = "unknown"
    if "refuse" in output_text.lower() or "cannot" in output_text.lower():
        decision = "refuse"
    elif "uncertain" in output_text.lower() or "unclear" in output_text.lower():
        decision = "uncertain"
    elif any(word in output_text.lower() for word in ["answer", "is", "equals", "are"]):
        decision = "correct"
    
    return {
        "decision": decision,
        "answer": output_text,
        "justification": "",
        "evidence_quality": "unknown"
    }


def generate_response(model, tokenizer, instruction: str, input_text: str, max_length: int = 512) -> str:
    """Generate a response from the model."""
    prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part (after the prompt)
    if "Output:" in generated_text:
        response = generated_text.split("Output:")[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response


def evaluate_model(model, tokenizer, test_data: List[Dict], model_name: str) -> Dict:
    """
    Evaluate a model on the test set.
    
    Returns a dictionary with evaluation metrics and predictions.
    """
    predictions = []
    
    print(f"\nEvaluating {model_name}...")
    
    for example in tqdm(test_data, desc=f"Evaluating {model_name}"):
        instruction = example["instruction"]
        input_text = example["input"]
        
        # Get ground truth
        ground_truth = json.loads(example["output"])
        
        # Generate prediction
        response = generate_response(model, tokenizer, instruction, input_text)
        prediction = parse_model_output(response)
        
        predictions.append({
            "input": input_text,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "raw_output": response
        })
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "predictions": predictions
    }


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute evaluation metrics from predictions.
    """
    total = len(predictions)
    
    # Decision accuracy
    correct_decisions = sum(
        1 for p in predictions 
        if p["prediction"]["decision"] == p["ground_truth"]["decision"]
    )
    decision_accuracy = correct_decisions / total if total > 0 else 0.0
    
    # Decision distribution
    predicted_decisions = [p["prediction"]["decision"] for p in predictions]
    ground_truth_decisions = [p["ground_truth"]["decision"] for p in predictions]
    
    decision_distribution = {
        "predicted": {
            "correct": predicted_decisions.count("correct"),
            "uncertain": predicted_decisions.count("uncertain"),
            "refuse": predicted_decisions.count("refuse"),
            "unknown": predicted_decisions.count("unknown")
        },
        "ground_truth": {
            "correct": ground_truth_decisions.count("correct"),
            "uncertain": ground_truth_decisions.count("uncertain"),
            "refuse": ground_truth_decisions.count("refuse")
        }
    }
    
    # Refusal rate
    refusal_rate = predicted_decisions.count("refuse") / total if total > 0 else 0.0
    
    # False confidence rate: predicted "correct" but was wrong
    false_confidence_count = sum(
        1 for p in predictions
        if p["prediction"]["decision"] == "correct" 
        and p["ground_truth"]["decision"] != "correct"
    )
    
    predicted_correct_count = predicted_decisions.count("correct")
    false_confidence_rate = (
        false_confidence_count / predicted_correct_count 
        if predicted_correct_count > 0 else 0.0
    )
    
    # Hallucination rate: should refuse but answered confidently
    hallucination_count = sum(
        1 for p in predictions
        if p["ground_truth"]["decision"] == "refuse"
        and p["prediction"]["decision"] == "correct"
    )
    
    should_refuse_count = ground_truth_decisions.count("refuse")
    hallucination_rate = (
        hallucination_count / should_refuse_count
        if should_refuse_count > 0 else 0.0
    )
    
    # Precision and recall for each decision type
    per_class_metrics = {}
    for decision_type in ["correct", "uncertain", "refuse"]:
        tp = sum(
            1 for p in predictions
            if p["prediction"]["decision"] == decision_type
            and p["ground_truth"]["decision"] == decision_type
        )
        fp = sum(
            1 for p in predictions
            if p["prediction"]["decision"] == decision_type
            and p["ground_truth"]["decision"] != decision_type
        )
        fn = sum(
            1 for p in predictions
            if p["prediction"]["decision"] != decision_type
            and p["ground_truth"]["decision"] == decision_type
        )
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[decision_type] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": ground_truth_decisions.count(decision_type)
        }
    
    # Calibration: confidence vs correctness
    # For simplicity, map decisions to confidence levels
    confidence_correctness = []
    for p in predictions:
        predicted_decision = p["prediction"]["decision"]
        is_correct = predicted_decision == p["ground_truth"]["decision"]
        
        # Map decision to confidence score
        confidence_map = {
            "correct": 1.0,
            "uncertain": 0.5,
            "refuse": 0.0,
            "unknown": 0.5
        }
        confidence = confidence_map.get(predicted_decision, 0.5)
        confidence_correctness.append((confidence, 1.0 if is_correct else 0.0))
    
    # Compute calibration error (Expected Calibration Error - ECE)
    n_bins = 10
    calibration_bins = defaultdict(list)
    for conf, correct in confidence_correctness:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        calibration_bins[bin_idx].append(correct)
    
    ece = 0.0
    for bin_idx in range(n_bins):
        if bin_idx in calibration_bins:
            bin_correct = calibration_bins[bin_idx]
            bin_confidence = (bin_idx + 0.5) / n_bins
            bin_accuracy = np.mean(bin_correct)
            ece += len(bin_correct) / len(confidence_correctness) * abs(bin_accuracy - bin_confidence)
    
    return {
        "decision_accuracy": decision_accuracy,
        "refusal_rate": refusal_rate,
        "false_confidence_rate": false_confidence_rate,
        "hallucination_rate": hallucination_rate,
        "expected_calibration_error": ece,
        "decision_distribution": decision_distribution,
        "per_class_metrics": per_class_metrics,
        "total_examples": total,
        "calibration_data": confidence_correctness
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate failure-aware ERNIE model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="nghuyong/ernie-3.0-base-zh",
        help="Base model name or path"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default=None,
        help="Fine-tuned model checkpoint path (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(Path(args.test_data))
    print(f"Loaded {len(test_data)} test examples")
    
    results = {}
    
    # Evaluate base model
    print(f"\nLoading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.to(args.device)
    base_model.eval()
    
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    results["base_model"] = evaluate_model(base_model, base_tokenizer, test_data, "Base ERNIE")
    
    # Free memory
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate fine-tuned model if provided
    if args.finetuned_model:
        print(f"\nLoading fine-tuned model: {args.finetuned_model}")
        ft_tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model)
        ft_model = AutoModelForCausalLM.from_pretrained(args.finetuned_model)
        ft_model.to(args.device)
        ft_model.eval()
        
        if ft_tokenizer.pad_token is None:
            ft_tokenizer.pad_token = ft_tokenizer.eos_token
        
        results["finetuned_model"] = evaluate_model(
            ft_model, ft_tokenizer, test_data, "Fine-tuned ERNIE"
        )
        
        del ft_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    for model_key in results:
        model_results = results[model_key]
        metrics = model_results["metrics"]
        
        print(f"\n{model_results['model_name']}:")
        print(f"  Decision Accuracy:        {metrics['decision_accuracy']:.3f}")
        print(f"  Refusal Rate:             {metrics['refusal_rate']:.3f}")
        print(f"  False Confidence Rate:    {metrics['false_confidence_rate']:.3f}")
        print(f"  Hallucination Rate:       {metrics['hallucination_rate']:.3f}")
        print(f"  Calibration Error (ECE):  {metrics['expected_calibration_error']:.3f}")
        
        print(f"\n  Per-class metrics:")
        for decision_type, class_metrics in metrics['per_class_metrics'].items():
            print(f"    {decision_type}:")
            print(f"      Precision: {class_metrics['precision']:.3f}")
            print(f"      Recall:    {class_metrics['recall']:.3f}")
            print(f"      F1 Score:  {class_metrics['f1_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
