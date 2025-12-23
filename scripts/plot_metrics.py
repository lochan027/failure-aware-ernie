#!/usr/bin/env python3
"""
Plotting Script for Failure-Aware ERNIE Evaluation

This script generates visualizations from evaluation results:
- Accuracy vs Refusal Rate comparison
- Hallucination reduction
- Calibration curves
- Per-class performance comparison
- Metric comparison table

Usage:
    python plot_metrics.py --results evaluation_results.json --output_dir plots/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def load_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_metric_comparison(results: Dict, output_dir: Path):
    """
    Create a bar chart comparing key metrics between base and fine-tuned models.
    """
    models = []
    metrics_data = {
        'Decision Accuracy': [],
        'Refusal Rate': [],
        'False Confidence': [],
        'Hallucination Rate': [],
        'Calibration Error': []
    }
    
    for model_key in ['base_model', 'finetuned_model']:
        if model_key in results:
            model_name = results[model_key]['model_name']
            metrics = results[model_key]['metrics']
            
            models.append(model_name)
            metrics_data['Decision Accuracy'].append(metrics['decision_accuracy'])
            metrics_data['Refusal Rate'].append(metrics['refusal_rate'])
            metrics_data['False Confidence'].append(metrics['false_confidence_rate'])
            metrics_data['Hallucination Rate'].append(metrics['hallucination_rate'])
            metrics_data['Calibration Error'].append(metrics['expected_calibration_error'])
    
    if len(models) == 0:
        print("No models to compare")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_data))
    width = 0.35 if len(models) == 2 else 0.5
    
    colors = ['#3498db', '#e74c3c']
    
    for i, model in enumerate(models):
        values = [metrics_data[metric][i] for metric in metrics_data]
        offset = (i - 0.5) * width if len(models) == 2 else 0
        ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison: Key Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data.keys(), rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metric comparison to {output_dir / 'metric_comparison.png'}")


def plot_calibration_curve(results: Dict, output_dir: Path):
    """
    Plot calibration curves showing the relationship between confidence and accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, model_key in enumerate(['base_model', 'finetuned_model']):
        if model_key not in results:
            continue
        
        model_name = results[model_key]['model_name']
        calibration_data = results[model_key]['metrics']['calibration_data']
        
        # Bin the data
        n_bins = 10
        bins = defaultdict(list)
        for conf, correct in calibration_data:
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append(correct)
        
        # Compute average confidence and accuracy per bin
        bin_confidences = []
        bin_accuracies = []
        bin_sizes = []
        
        for bin_idx in range(n_bins):
            if bin_idx in bins:
                bin_confidence = (bin_idx + 0.5) / n_bins
                bin_accuracy = np.mean(bins[bin_idx])
                bin_size = len(bins[bin_idx])
                
                bin_confidences.append(bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_sizes.append(bin_size)
        
        # Plot calibration curve
        ax.plot(bin_confidences, bin_accuracies, 'o-', 
                label=model_name, color=colors[idx], 
                linewidth=2, markersize=8, alpha=0.8)
        
        # Plot confidence bars
        # for conf, acc, size in zip(bin_confidences, bin_accuracies, bin_sizes):
        #     ax.bar(conf, acc, width=0.08, alpha=0.2, color=colors[idx])
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Calibration Curve: Confidence vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved calibration curve to {output_dir / 'calibration_curve.png'}")


def plot_per_class_performance(results: Dict, output_dir: Path):
    """
    Plot per-class precision, recall, and F1 scores for each model.
    """
    decision_types = ['correct', 'uncertain', 'refuse']
    
    for model_key in ['base_model', 'finetuned_model']:
        if model_key not in results:
            continue
        
        model_name = results[model_key]['model_name']
        per_class = results[model_key]['metrics']['per_class_metrics']
        
        # Extract metrics
        precision = [per_class[dt]['precision'] for dt in decision_types]
        recall = [per_class[dt]['recall'] for dt in decision_types]
        f1 = [per_class[dt]['f1_score'] for dt in decision_types]
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(decision_types))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance: {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([dt.capitalize() for dt in decision_types])
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        
        plt.tight_layout()
        
        filename = f"per_class_{model_key}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-class performance to {output_dir / filename}")


def plot_decision_distribution(results: Dict, output_dir: Path):
    """
    Plot the distribution of predicted decisions vs ground truth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    decision_types = ['correct', 'uncertain', 'refuse']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for idx, model_key in enumerate(['base_model', 'finetuned_model']):
        if model_key not in results:
            continue
        
        model_name = results[model_key]['model_name']
        dist = results[model_key]['metrics']['decision_distribution']
        
        # Ground truth is same for both models
        if idx == 0:
            gt_counts = [dist['ground_truth'][dt] for dt in decision_types]
            axes[0].bar(decision_types, gt_counts, color=colors, alpha=0.7)
            axes[0].set_title('Ground Truth Distribution', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=11)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (dt, count) in enumerate(zip(decision_types, gt_counts)):
                axes[0].text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)
    
    # Create comparison chart
    if len(results) == 2:
        models = []
        pred_data = {dt: [] for dt in decision_types}
        
        for model_key in ['base_model', 'finetuned_model']:
            models.append(results[model_key]['model_name'])
            dist = results[model_key]['metrics']['decision_distribution']
            for dt in decision_types:
                pred_data[dt].append(dist['predicted'][dt])
        
        x = np.arange(len(decision_types))
        width = 0.35
        
        for i, model in enumerate(models):
            counts = [pred_data[dt][i] for dt in decision_types]
            offset = (i - 0.5) * width
            axes[1].bar(x + offset, counts, width, label=model, alpha=0.7)
        
        axes[1].set_title('Predicted Distribution Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([dt.capitalize() for dt in decision_types])
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved decision distribution to {output_dir / 'decision_distribution.png'}")


def plot_hallucination_reduction(results: Dict, output_dir: Path):
    """
    Visualize the reduction in hallucinations and false confidence.
    """
    if len(results) < 2:
        print("⚠ Need both base and fine-tuned models for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_compare = ['Hallucination Rate', 'False Confidence Rate']
    base_values = [
        results['base_model']['metrics']['hallucination_rate'],
        results['base_model']['metrics']['false_confidence_rate']
    ]
    ft_values = [
        results['finetuned_model']['metrics']['hallucination_rate'],
        results['finetuned_model']['metrics']['false_confidence_rate']
    ]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model', 
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, ft_values, width, label='Fine-tuned Model', 
                   color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Hallucination & False Confidence Reduction', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_compare)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(base_values + ft_values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and improvement indicators
    for i, (b1, b2) in enumerate(zip(base_values, ft_values)):
        ax.text(i - width/2, b1 + 0.02, f'{b1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, b2 + 0.02, f'{b2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Calculate and show improvement
        if b1 > 0:
            improvement = ((b1 - b2) / b1) * 100
            ax.text(i, max(b1, b2) + 0.05, f'↓{improvement:.1f}%', 
                   ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hallucination_reduction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved hallucination reduction to {output_dir / 'hallucination_reduction.png'}")


def create_summary_table(results: Dict, output_dir: Path):
    """
    Create a text summary table of all metrics.
    """
    summary = []
    summary.append("=" * 80)
    summary.append("EVALUATION RESULTS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    for model_key in ['base_model', 'finetuned_model']:
        if model_key not in results:
            continue
        
        model_name = results[model_key]['model_name']
        metrics = results[model_key]['metrics']
        
        summary.append(f"{model_name}")
        summary.append("-" * 80)
        summary.append(f"  Decision Accuracy:        {metrics['decision_accuracy']:.4f}")
        summary.append(f"  Refusal Rate:             {metrics['refusal_rate']:.4f}")
        summary.append(f"  False Confidence Rate:    {metrics['false_confidence_rate']:.4f}")
        summary.append(f"  Hallucination Rate:       {metrics['hallucination_rate']:.4f}")
        summary.append(f"  Calibration Error (ECE):  {metrics['expected_calibration_error']:.4f}")
        summary.append("")
        summary.append("  Per-Class Metrics:")
        
        for decision_type in ['correct', 'uncertain', 'refuse']:
            class_metrics = metrics['per_class_metrics'][decision_type]
            summary.append(f"    {decision_type.capitalize()}:")
            summary.append(f"      Precision: {class_metrics['precision']:.4f}")
            summary.append(f"      Recall:    {class_metrics['recall']:.4f}")
            summary.append(f"      F1 Score:  {class_metrics['f1_score']:.4f}")
            summary.append(f"      Support:   {class_metrics['support']}")
        
        summary.append("")
        summary.append("  Decision Distribution:")
        dist = metrics['decision_distribution']
        summary.append(f"    Predicted correct:   {dist['predicted']['correct']}")
        summary.append(f"    Predicted uncertain: {dist['predicted']['uncertain']}")
        summary.append(f"    Predicted refuse:    {dist['predicted']['refuse']}")
        summary.append("")
    
    # Add comparison if both models present
    if 'base_model' in results and 'finetuned_model' in results:
        summary.append("IMPROVEMENT SUMMARY")
        summary.append("-" * 80)
        
        base = results['base_model']['metrics']
        ft = results['finetuned_model']['metrics']
        
        improvements = {
            'Decision Accuracy': ft['decision_accuracy'] - base['decision_accuracy'],
            'Hallucination Reduction': base['hallucination_rate'] - ft['hallucination_rate'],
            'False Confidence Reduction': base['false_confidence_rate'] - ft['false_confidence_rate'],
            'Calibration Improvement': base['expected_calibration_error'] - ft['expected_calibration_error']
        }
        
        for metric, value in improvements.items():
            sign = "+" if value > 0 else ""
            summary.append(f"  {metric:30s}: {sign}{value:.4f}")
        
        summary.append("")
    
    summary.append("=" * 80)
    
    # Write to file
    summary_text = "\n".join(summary)
    with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✓ Saved summary table to {output_dir / 'summary.txt'}")
    print("\n" + summary_text)


# Import defaultdict at the top
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from evaluation results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    
    print(f"\nGenerating plots in {output_dir}...")
    print("-" * 70)
    
    # Generate all plots
    plot_metric_comparison(results, output_dir)
    plot_calibration_curve(results, output_dir)
    plot_per_class_performance(results, output_dir)
    plot_decision_distribution(results, output_dir)
    plot_hallucination_reduction(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("-" * 70)
    print(f"✅ All plots generated successfully in {output_dir}")


if __name__ == "__main__":
    main()
