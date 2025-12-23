"""
Research-Grade Evaluation Plots for Failure-Aware ERNIE Fine-Tuning
Generates 4 publication-quality plots comparing Base vs Fine-tuned models
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("GENERATING EVALUATION PLOTS")
print("=" * 80)

# Load evaluation results
eval_dir = Path("evaluation_results")
with open(eval_dir / "combined_evaluation.json", "r", encoding="utf-8") as f:
    all_results = json.load(f)

# Split by model version
base_results = [r for r in all_results if r['model_version'] == 'base']
finetuned_results = [r for r in all_results if r['model_version'] == 'finetuned']

print(f"Loaded {len(base_results)} base model results")
print(f"Loaded {len(finetuned_results)} fine-tuned model results")

# =============================================================================
# PLOT 1: ACCURACY VS REFUSAL RATE
# =============================================================================
print("\nGenerating Plot 1: Accuracy vs Refusal Rate...")

fig1, ax1 = plt.figure(figsize=(10, 7)), plt.gca()

# Compute metrics for base model
base_refusal_rate = sum(1 for r in base_results if r['decision'] == 'refuse') / len(base_results) * 100
base_answerable = [r for r in base_results if r['expected_decision'] == 'answer']
base_accuracy = sum(1 for r in base_answerable if r['is_correct']) / len(base_answerable) * 100 if base_answerable else 0

# Compute metrics for fine-tuned model
ft_refusal_rate = sum(1 for r in finetuned_results if r['decision'] == 'refuse') / len(finetuned_results) * 100
ft_answerable = [r for r in finetuned_results if r['expected_decision'] == 'answer']
ft_accuracy = sum(1 for r in ft_answerable if r['is_correct']) / len(ft_answerable) * 100 if ft_answerable else 0

# Plot points
ax1.scatter(base_refusal_rate, base_accuracy, s=300, alpha=0.7, 
           color='#ff6b6b', edgecolors='black', linewidths=2, 
           label='Base ERNIE', marker='o', zorder=3)
ax1.scatter(ft_refusal_rate, ft_accuracy, s=300, alpha=0.7,
           color='#51cf66', edgecolors='black', linewidths=2,
           label='Fine-tuned ERNIE', marker='s', zorder=3)

# Add annotations
ax1.annotate(f'Base\n({base_refusal_rate:.1f}%, {base_accuracy:.1f}%)',
            xy=(base_refusal_rate, base_accuracy), xytext=(15, 15),
            textcoords='offset points', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ff6b6b', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax1.annotate(f'Fine-tuned\n({ft_refusal_rate:.1f}%, {ft_accuracy:.1f}%)',
            xy=(ft_refusal_rate, ft_accuracy), xytext=(15, -30),
            textcoords='offset points', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#51cf66', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax1.set_xlabel('Refusal Rate (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy on Answerable Questions (%)', fontsize=14, fontweight='bold')
ax1.set_title('Plot 1: Accuracy vs Refusal Rate\nComparing Base and Fine-tuned ERNIE', 
             fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-1, max(base_refusal_rate, ft_refusal_rate) + 5)
ax1.set_ylim(85, 95)

plt.tight_layout()
plt.savefig('PLOT1_Accuracy_vs_Refusal_Rate.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: PLOT1_Accuracy_vs_Refusal_Rate.png")
plt.close()

# =============================================================================
# PLOT 2: FALSE CONFIDENCE RATE
# =============================================================================
print("Generating Plot 2: False Confidence Rate...")

fig2, ax2 = plt.figure(figsize=(10, 7)), plt.gca()

# Compute false confidence rate
def compute_false_confidence_rate(results):
    answered = [r for r in results if r['decision'] == 'answer']
    if not answered:
        return 0
    false_confident = sum(1 for r in answered if not r['is_correct'])
    return false_confident / len(answered) * 100

base_false_conf = compute_false_confidence_rate(base_results)
ft_false_conf = compute_false_confidence_rate(finetuned_results)

models = ['Base ERNIE', 'Fine-tuned ERNIE']
false_conf_rates = [base_false_conf, ft_false_conf]
colors = ['#ff6b6b', '#51cf66']

bars = ax2.bar(models, false_conf_rates, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=2, width=0.6)

# Add value labels on bars
for bar, val in zip(bars, false_conf_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
improvement = base_false_conf - ft_false_conf
ax2.text(0.5, max(false_conf_rates) * 0.85, 
        f'{improvement:.1f}% Reduction\nin False Confidence',
        ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='gold', alpha=0.7))

ax2.set_ylabel('False Confidence Rate (%)', fontsize=14, fontweight='bold')
ax2.set_title('Plot 2: False Confidence Rate\n(Incorrect answers with "answer" decision)', 
             fontsize=16, fontweight='bold', pad=20)
ax2.set_ylim(0, max(false_conf_rates) * 1.3)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('PLOT2_False_Confidence_Rate.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: PLOT2_False_Confidence_Rate.png")
plt.close()

# =============================================================================
# PLOT 3: DECISION DISTRIBUTION COMPARISON
# =============================================================================
print("Generating Plot 3: Decision Distribution...")

fig3, ax3 = plt.figure(figsize=(10, 7)), plt.gca()

# Compute decision distributions
def get_decision_distribution(results):
    total = len(results)
    answer_pct = sum(1 for r in results if r['decision'] == 'answer') / total * 100
    uncertain_pct = sum(1 for r in results if r['decision'] == 'uncertain') / total * 100
    refuse_pct = sum(1 for r in results if r['decision'] == 'refuse') / total * 100
    return answer_pct, uncertain_pct, refuse_pct

base_dist = get_decision_distribution(base_results)
ft_dist = get_decision_distribution(finetuned_results)

models = ['Base ERNIE', 'Fine-tuned ERNIE']
answer_pcts = [base_dist[0], ft_dist[0]]
uncertain_pcts = [base_dist[1], ft_dist[1]]
refuse_pcts = [base_dist[2], ft_dist[2]]

width = 0.5
x = np.arange(len(models))

# Create stacked bars
p1 = ax3.bar(x, answer_pcts, width, label='Answer (Correct decision)', 
            color='#51cf66', alpha=0.8, edgecolor='black', linewidth=1.5)
p2 = ax3.bar(x, uncertain_pcts, width, bottom=answer_pcts,
            label='Uncertain', color='#ffd43b', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
p3 = ax3.bar(x, refuse_pcts, width, 
            bottom=np.array(answer_pcts) + np.array(uncertain_pcts),
            label='Refuse', color='#ff6b6b', alpha=0.8,
            edgecolor='black', linewidth=1.5)

# Add percentage labels
for i in range(len(models)):
    # Answer label
    if answer_pcts[i] > 5:
        ax3.text(x[i], answer_pcts[i]/2, f'{answer_pcts[i]:.1f}%',
                ha='center', va='center', fontsize=11, fontweight='bold')
    # Uncertain label
    if uncertain_pcts[i] > 5:
        ax3.text(x[i], answer_pcts[i] + uncertain_pcts[i]/2, f'{uncertain_pcts[i]:.1f}%',
                ha='center', va='center', fontsize=11, fontweight='bold')
    # Refuse label
    if refuse_pcts[i] > 5:
        ax3.text(x[i], answer_pcts[i] + uncertain_pcts[i] + refuse_pcts[i]/2, 
                f'{refuse_pcts[i]:.1f}%',
                ha='center', va='center', fontsize=11, fontweight='bold')

ax3.set_ylabel('Distribution (%)', fontsize=14, fontweight='bold')
ax3.set_title('Plot 3: Decision Distribution Comparison\nNormalized to Percentages', 
             fontsize=16, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=13)
ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add insight box
insight_text = f'Fine-tuned model:\n↑ {ft_dist[1]-base_dist[1]:.1f}% more uncertain\n↑ {ft_dist[2]-base_dist[2]:.1f}% more refuse'
ax3.text(0.02, 0.98, insight_text, transform=ax3.transAxes,
        fontsize=10, fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('PLOT3_Decision_Distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: PLOT3_Decision_Distribution.png")
plt.close()

# =============================================================================
# PLOT 4: CALIBRATION CURVE
# =============================================================================
print("Generating Plot 4: Calibration Curve...")

fig4, ax4 = plt.figure(figsize=(10, 8)), plt.gca()

def compute_calibration(results, n_bins=10):
    """Compute calibration curve with confidence binning"""
    # Only use samples where model made a decision (not refuse)
    decided = [r for r in results if r['decision'] in ['answer', 'uncertain']]
    
    if not decided:
        return [], []
    
    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    
    for i in range(n_bins):
        bin_low, bin_high = bins[i], bins[i+1]
        bin_samples = [r for r in decided if bin_low <= r['confidence'] < bin_high]
        
        if bin_samples:
            avg_confidence = np.mean([r['confidence'] for r in bin_samples])
            accuracy = sum(1 for r in bin_samples if r['is_correct']) / len(bin_samples)
            bin_confidences.append(avg_confidence)
            bin_accuracies.append(accuracy)
    
    return bin_confidences, bin_accuracies

# Compute calibration for both models
base_conf, base_acc = compute_calibration(base_results)
ft_conf, ft_acc = compute_calibration(finetuned_results)

# Plot diagonal reference line (perfect calibration)
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

# Plot calibration curves
if base_conf:
    ax4.plot(base_conf, base_acc, 'o-', linewidth=3, markersize=10,
            color='#ff6b6b', label='Base ERNIE', alpha=0.7)
if ft_conf:
    ax4.plot(ft_conf, ft_acc, 's-', linewidth=3, markersize=10,
            color='#51cf66', label='Fine-tuned ERNIE', alpha=0.7)

ax4.set_xlabel('Confidence Score', fontsize=14, fontweight='bold')
ax4.set_ylabel('Actual Accuracy', fontsize=14, fontweight='bold')
ax4.set_title('Plot 4: Model Calibration Curve\nConfidence vs Actual Correctness', 
             fontsize=16, fontweight='bold', pad=20)
ax4.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_aspect('equal')

# Add calibration quality annotation
# Closer to diagonal = better calibrated
base_ece = np.mean(np.abs(np.array(base_conf) - np.array(base_acc))) if base_conf else 0
ft_ece = np.mean(np.abs(np.array(ft_conf) - np.array(ft_acc))) if ft_conf else 0

calib_text = f'Expected Calibration Error (ECE):\nBase: {base_ece:.3f}\nFine-tuned: {ft_ece:.3f}'
ax4.text(0.02, 0.98, calib_text, transform=ax4.transAxes,
        fontsize=10, fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('PLOT4_Calibration_Curve.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: PLOT4_Calibration_Curve.png")
plt.close()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("EVALUATION METRICS SUMMARY")
print("=" * 80)

print("\nBase Model:")
print(f"  Overall Accuracy: {sum(1 for r in base_results if r['is_correct'])/len(base_results)*100:.1f}%")
print(f"  Refusal Rate: {base_refusal_rate:.1f}%")
print(f"  Accuracy on Answerable: {base_accuracy:.1f}%")
print(f"  False Confidence Rate: {base_false_conf:.1f}%")

print("\nFine-tuned Model:")
print(f"  Overall Accuracy: {sum(1 for r in finetuned_results if r['is_correct'])/len(finetuned_results)*100:.1f}%")
print(f"  Refusal Rate: {ft_refusal_rate:.1f}%")
print(f"  Accuracy on Answerable: {ft_accuracy:.1f}%")
print(f"  False Confidence Rate: {ft_false_conf:.1f}%")

print("\nImprovements:")
print(f"  Overall Accuracy: +{sum(1 for r in finetuned_results if r['is_correct'])/len(finetuned_results)*100 - sum(1 for r in base_results if r['is_correct'])/len(base_results)*100:.1f}%")
print(f"  False Confidence: -{base_false_conf - ft_false_conf:.1f}%")
print(f"  Better Calibration: ECE {base_ece:.3f} → {ft_ece:.3f}")

print("\n" + "=" * 80)
print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated files:")
print("  • PLOT1_Accuracy_vs_Refusal_Rate.png")
print("  • PLOT2_False_Confidence_Rate.png")
print("  • PLOT3_Decision_Distribution.png")
print("  • PLOT4_Calibration_Curve.png")
print("\nReady for hackathon presentation!")
