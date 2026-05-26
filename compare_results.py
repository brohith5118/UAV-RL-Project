import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import main

def run_comparison():
    art_dir = r"C:\Users\emand\.gemini\antigravity-ide\brain\dab6e8c9-bc2b-4e5d-afc9-6fee29422b1b"
    os.makedirs(art_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING BASELINE SIMULATION (optimize=False)")
    print("=" * 60)
    random.seed(42)
    np.random.seed(42)
    baseline_metrics = main.main(optimize=False, save_dir=art_dir, prefix="baseline_")

    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZED SIMULATION (optimize=True)")
    print("=" * 60)
    random.seed(42)
    np.random.seed(42)
    optimized_metrics = main.main(optimize=True, save_dir=art_dir, prefix="optimized_")

    # Generate the Markdown table
    report = f"""# Performance Comparison Report: DMMP-PR-TSA Optimization

| Performance Metric | Baseline (Memoryless RL & Static Constraints) | Optimized (State Bitmask, EDF Bootstrapping, Dynamic Resource Tracking & Load Balancing) |
| :--- | :---: | :---: |
| **Overall Completion Rate** | {baseline_metrics['completion_rate']*100:.1f}% | {optimized_metrics['completion_rate']*100:.1f}% |
| **High-Priority Completion Rate** | {baseline_metrics['high_priority_completion_rate']*100:.1f}% | {optimized_metrics['high_priority_completion_rate']*100:.1f}% |
| **Overloaded UAV Count** | {baseline_metrics['overloaded_uav_count']} | {optimized_metrics['overloaded_uav_count']} |
| **Total Travel Distance** | {baseline_metrics['total_travel_distance']:.1f} m | {optimized_metrics['total_travel_distance']:.1f} m |
| **Mean Energy Utilisation** | {baseline_metrics['energy_utilisation']*100:.1f}% | {optimized_metrics['energy_utilisation']*100:.1f}% |
| **Mean Compute Utilisation** | {baseline_metrics['compute_utilisation']*100:.1f}% | {optimized_metrics['compute_utilisation']*100:.1f}% |
| **Jain's Fairness Index** | {baseline_metrics['jains_fairness_index']:.3f} | {optimized_metrics['jains_fairness_index']:.3f} |
"""

    print("\n" + "=" * 60)
    print("             PERFORMANCE COMPARISON REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Save report to artifacts directory
    report_path = os.path.join(art_dir, "performance_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved performance report to {report_path}")

if __name__ == '__main__':
    run_comparison()
