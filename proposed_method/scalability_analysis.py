# =========================================================
# SCALABILITY ANALYSIS
# =========================================================

import os
import sys
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt

# Add project root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from proposed_method import main as proposed_main


# =========================================================
# SETTINGS
# =========================================================

TASK_COUNTS = [10, 20, 30, 40, 50, 60, 80, 100]
RUNS_PER_SETTING = 3

SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated_graphs",
    "scalability"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# RUN EXPERIMENTS
# =========================================================

results = {
    "tasks": [],
    "completion_rate": [],
    "high_priority_completion_rate": [],
    "travel_distance": [],
    "energy_utilisation": [],
    "compute_utilisation": [],
    "fairness": [],
    "overloaded": [],
    "runtime": []
}

for task_count in TASK_COUNTS:

    print("\n" + "=" * 60)
    print(f"Testing {task_count} Tasks")
    print("=" * 60)

    completion_rates = []
    hp_rates = []
    travel_distances = []
    energy_utils = []
    compute_utils = []
    fairness_scores = []
    overloaded_counts = []
    runtimes = []

    for run in range(RUNS_PER_SETTING):

        print(f"Run {run + 1}/{RUNS_PER_SETTING}")

        config.NUM_TASKS = task_count

        importlib.reload(proposed_main)

        start_time = time.perf_counter()

        metrics = proposed_main.main(
            optimize=True,
            prefix=f"scale_{task_count}_run_{run}"
        )

        elapsed = time.perf_counter() - start_time

        completion_rates.append(
            metrics["completion_rate"]
        )

        hp_rates.append(
            metrics["high_priority_completion_rate"]
        )

        travel_distances.append(
            metrics["total_travel_distance"]
        )

        energy_utils.append(
            metrics["energy_utilisation"]
        )

        compute_utils.append(
            metrics["compute_utilisation"]
        )

        fairness_scores.append(
            metrics["jains_fairness_index"]
        )

        overloaded_counts.append(
            metrics["overloaded_uav_count"]
        )

        runtimes.append(elapsed)

    results["tasks"].append(task_count)
    results["completion_rate"].append(np.mean(completion_rates))
    results["high_priority_completion_rate"].append(np.mean(hp_rates))
    results["travel_distance"].append(np.mean(travel_distances))
    results["energy_utilisation"].append(np.mean(energy_utils))
    results["compute_utilisation"].append(np.mean(compute_utils))
    results["fairness"].append(np.mean(fairness_scores))
    results["overloaded"].append(np.mean(overloaded_counts))
    results["runtime"].append(np.mean(runtimes))


# =========================================================
# PLOTTING
# =========================================================

def save_plot(y, ylabel, filename):

    plt.figure(figsize=(8, 5))

    plt.plot(
        results["tasks"],
        y,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of Tasks")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Task Count")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, filename),
        dpi=300
    )

    plt.close()


save_plot(
    results["completion_rate"],
    "Completion Rate (%)",
    "scalability_completion_rate.png"
)

save_plot(
    results["high_priority_completion_rate"],
    "High Priority Completion Rate (%)",
    "scalability_high_priority_completion.png"
)

save_plot(
    results["travel_distance"],
    "Travel Distance",
    "scalability_travel_distance.png"
)

save_plot(
    results["energy_utilisation"],
    "Energy Utilisation (%)",
    "scalability_energy_utilisation.png"
)

save_plot(
    results["compute_utilisation"],
    "Compute Utilisation (%)",
    "scalability_compute_utilisation.png"
)

save_plot(
    results["fairness"],
    "Jain Fairness Index",
    "scalability_fairness.png"
)

save_plot(
    results["overloaded"],
    "Overloaded UAV Count",
    "scalability_overloaded_uavs.png"
)

save_plot(
    results["runtime"],
    "Runtime (seconds)",
    "scalability_runtime.png"
)


# =========================================================
# SAVE CSV
# =========================================================

csv_path = os.path.join(
    SAVE_DIR,
    "scalability_results.csv"
)

with open(csv_path, "w") as f:

    headers = [
        "tasks",
        "completion_rate",
        "high_priority_completion_rate",
        "travel_distance",
        "energy_utilisation",
        "compute_utilisation",
        "fairness",
        "overloaded",
        "runtime"
    ]

    f.write(",".join(headers) + "\n")

    for i in range(len(results["tasks"])):

        row = [
            results["tasks"][i],
            results["completion_rate"][i],
            results["high_priority_completion_rate"][i],
            results["travel_distance"][i],
            results["energy_utilisation"][i],
            results["compute_utilisation"][i],
            results["fairness"][i],
            results["overloaded"][i],
            results["runtime"][i]
        ]

        f.write(",".join(map(str, row)) + "\n")

print("\nScalability analysis completed.")
print(f"Results saved to:\n{SAVE_DIR}")