import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def print_performance(results):
    for impl in results:
        lst = results[impl]
        print(impl.upper())
        for tup in lst:
            print(tup)


def extract_times():
    # Initialize storage for totals
    implementations = ["Sequential", "OpenMP", "CUDA"]
    csvs = ["performance_airport_terminal", "performance_campus", "performance_desert", "performance_elevator", "performance_forest", "performance_kitchen", "performance_lake", "performance_swimming_pool"]
    results = {impl: [] for impl in implementations}
    directories = []

    # Read CSVs and process
    for impl in implementations:
        results_dir = "../../results/"
        results_dir = os.path.join(results_dir, f"{impl.lower()}/performance_data/") #../results/cuda/performance_data
        for test_data in csvs:
            csv_path = os.path.join(results_dir, f"{test_data}.csv")
            # print("csv path: ", csv_path)
            df = pd.read_csv(csv_path, skiprows=1)

            # Ensure the column 'Image' exists
            if "Image" not in df.columns:
                raise ValueError(f"'Image' column not found in {csv_path}")

            # Ensure 'Total' row exists
            if "Total" in df["Image"].values:
                total_time = df[df["Image"] == "Total"]["Time (seconds)"].values[0]
                dataset = test_data[12:] + " dataset"
                tup = (dataset, total_time)
                results[impl].append(tup)
            else:
                raise ValueError(f"'Total' row not found in {csv_path}")
    return results

def plot_total_execution_times(results):
    implementations = list(results.keys())
    test_sets = [tup[0] for tup in results["Sequential"]]  # Extract test set names

    # Prepare data for plotting
    x = np.arange(len(test_sets))  # x-axis positions
    width = 0.25  # Bar width
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, impl in enumerate(implementations):
        times = [tup[1] for tup in results[impl]]
        ax.bar(x + i * width, times, width, label=impl)

    # Customize chart
    ax.set_xticks(x + width)
    ax.set_xticklabels(test_sets, rotation=45, ha="right")
    ax.set_ylabel("Total Execution Time (seconds)")
    ax.set_title("Total Execution Time Comparison")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_speedups(results):
    test_sets = [tup[0] for tup in results["Sequential"]]
    sequential_times = [tup[1] for tup in results["Sequential"]]

    # Compute speedups
    speedups = {"OpenMP": [], "CUDA": []}
    for impl in ["OpenMP", "CUDA"]:
        times = [tup[1] for tup in results[impl]]
        speedups[impl] = [sequential_times[i] / times[i] for i in range(len(test_sets))]

    # Plotting
    x = np.arange(len(test_sets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, impl in enumerate(speedups.keys()):
        ax.bar(x + i * width, speedups[impl], width, label=impl)

    # Customize chart
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(test_sets, rotation=45, ha="right")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup Relative to Sequential Implementation")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_execution_trends(results):
    implementations = list(results.keys())
    test_sets = [tup[0] for tup in results["Sequential"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for impl in implementations:
        times = [tup[1] for tup in results[impl]]
        ax.plot(test_sets, times, marker='o', label=impl)

    # Customize chart
    ax.set_xticks(test_sets)
    ax.set_xticklabels(test_sets, rotation=45, ha="right")
    ax.set_ylabel("Total Execution Time (seconds)")
    ax.set_title("Execution Time Trends Across Test Sets")
    ax.legend()

    plt.tight_layout()
    plt.show()


performance = extract_times()
print_performance(performance)
plot_total_execution_times(performance)
plot_speedups(performance)
plot_execution_trends(performance)



# # Compute speedup
# speedup = {
#     "OpenMP": [results["Sequential"][i] / results["OpenMP"][i] for i in range(len(directories))],
#     "CUDA": [results["Sequential"][i] / results["CUDA"][i] for i in range(len(directories))],
# }


# # Total execution time plot
# plt.figure(figsize=(10, 6))
# x = range(len(directories))
# for impl in implementations:
#     plt.bar([i + (0.2 * idx) for idx, i in enumerate(x)], results[impl], width=0.2, label=impl)

# plt.xticks([i + 0.2 for i in x], directories, rotation=45)
# plt.ylabel("Execution Time (seconds)")
# plt.title("Total Execution Time Comparison")
# plt.legend()
# plt.tight_layout()
# plt.show()


# # Speedup plot
# plt.figure(figsize=(10, 6))
# x = range(len(directories))
# for idx, impl in enumerate(["OpenMP", "CUDA"]):
#     plt.bar([i + (0.2 * idx) for i in x], speedup[impl], width=0.2, label=impl)

# plt.xticks([i + 0.1 for i in x], directories, rotation=45)
# plt.ylabel("Speedup")
# plt.title("Speedup Comparison (Relative to Sequential)")
# plt.legend()
# plt.tight_layout()
# plt.show()
