import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

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
        results_dir = os.path.join(results_dir, f"{impl.lower()}/performance_data/") 
        for test_data in csvs:
            csv_path = os.path.join(results_dir, f"{test_data}.csv")
            df = pd.read_csv(csv_path, skiprows=1)

            if "Image" not in df.columns:
                raise ValueError(f"'Image' column not found in {csv_path}")

            if "Total" in df["Image"].values:
                total_time = df[df["Image"] == "Total"]["Time (seconds)"].values[0]
                dataset = test_data[12:] + " dataset"
                tup = (dataset, total_time)
                results[impl].append(tup)
            else:
                raise ValueError(f"'Total' row not found in {csv_path}")
    print(results)
    return results

def plot_total_execution_times(results):
    implementations = list(results.keys())
    test_sets = [tup[0] for tup in results["Sequential"]]

    x = np.arange(len(test_sets)) 
    width = 0.25  
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, impl in enumerate(implementations):
        times = [tup[1] for tup in results[impl]]
        ax.bar(x + i * width, times, width, label=impl)

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


# Generate tables for each method
one_thread = [14.2091,
11.0442,
12.0534,
3.67932,
7.31491,
14.7302,
3.83371,
2.30974]

two_threads = [9.70634,
7.5165,
8.12776,
2.44271,
4.92037,
10.0251,
2.63399,
1.57296]

three_threads = [6.0041,
4.59131,
4.93571,
1.49061,
3.05492,
6.05777,
1.61262,
0.922758]

four_threads = [5.97598,
4.63788,
4.95254,
1.48201,
3.05902,
6.13232,
1.63557,
0.94697]


def plot_speedup_graph():
    one_thread = [54.8714,
43.4423,
47.7161,
15.8674,
30.039,
57.4081,
15.8753,
10.5111]
    two_threads = [32.6689,
26.0217,
28.1134,
9.31847,
17.7787,
34.4079,
9.64745,
6.35959]
    four_threads = [22.5398,
18.0288,
19.488,
6.434,
11.9573,
23.7573,
6.52952,
4.15753]
    eight_threads = [18.348,
14.1643,
15.2821,
4.96235,
9.62644,
19.2303,
5.28523,
3.60523]
    
    sixteen_threads = [16.7057,
13.159,
14.2689,
4.5622,
8.85454,
17.4015,
4.60033,
2.95985,
]
    
    thirytwo_threads = [16.8371,
13.1213,
14.2011,
4.50177,
8.89764,
17.375,
4.62747,
2.92217]

    # Calculate the average times
    avg_1_thread = sum(one_thread) / len(one_thread)
    avg_2_threads = sum(two_threads) / len(two_threads)
    avg_4_threads = sum(four_threads) / len(four_threads)
    avg_8_threads = sum(eight_threads) / len(eight_threads)
    avg_16_threads = sum(sixteen_threads) / len(sixteen_threads)
    avg_32_threads = sum(thirytwo_threads) / len(thirytwo_threads)
    # Compute speedup
    speedup = [
        avg_1_thread / avg_1_thread,  
        avg_1_thread / avg_2_threads,
        avg_1_thread / avg_4_threads,
        avg_1_thread / avg_8_threads,
        avg_1_thread / avg_16_threads,
        avg_1_thread / avg_32_threads
    ]

    # Plotting the graph
    threads = [1, 2, 4, 8, 16, 32]

    plt.figure(figsize=(8, 6))
    plt.plot(threads, speedup, marker='o', label="Speedup")
    plt.xlabel("Number of Threads", fontsize=14)
    plt.ylabel("Speedup (relative to 1 thread)", fontsize=14)
    plt.title("Speedup vs Number of Threads", fontsize=16)
    plt.xticks(threads, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

plot_speedup_graph()





performance = extract_times()
print_performance(performance)
# plot_table(performance, 'Sequential', '../../figures/sequential_table.png')
# plot_table(performance, 'OpenMP', '../../figures/openmp_table.png')
# plot_table(performance, 'CUDA', '../../figures/cuda_table.png')

plot_total_execution_times(performance)
plot_speedups(performance)
# plot_execution_trends(performance)



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
