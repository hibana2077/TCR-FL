import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUTS_DIR = r"c:\Users\hiban\Desktop\code space\TCR-FL\out\outputs"

def load_results():
    results = {}
    if not os.path.exists(OUTPUTS_DIR):
        print(f"Directory not found: {OUTPUTS_DIR}")
        return results
        
    for dirname in os.listdir(OUTPUTS_DIR):
        dirpath = os.path.join(OUTPUTS_DIR, dirname)
        if os.path.isdir(dirpath):
            json_path = os.path.join(dirpath, "results.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    try:
                        data = json.load(f)
                        results[dirname] = data
                    except json.JSONDecodeError:
                        print(f"Error decoding {json_path}")
    return results

def plot_accuracy(results, run_names, title, filename):
    plt.figure(figsize=(10, 6))
    for name in run_names:
        if name in results:
            acc = results[name]["metrics"]["accuracy"]
            label = name
            if "tcr" in name:
                label = "TCR-FL"
                if "trimmed" in name: 
                    label += " + TrimmedMean"
            elif "fedavg" in name:
               label = "FedAvg"
            elif "krum" in name:
                label = "Krum" 
            elif "median" in name:
                label = "Median"
            elif "trimmed" in name:
                label = "TrimmedMean"
            
            plt.plot(acc, label=label)
    
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Global Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def calculate_asr(results, run_names):
    asr_data = []
    for name in run_names:
        if name in results:
            asr = results[name]["metrics"].get("asr", "N/A")
            label = name
             # Simplify labels
            if "tcr" in name:
                label = "TCR-FL"
                if "trimmed" in name: label += " + TrimmedMean"
            elif "fedavg" in name: label = "FedAvg"
            elif "krum" in name: label = "Krum" 
            elif "median" in name: label = "Median"
            elif "trimmed" in name: label = "TrimmedMean"
            
            asr_data.append({"Method": label, "ASR": asr})
    return pd.DataFrame(asr_data)

def plot_weights(results, run_name, filename):
    if run_name not in results:
        print(f"{run_name} not found")
        return
    
    data = results[run_name]
    alphas = data.get("alphas")
    if not alphas or alphas[0] is None:
        print(f"No alpha data for {run_name}")
        return

    # alphas is a list of lists (rounds x n_clients)
    # But wait, in the json snippet I saw earlier:
    # "alphas": [ null, ... ] for fedavg. 
    # Let's check a TCR run. 
    
    alphas = np.array([a for a in alphas if a is not None])
    if len(alphas) == 0:
        print("Empty alphas")
        return

    n_clients = len(alphas[0])
    malicious_ids = data["malicious_ids"]
    
    plt.figure(figsize=(10, 6))
    rounds = range(len(alphas))
    
    for i in range(n_clients):
        style = '--' if i in malicious_ids else '-'
        color = 'red' if i in malicious_ids else 'blue'
        alpha_val = 1.0 if i in malicious_ids else 0.3
        label = f"Client {i} ({'Malicious' if i in malicious_ids else 'Honest'})"
        
        # Plot only a few label for legend cleanliness
        if i not in malicious_ids and i > 0:
            label = None
        if i in malicious_ids and malicious_ids.index(i) > 0:
            label = None
            
        plt.plot(rounds, alphas[:, i], linestyle=style, color=color, alpha=alpha_val, label=label)

    plt.title(f"Client Weights over Time ({run_name})")
    plt.xlabel("Round")
    plt.ylabel("Weight (alpha)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def analyze_suppression(results, run_name, threshold=0.01):
    if run_name not in results: return "N/A"
    data = results[run_name]
    alphas = data.get("alphas")
    malicious_ids = data["malicious_ids"]
    
    if not alphas or alphas[0] is None: return "N/A"
    
    suppression_times = {}
    
    for mid in malicious_ids:
        for t, round_alphas in enumerate(alphas):
            if round_alphas is None: continue
            if round_alphas[mid] < threshold:
                suppression_times[mid] = t
                break
        if mid not in suppression_times:
            suppression_times[mid] = -1 # Never suppressed
            
    return suppression_times

def main():
    results = load_results()
    
    # --- Experiment A ---
    exp_a_runs = [
        "A_fedavg_T60",
        "A_krum_f2_T60",
        "A_median_T60",
        "A_trimmed02_T60",
        "A_tcr_l2_b09_l6_T60",
        "A_tcr_trimmed02_T60"
    ]
    
    # 1. Global Accuracy
    plot_accuracy(results, exp_a_runs, "Experiment A: Accuracy vs Round (Toy Synthetic)", "exp_a_accuracy.png")
    
    # 2. ASR Table
    asr_df = calculate_asr(results, exp_a_runs)
    print(" Experiment A: ASR")
    print(asr_df.to_markdown(index=False))
    
    # 3. Weight Curves for TCR
    plot_weights(results, "A_tcr_l2_b09_l6_T60", "exp_a_tcr_weights.png")
    
    # 4. Suppression Time
    suppression = analyze_suppression(results, "A_tcr_l2_b09_l6_T60")
    print("\nExperiment A: Time to Suppress (TCR-FL)")
    print(suppression)

    # --- Experiment B ---
    exp_b_runs = [
        "B_switch15_fedavg_T60",
        "B_switch15_tcr_T60"
    ]
    plot_accuracy(results, exp_b_runs, "Experiment B: Task Switching (Every 15 rounds)", "exp_b_accuracy.png")
    
    # --- Experiment C (Ablations) ---
    # Effect of Lambda
    exp_c_lambda = [
        "A_tcr_l2_b09_l6_T60", # Lambda 6 (Baseline from A) - Actually name is C_tcr_lam6_T60 in folder list? No, there is C_tcr_lam6_T60
        "C_tcr_lam1_T60",
        "C_tcr_lam3_T60",
        "C_tcr_lam6_T60", 
        "C_tcr_lam10_T60"
    ]
    # Check which ones exist
    existing_c = [r for r in exp_c_lambda if r in results]
    plot_accuracy(results, existing_c, "Experiment C: Effect of Lambda", "exp_c_lambda.png")

if __name__ == "__main__":
    main()
