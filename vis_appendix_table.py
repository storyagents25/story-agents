import pandas as pd
import glob
import re

# Define categories for different experiments
CATEGORIES = {
    "homogeneous": [
        "game_results_same_story_*_ag4_ro5_end10_mult1.5.csv",
        "game_results_same_story_*_ag16_ro5_end10_mult1.5.csv",
        "game_results_same_story_*_ag32_ro5_end10_mult1.5.csv",
    ],
    "heterogeneous": [
        "game_results_different_story_ag4_ro5_end10_mult1.5.csv",
    ],
    "robustness": [
        "game_results_bad_apple_*_ag4_ro5_end10_mult1.5.csv",
    ],
}

# Define baseline and meaningful story prompts
BASELINE_STORIES = ["noinstruct", "nsCarrot", "maxreward", "nsPlumber"]
MEANINGFUL_STORIES = ["OldManSons", "Odyssey", "Soup", "Peacemaker",
                "Musketeers", "Teamwork", "Spoons", "Turnip"]

def truncate_float(val, decimals=2):
    factor = 10 ** decimals
    return int(val * factor) / factor

# Extract agent size from filename pattern
def extract_agent_size(pattern):
    match = re.search(r'_ag(\d+)_', pattern)
    return match.group(1) if match else "Unknown"

# Dictionary to store collaboration scores
collab_scores = {story: {"Homogeneous_4": None, "Homogeneous_16": None, "Homogeneous_32": None,
                        "Robustness_4": None, "Heterogeneous_4": None} for story in BASELINE_STORIES + MEANINGFUL_STORIES}

# Process experiment data and extract Mean ± SD for each story
for category, patterns in CATEGORIES.items():
    for pattern in patterns:
        files = glob.glob(pattern)
        if not files:
            print(f"No files found for pattern: {pattern}")
            continue
        print(f"Category: {pattern} | Found {len(files)} files.")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        agent_size = extract_agent_size(pattern)
        
        if category in ["homogeneous", "robustness"]:
            df = df[df["Round"] == "final"]  # Filter for final round Collaboration Score
            stats = df.groupby("PromptType")["CollaborationScore"].agg(["mean", "std"]).reset_index()
        else:
            df = df[df["Round"] != "final"].copy()  # Exclude final round
            df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
            df.dropna(subset=["Round"], inplace=True)
            df = df.loc[df.groupby(["Game", "AgentName"])["Round"].idxmax()].copy()  # Take last available round per agent
            stats = df.groupby("PromptType")["CumulativePayoff"].agg(["mean", "std"]).reset_index()

        # Map results to correct table column
        column_key = f"{category.capitalize()}_{agent_size}"
        
        for _, row in stats.iterrows():
            story = row["PromptType"]

            
            if story in collab_scores:
                #mean = truncate_float(row['mean'], 2)
                #std = truncate_float(row['std'], 2)
                #collab_scores[story][column_key] = f"{mean:.2f} ± {std:.2f}"
                mean_val = row['mean']
                std_val = row['std']

                # Format dynamically: more decimals for small std
                if std_val < 0.01:
                    formatted = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    formatted = f"{mean_val:.2f} ± {std_val:.2f}"
                collab_scores[story][column_key] = formatted


latex_output = """\\begin{table*}[t]
    \\centering
    \\caption{Mean ± standard deviation of final Collaboration Scores (for homogeneous and robustness agents) and final Cumulative Payoffs (for heterogeneous agents) across all story prompts. Values are shown with higher decimal precision where variation is small, to reflect statistically meaningful differences observed in pairwise confidence intervals.}
    \\setlength{\\tabcolsep}{8pt} 
    \\renewcommand{\\arraystretch}{1.3} 
    \\fontsize{11pt}{13pt}\\selectfont 
    \\resizebox{\\textwidth}{!}{
    \\begin{tabular}{llccccc}
        \\toprule
        \\multirow{2}{*}{\\textbf{Story Type}} & \\multirow{2}{*}{\\textbf{Story Prompt}} & \\multicolumn{3}{c}{\\textbf{Homogeneous Agents}} & \\textbf{Robustness} & \\textbf{Heterogeneous} \\\\
        \\cmidrule(lr){3-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-7}
        & & \\textbf{N=4} & \\textbf{N=16} & \\textbf{N=32} & \\textbf{N=4} & \\textbf{N=4} \\\\
        \\midrule
"""

# Add Baseline Stories Section
latex_output += "        \\multirow{4}{*}{\\centering \\textbf{Baseline Stories}}  \n"
for i, story in enumerate(BASELINE_STORIES):
    row_data = [collab_scores[story].get(col, "N/A") for col in ["Homogeneous_4", "Homogeneous_16", "Homogeneous_32", "Robustness_4", "Heterogeneous_4"]]
    prefix = "        & "
    latex_output += f"{prefix}{story}  & {' & '.join(row_data)} \\\\\n"

latex_output += "        \\midrule\n"

# Add Meaningful Stories Section
latex_output += "        \\multirow{8}{*}{\\centering \\textbf{Meaningful Stories}}  \n"
for i, story in enumerate(MEANINGFUL_STORIES):
    row_data = [collab_scores[story].get(col, "N/A") for col in ["Homogeneous_4", "Homogeneous_16", "Homogeneous_32", "Robustness_4", "Heterogeneous_4"]]
    prefix = "        & "
    latex_output += f"{prefix}{story}  & {' & '.join(row_data)} \\\\\n"

latex_output += """        \\bottomrule
    \\end{tabular}
    }
    \\label{tab:all_agents_scores}
\\end{table*}
"""
print(latex_output)
# Save to file
with open("all_agents_table.tex", "w") as f:
    f.write(latex_output)