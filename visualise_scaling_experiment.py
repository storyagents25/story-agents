import matplotlib.pyplot as plt
import pandas as pd
import glob
import matplotlib.lines as mlines

# Define agent sizes used in the experiment
agent_sizes = [4, 16, 32]

# Define color gradients for baseline (blue) and meaningful stories (pink)
BLUE_SHADES = ["#87CEFA", "#4682B4", "#4169E1", "#27408B"]  # Light to Dark Blue
PINK_SHADES = ["#FFB3E6", "#FF99CC", "#FF66B3", "#FF4D9E", "#F02278", "#D81B60", "#B83B7D", "#B22272"]  # Light to Dark Pink

# Define file categories for different temperatures
CATEGORY_GROUPS = {
    "temp_0.6": {
        4: "game_results_same_story_*_ag4_ro5_end10_mult1.5.csv",
        16: "game_results_same_story_*_ag16_ro5_end10_mult1.5.csv",
        32: "game_results_same_story_*_ag32_ro5_end10_mult1.5.csv",
    },
}

def process_category(temp_label, CATEGORIES):
    """Processes a single temperature condition and generates a visualization."""
    story_scores = {}

    # Load CSV data and extract mean collaboration scores
    for agent_count, pattern in CATEGORIES.items():
        files = glob.glob(pattern)
        if not files:
            print(f"No files found for pattern: {pattern}")
            continue

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        # Compute mean collaboration scores per story
        story_means = df.groupby("PromptType")["CollaborationScore"].mean()
        story_scores[agent_count] = story_means.to_dict()  # Store scores

    # Extract the order of stories as they appear at N=4 in ascending order
    starting_order = sorted(story_scores[4].items(), key=lambda x: x[1])  
    starting_stories = [story for story, _ in starting_order]

    print(f"Processing {temp_label}: starting_stories = {starting_stories}")

    # Dynamically assign colors based on order in starting_stories
    COLOR_DICT = {}
    blue_idx, pink_idx = 0, 0  # Track index for blue and pink shades

    for story in starting_stories:
        if story in ["noinstruct", "nsCarrot", "nsPlumber", "maxreward"]:  # Baseline stories
            COLOR_DICT[story] = BLUE_SHADES[blue_idx]
            blue_idx += 1  # Move to next darker shade
        else:  # Meaningful stories
            COLOR_DICT[story] = PINK_SHADES[pink_idx]
            pink_idx += 1  # Move to next darker shade

    # Plot settings
    plt.figure(figsize=(12, 6))
    legend_handles = []

    # Plot each story's progression across agent sizes
    for story in COLOR_DICT.keys():  # Only plot stories in COLOR_DICT
        positions = []
        
        for agent_count in agent_sizes:
            if agent_count in story_scores and story in story_scores[agent_count]:
                x_pos = story_scores[agent_count][story]
                y_pos = agent_sizes.index(agent_count)  
                positions.append((x_pos, y_pos))

                # Scatter plot for each point
                plt.scatter(x_pos, y_pos, s=70, facecolors="none", edgecolors=COLOR_DICT[story], linewidths=1.5, label=story if agent_count == 4 else "")

        if len(positions) > 1:
            x_vals, y_vals = zip(*positions)
            plt.plot(x_vals, y_vals, linestyle="dashed", color=COLOR_DICT[story], alpha=0.7)
        
    for story in starting_stories:
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", linestyle="None", markersize=8, color=COLOR_DICT.get(story, "#888888"), label=story
            )
        )

    # Customizing plot
    plt.xlabel("Mean Collaboration Score", fontsize=18, labelpad=15)
    plt.ylabel("Agent Size", fontsize=18, labelpad=15)
    plt.yticks(range(len(agent_sizes)), [f"N = {n}" for n in agent_sizes], fontsize=14, weight="bold")
    plt.title(f"Scaling Experiment", fontsize=20, weight="bold", pad=20)
    plt.grid(axis="y", linestyle="dotted")

    plt.legend(
        handles=legend_handles, title="Story", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12
    )

    plt.tight_layout()

    # Save as Pdf
    filename_base = f"scaling_experiment_collab_score_{temp_label}"
    plt.savefig(f"{filename_base}.pdf", bbox_inches="tight", format="pdf")

    print(f"Scaling experiment figures saved as {filename_base}.pdf")

    plt.show()


# Run the process for each category
for temp_label, category_dict in CATEGORY_GROUPS.items():
    process_category(temp_label, category_dict)
