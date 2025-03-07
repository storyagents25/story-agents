import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define color dictionary for plot consistency

COLOR_DICT = {
    # Baseline condition (Shades of Blue)
    "noinstruct": "#1f77b4",  

    # Nonsense stories (Shades of Orange)
    "nsCarrot": "#FFB347",  
    "nsPlumber": "#FF7F00",  

    # Meaningful stories (Shades of Purple/Pink)
    "OldManSons": "#ffb3e6",  
    "Peacemaker": "#ff66b3",  
    "Soup": "#e377c2",  
    "Spoons": "#d147a3",  
    "Teamwork": "#c71585",  
    "Turnip": "#800040",  
    "Musketeers": "#660033",  
    "Odyssey": "#99004d",  

    # Greedy story (Shades of Green)
    "maxreward": "#2ca02c",
}


def load_csv_files(pattern):
    """Loads all CSV files matching a pattern and merges them into a DataFrame."""
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def preprocess_data(df, metric):
    """
    Prepares data by filtering only final round rows and converting columns to numeric types.
    Metric can be 'CollaborationScore' or 'CumulativePayoff'.
    """
    if df is None or df.empty:
        return None
    
    if metric == "CollaborationScore":
        df = df[df["Round"] == "final"].copy()
    else:  # "CumulativePayoff"
        df = df[df["Round"] != "final"].copy()
        df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
        df.dropna(subset=["Round"], inplace=True)
        df = df.loc[df.groupby(["Game", "AgentName"])["Round"].idxmax()].copy()

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df.dropna(subset=[metric], inplace=True)
    
    return df


def plot_violin(df, metric, title, output_jpg, plot_mean_line=True, show_legend=False):
    """
    Plots a violin plot for CollaborationScore or CumulativePayoff.
    """
    if df is None or df.empty:
        print(f"No data to visualize for {title}")
        return

    # Compute x-axis order based on mean metric values
    order = (df.groupby("PromptType")[metric]
            .mean()
            .sort_values(ascending=False)
            .index.tolist())

    plt.figure(figsize=(10, 6))

    # Define color palette
    palette = {cat: COLOR_DICT.get(cat, "#888888") for cat in order}
    
    # Violin plot with embedded box plot    
    ax = sns.violinplot(
        data=df,
        x="PromptType",
        y=metric,
        hue="PromptType",
        palette=palette,
        inner="box",
        dodge=False,
        order=order,
        bw_adjust=5, # Adjusting KDE bandwidth
        scale="area", # Uniform width across all violins
    )

    if ax.get_legend():
        ax.get_legend().remove()

    # Overlay scatter points
    sns.stripplot(
        data=df,
        x="PromptType",
        y=metric,
        color="black",
        dodge=False,
        alpha=0.2,
        size=2,
        zorder=2,
        order=order
    )

    # (Optional) Plot mean trend line
    if plot_mean_line:
        means = df.groupby("PromptType")[metric].mean().loc[order]
        x_positions = list(range(len(order)))
        plt.plot(
            x_positions, means.values, marker='o',
            color='black', linestyle='-', linewidth=2,
            markersize=6, alpha=0.5, label="Mean Trend"
        )
        if show_legend:
            plt.legend(["Mean Trend"])

    # Customize plot

    if metric == "CollaborationScore":
        plt.ylim(0, 1.3)  # Set range for Collaboration Score
    elif metric == "CumulativePayoff":
        plt.ylim(0, 120) #Set range for Cumulative Payoff
    plt.xlabel("Story Prompt", fontsize=18, labelpad=15)  # Add padding between xlabel and xticks
    
    ylabel_text = "Payoff per Agent" if metric == "CumulativePayoff" else "Collaboration Score"

    plt.ylabel(f"{ylabel_text}", fontsize=18, labelpad=15)  # Add padding between ylabel and yticks
    plt.title(title, fontsize=20, weight="bold" , pad=20)  # Make title bold
    plt.xticks(rotation=90 if len(order) > 5 else 0, fontsize=14)
    plt.yticks(fontsize=14)

    sns.despine()
    plt.grid(False)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_jpg}")
    plt.show()

if __name__ == "__main__":
    # Define experiment types and file patterns
    CATEGORIES = {
        "same_story_4_agents": "game_results_same_story_*_ag4_ro5_end10_mult1.5.csv",
        "same_story_16_agents": "game_results_same_story_*_ag16_ro5_end10_mult1.5.csv",
        "same_story_32_agents": "game_results_same_story_*_ag32_ro5_end10_mult1.5.csv",
        "different_story_16_agents": "game_results_different_story_ag16_ro5_end10_mult1.5.csv",
        "bad_apple_16_agents": "game_results_bad_apple_*_ag16_ro5_end10_mult1.5.csv"
    }

    for category, pattern in CATEGORIES.items():
        agent_count = category.split("_")[-2]  # Extract agent count dynamically

        if "different_story" in category:
            # Different story -> Cumulative Payoff
            csv_files = glob.glob(pattern)
            for csv_file in csv_files:
                output_file = csv_file.replace("game_results", "cumulative_payoffs").replace(".csv", ".jpg")
                df = preprocess_data(pd.read_csv(csv_file), "CumulativePayoff")

                title = f"Cumulative Payoff ({agent_count} Agents)"

                plot_violin(df, "CumulativePayoff", title, output_file)
        else:
            # Same story & bad apple -> Collaboration Score
            df = load_csv_files(pattern)
            df = preprocess_data(df, "CollaborationScore")
            if df is not None:
                output_file = f"{category}_collaboration_scores.jpg"

                if "bad_apple" in category:
                    title = f"Collaboration Score - Robustness ({agent_count} Agents)"
                else:
                    title = f"Final Collaboration Score ({agent_count} Agents)"

                plot_violin(df, "CollaborationScore", title, output_file)
