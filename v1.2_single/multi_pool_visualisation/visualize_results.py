"""
Data visualization and analysis for multi-pool story agents experiments.
Generates plots for both single-pool and multi-pool results.
"""

import os
import glob
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

# Configure matplotlib for vector output
import matplotlib as mpl
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# Story categorization and color schemes
BASELINE_STORIES = ["noinstruct", "nsCarrot", "maxreward", "nsPlumber"]
MEANINGFUL_STORIES = ["OldManSons", "Odyssey", "Soup", "Peacemaker", "Musketeers", "Teamwork", "Spoons", "Turnip"]

# Color schemes
BLUE_SHADES = ["#87CEFA", "#4682B4", "#4169E1", "#27408B"]  # Baseline stories
PINK_SHADES = ["#FFB3E6", "#FF99CC", "#FF66B3", "#FF4D9E", "#F02278", "#D81B60", "#B83B7D", "#B22272"]  # Meaningful stories

COLOR_DICT = {
    # Baseline stories (Blue shades)
    "maxreward": "#87CEFA",
    "noinstruct": "#4682B4",
    "nsCarrot": "#4169E1",
    "nsPlumber": "#27408B",
    # Meaningful stories (Pink/Purple shades)
    "Odyssey": "#FFB3E6",
    "Soup": "#FF99CC",
    "Peacemaker": "#FF66B3",
    "Musketeers": "#FF4D9E",
    "Teamwork": "#F02278",
    "Spoons": "#D81B60",
    "Turnip": "#B83B7D",
    "OldManSons": "#B22272",
}


def load_csv_files(pattern: str) -> pd.DataFrame:
    """Load all CSV files matching a pattern and merge them."""
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} files for pattern: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def preprocess_data(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    """Prepare data by filtering and converting columns to numeric types."""
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


def plot_violin(df: pd.DataFrame, metric: str, title: str, output_path: str, 
                plot_mean_line: bool = True, show_legend: bool = False):
    """Generate violin plots showing distribution of scores or payoffs."""
    if df is None or df.empty:
        print(f"No data to visualize for {title}")
        return

    # Order stories by mean performance
    order = df.groupby("PromptType")[metric].mean().sort_values(ascending=True).index.tolist()

    plt.figure(figsize=(12, 7))

    # Define color palette
    palette = {cat: COLOR_DICT.get(cat, "#888888") for cat in order}

    # Create violin plot
    ax = sns.violinplot(
        data=df,
        x="PromptType",
        y=metric,
        hue="PromptType",
        palette=palette,
        inner="box",
        dodge=False,
        order=order,
        bw_adjust=5,
        scale="area",
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

    # Plot mean trend line
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
        plt.ylim(0, 1.3)
    elif metric == "CumulativePayoff":
        plt.ylim(0, 120)
    
    plt.xlabel("Story Prompt", fontsize=18, labelpad=15)
    ylabel_text = "Payoff per Agent" if metric == "CumulativePayoff" else "Collaboration Score"
    plt.ylabel(ylabel_text, fontsize=18, labelpad=15)
    plt.title(title, fontsize=20, weight="bold", pad=20)
    plt.xticks(rotation=90 if len(order) > 5 else 0, fontsize=14)
    plt.yticks(fontsize=14)

    sns.despine()
    plt.grid(False)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, bbox_inches='tight', format='pdf', transparent=False)
    print(f"Violin plot saved as {output_path}")
    plt.close()


def plot_scaling_experiment(agent_sizes: List[int] = [4, 16, 32]) -> None:
    """Generate scaling experiment plot showing how cooperation changes with group size."""
    
    story_scores = {}
    pattern_template = "experiments/single_pool/same_story/game_results_single_pool_same_story_*_ag{N}_ro5_end10_mult1.5.csv"
    
    # Load data for each agent size
    for agent_count in agent_sizes:
        pattern = pattern_template.format(N=agent_count)
        df = load_csv_files(pattern)
        if df is None or df.empty:
            continue
        
        df = preprocess_data(df, "CollaborationScore")
        if df is not None:
            story_means = df.groupby("PromptType")["CollaborationScore"].mean()
            story_scores[agent_count] = story_means.to_dict()

    if not story_scores:
        print("No data found for scaling experiment")
        return

    # Get story order from 4-agent case
    if 4 in story_scores:
        starting_order = sorted(story_scores[4].items(), key=lambda x: x[1])
        starting_stories = [story for story, _ in starting_order]
    else:
        starting_stories = list(next(iter(story_scores.values())).keys())

    # Assign colors
    color_dict_local = {}
    blue_idx, pink_idx = 0, 0
    
    for story in starting_stories:
        if story in BASELINE_STORIES:
            color_dict_local[story] = BLUE_SHADES[blue_idx % len(BLUE_SHADES)]
            blue_idx += 1
        else:
            color_dict_local[story] = PINK_SHADES[pink_idx % len(PINK_SHADES)]
            pink_idx += 1

    # Create plot
    plt.figure(figsize=(12, 6))
    legend_handles = []

    # Plot each story's progression
    for story in starting_stories:
        if story not in color_dict_local:
            continue
            
        positions = []
        for agent_count in agent_sizes:
            if agent_count in story_scores and story in story_scores[agent_count]:
                x_pos = story_scores[agent_count][story]
                y_pos = agent_sizes.index(agent_count)
                positions.append((x_pos, y_pos))
                
                # Scatter plot for each point
                plt.scatter(x_pos, y_pos, s=70, facecolors="none", 
                          edgecolors=color_dict_local[story], linewidths=1.5)

        if len(positions) > 1:
            x_vals, y_vals = zip(*positions)
            plt.plot(x_vals, y_vals, linestyle="dashed", 
                   color=color_dict_local[story], alpha=0.7)

    # Create legend
    for story in starting_stories:
        if story in color_dict_local:
            legend_handles.append(
                mlines.Line2D([], [], marker="o", linestyle="None", markersize=8,
                            color=color_dict_local[story], label=story)
            )

    # Customize plot
    plt.xlabel("Mean Collaboration Score", fontsize=18, labelpad=15)
    plt.ylabel("Agent Size", fontsize=18, labelpad=15)
    plt.yticks(range(len(agent_sizes)), [f"N = {n}" for n in agent_sizes], 
              fontsize=14, weight="bold")
    plt.title("Scaling Experiment", fontsize=20, weight="bold", pad=20)
    plt.grid(axis="y", linestyle="dotted")
    
    plt.legend(handles=legend_handles, title="Story", bbox_to_anchor=(1.05, 1), 
              loc="upper left", fontsize=12)

    sns.despine()
    plt.tight_layout()

    # Save plot
    vis_dir = Path("experiments") / "vis" / "single_pool" / "same_story"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / "scaling_experiment_collab_score.pdf"
    plt.savefig(str(output_path), bbox_inches="tight", format="pdf")
    print(f"Scaling experiment plot saved as {output_path}")
    plt.close()


def plot_multi_pool_violin_collaboration():
    """Generate violin plot for multi-pool collaboration scores."""
    pattern = "experiments/multi_pool/same_story/game_results_multi_pool_2_same_story_*_ag4_ro5_end10_mult1.5.csv"
    df = load_csv_files(pattern)
    
    if df.empty:
        print("No multi-pool same story data found")
        return
    
    df = preprocess_data(df, "CollaborationScore")
    if df is None:
        return
    
    output_dir = Path("experiments") / "vis" / "multi_pool" / "same_story"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multipool_same_story_violin_collaboration.pdf"
    
    plot_violin(df, "CollaborationScore", "Multi-Pool Homogeneous", str(output_path))


def plot_multi_pool_scatter_collaboration():
    """Generate scatter plot for multi-pool collaboration vs global pool allocation."""
    pattern = "experiments/multi_pool/same_story/game_results_multi_pool_2_same_story_*_ag4_ro5_end10_mult1.5.csv"
    df = load_csv_files(pattern)
    
    if df.empty:
        print("No multi-pool same story data found")
        return
    
    # Calculate collaboration scores
    df_final = df[df["Round"] == "final"].copy()
    collab_scores = df_final.groupby("PromptType")["CollaborationScore"].mean()
    
    # Calculate global pool proportions
    df_rounds = df[df["Round"] != "final"].copy()
    df_rounds["GlobalContrib"] = pd.to_numeric(df_rounds["GlobalContrib"], errors="coerce")
    df_rounds["TotalContrib"] = pd.to_numeric(df_rounds["TotalContrib"], errors="coerce")
    
    global_props = {}
    for story in collab_scores.index:
        story_data = df_rounds[df_rounds["PromptType"] == story]
        if len(story_data) > 0:
            story_data = story_data.copy()
            story_data["GlobalProp"] = story_data["GlobalContrib"] / story_data["TotalContrib"]
            story_data["GlobalProp"] = story_data["GlobalProp"].fillna(0.5)
            global_props[story] = story_data["GlobalProp"].mean()
    
    # Filter to valid stories
    valid_stories = [s for s in collab_scores.index if s in global_props]
    xs = [global_props[s] for s in valid_stories]
    ys = [collab_scores[s] for s in valid_stories]
    
    # Create scatter plot
    plt.figure(figsize=(10, 7))
    colors = [COLOR_DICT.get(s, "#888888") for s in valid_stories]
    plt.scatter(xs, ys, s=120, c=colors, edgecolors="white", linewidths=1.5, alpha=1.0)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label=s,
               markerfacecolor=COLOR_DICT.get(s, "#888888"),
               markeredgecolor="white", markersize=10)
        for s in valid_stories
    ]
    plt.legend(handles=legend_elems, bbox_to_anchor=(1.05, 1), loc="upper left",
               title="Story", fontsize=18, title_fontsize=20, frameon=True)
    
    plt.title("Multi-Pool Homogeneous\nCollaboration Score vs Global-Pool Contribution Fraction",
              fontsize=20, fontweight="bold", pad=20)
    plt.xlabel("Proportion of Contributions to Global Pool", fontsize=22)
    plt.ylabel("Collaboration Score", fontsize=22)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(ls="--", alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("experiments") / "vis" / "multi_pool" / "same_story"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multipool_same_story_scatter_collab_vs_global.pdf"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight", pad_inches=0.2)
    print(f"Multi-pool scatter plot saved as {output_path}")
    plt.close()


def plot_multi_pool_violin_payoff():
    """Generate violin plot for multi-pool payoffs by story."""
    pattern = "experiments/multi_pool/different_story/game_results_multi_pool_2_different_story_ag4_ro5_end10_mult1.5.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("No multi-pool different story data found")
        return
    
    df = pd.read_csv(files[0])
    df_rounds = df[df["Round"] != "final"].copy()
    df_rounds["CumulativePayoff"] = pd.to_numeric(df_rounds["CumulativePayoff"], errors="coerce")
    df_rounds.dropna(subset=["CumulativePayoff"], inplace=True)
    
    # Get final payoffs
    df_final_payoffs = df_rounds.loc[df_rounds.groupby(["Game", "AgentName"])["Round"].idxmax()].copy()
    df_final_payoffs = df_final_payoffs[df_final_payoffs["PromptType"] != "All"]
    
    output_dir = Path("experiments") / "vis" / "multi_pool" / "different_story"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multipool_different_story_violin_payoff.pdf"
    
    plot_violin(df_final_payoffs, "CumulativePayoff", "Multi-Pool Heterogeneous", str(output_path))


def generate_single_pool_visualizations():
    """Generate all single-pool visualizations."""
    print("Generating single-pool visualizations...")
    
    # Violin plots for same_story experiments
    for agent_size in [4, 16, 32]:
        pattern = f"experiments/single_pool/same_story/game_results_single_pool_same_story_*_ag{agent_size}_ro5_end10_mult1.5.csv"
        df = load_csv_files(pattern)
        df = preprocess_data(df, "CollaborationScore")
        
        if df is not None:
            vis_dir = Path("experiments") / "vis" / "single_pool" / "same_story"
            vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = vis_dir / f"collaboration_violin_same_story_{agent_size}_agents.pdf"
            title = f"Homogeneous Experiment (N={agent_size})"
            plot_violin(df, "CollaborationScore", title, str(output_path))
    
    # Violin plot for bad_apple
    pattern = "experiments/single_pool/bad_apple/game_results_single_pool_bad_apple_*_ag4_ro5_end10_mult1.5.csv"
    df = load_csv_files(pattern)
    df = preprocess_data(df, "CollaborationScore")
    
    if df is not None:
        vis_dir = Path("experiments") / "vis" / "single_pool" / "bad_apple"
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_path = vis_dir / "collaboration_violin_bad_apple_4_agents.pdf"
        plot_violin(df, "CollaborationScore", "Robustness", str(output_path))
    
    # Violin plot for different_story
    pattern = "experiments/single_pool/different_story/game_results_single_pool_different_story_ag4_ro5_end10_mult1.5.csv"
    files = glob.glob(pattern)
    
    if files:
        df = pd.read_csv(files[0])
        df = preprocess_data(df, "CumulativePayoff")
        
        if df is not None:
            vis_dir = Path("experiments") / "vis" / "single_pool" / "different_story"
            vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = vis_dir / "cumulative_payoffs_different_story_4_agents.pdf"
            plot_violin(df, "CumulativePayoff", "Heterogeneous Experiment", str(output_path))
    
    # Scaling experiment
    plot_scaling_experiment()


def generate_multi_pool_visualizations():
    """Generate all multi-pool visualizations."""
    print("Generating multi-pool visualizations...")
    
    plot_multi_pool_violin_collaboration()
    plot_multi_pool_scatter_collaboration()
    plot_multi_pool_violin_payoff()



def generate_summary_table():
    """Generate LaTeX summary table of all results."""
    print("Generating summary statistics table...")
    
    # This would be a complex function to generate the LaTeX table
    # Similar to the notebook version but adapted for the new file structure
    # Implementation details would depend on specific requirements
    
    print("Summary table generation not yet implemented")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate visualizations for story agents experiments")
    parser.add_argument("--exp_type", choices=["single_pool", "multi_pool", "all"], 
                       default="all", help="Type of experiment to visualize")
    parser.add_argument("--generate_all", action="store_true", 
                       help="Generate all visualizations")
    parser.add_argument("--output_dir", type=str, default="experiments/vis",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    if args.generate_all or args.exp_type in ["single_pool", "all"]:
        generate_single_pool_visualizations()
    
    if args.generate_all or args.exp_type in ["multi_pool", "all"]:
        generate_multi_pool_visualizations()
    
    if args.generate_all:
        generate_summary_table()
    
    print("Visualization generation completed!")


if __name__ == "__main__":
    main()