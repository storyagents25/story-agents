# -----------------------------
# Global Settings and Imports
# -----------------------------

import argparse
from experiment_runner import run_same_story_experiment, run_different_story_experiment

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description='Negotiation Experiment')
parser.add_argument('--exp_type', type=str, choices=['same_story', 'different_story', 'bad_apple'], required=True)
parser.add_argument('--story_index', type=int, default=0, help="Index of the story for the same_story experiment")

args = parser.parse_args()
exp_type = args.exp_type

# -----------------------------
# Experiment Configurations
# -----------------------------
num_rounds_list = [5]  # rounds per game
endowment_list = [10]
multiplier_list = [1.5]
if exp_type in ["same_story", "bad_apple"]:
    num_games = 100
    if exp_type == "same_story":
        num_agents_list = [4 , 16 , 32]
    else:
        num_agents_list = [16]
elif exp_type == 'different_story':
    num_games = 400
    num_agents_list = [16]

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if exp_type in ["same_story", "bad_apple"]:
        run_same_story_experiment(
            is_bad_apple=(exp_type == "bad_apple"),
            story_index=args.story_index,
            num_rounds_list=num_rounds_list,
            endowment_list=endowment_list,
            multiplier_list=multiplier_list,
            num_games=num_games,
            num_agents_list=num_agents_list,
            exp_type = exp_type
        )
    elif exp_type == "different_story":
        run_different_story_experiment(
            num_rounds_list, endowment_list, multiplier_list, num_games, num_agents_list, exp_type
        )
