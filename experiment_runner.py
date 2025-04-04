import os
import sys
import glob
import itertools
import statistics
import csv
from agent_utils import (
    run_single_game,
    run_single_game_random_story,
    load_intermediate_results,
    get_system_prompt,
    save_intermediate_results,
    compute_and_print_statistics,
)

CSV_HEADER = ["Game", "PromptType", "Round", "AgentName", "Contribution", "RoundPayoff", "CumulativePayoff", "CollaborationScore"]


def run_same_story_experiment(is_bad_apple, story_index, num_rounds_list, endowment_list, multiplier_list, num_games, num_agents_list, exp_type):
    """Runs the same story experiment using Story index."""
    story_files = sorted(glob.glob("stories/*.txt"))
    if story_index >= len(story_files):
        print("Invalid story index. Exiting.")
        sys.exit(1)

    selected_story_file = story_files[story_index]
    story_name = os.path.splitext(os.path.basename(selected_story_file))[0]

    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        exp_name = f"{'bad_apple' if is_bad_apple else 'same_story'}_{story_name}_ag{na}_ro{nr}_end{e}_mult{m}"
        num_dummy_agents = 1 if is_bad_apple else 0

        print(f"\n\n######################\nRunning experiment: {exp_name}\n######################\n")
        intermediate_results, results_path = load_intermediate_results(exp_name, is_dict=True)
        
        scores_by_prompt = run_same_story_games(
            exp_name, selected_story_file, story_name, na, nr, e, m, num_games, num_dummy_agents,
            intermediate_results, results_path, exp_type
        )
        compute_and_print_statistics(scores_by_prompt, exp_name)


def run_different_story_experiment(num_rounds_list, endowment_list, multiplier_list, num_games, num_agents_list, exp_type):
    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        exp_name = f"different_story_ag{na}_ro{nr}_end{e}_mult{m}"
        print(f"\n\n######################\nRunning experiment: {exp_name}\n######################\n")

        story_prompts = load_all_story_prompts()
        system_prompt = get_system_prompt(na, e, m)
        intermediate_results, results_path = load_intermediate_results(exp_name, is_dict=False)

        scores_list, rewards_by_story = run_different_story_games(
            exp_name, na, nr, e, m, num_games, story_prompts, system_prompt,
            intermediate_results, results_path, exp_type
        )

        print(f"\n=== Rewards by Story for experiment {exp_name} ===")
        compute_and_print_statistics(rewards_by_story, exp_name)
        if scores_list:
            print(f"\nOverall Effective Collaboration Score: Mean = {statistics.mean(scores_list):.2f}, SD = {statistics.stdev(scores_list):.2f}")


def run_same_story_games(exp_name, story_file, story_name, na, nr, e, m, num_games, num_dummy_agents, intermediate_results, results_path, exp_type):
    with open(story_file, "r", encoding="utf-8") as f:
        story_content = f.read()
    if story_name not in ["maxreward", "noinstruct"]:
        story_content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + story_content
    # Build a dynamic system prompt using the current parameters.
    prompt_text = get_system_prompt(na, e, m).replace("STORY", story_content)

    if story_name not in intermediate_results:
        intermediate_results[story_name] = []

    # If intermediate results exist, re-populate the CSV file with final rows from previous games.
    scores = intermediate_results[story_name][:]
    with open(f"records_{exp_name}.txt", "w", encoding="utf-8") as records_file, \
         open(f"game_results_{exp_name}.csv", "w", newline="", encoding="utf-8") as results_file:

        writer = csv.writer(results_file)
        writer.writerow(CSV_HEADER)

        # Re-write completed game rows if reloading
        for idx, score in enumerate(scores, start=1):
            writer.writerow([idx, story_name, "final", "All", "", "", "", f"{score:.2f}"])
        print(f"\n=== Running Games with prompt: {story_name} for experiment {exp_name} ===")
        # Determine how many games have already been run for this prompt.
            
        for game_index in range(len(scores) + 1, num_games + 1):
            print(f"\n=== Game {game_index} ({story_name}) for experiment {exp_name} ===")
            score = run_single_game(game_index, story_name, prompt_text, na, nr, e, m, writer, records_file, num_dummy_agents, exp_type)
            scores.append(score)
            intermediate_results[story_name].append(score)
            save_intermediate_results(intermediate_results, results_path)

    return {story_name: scores}


def run_different_story_games(exp_name, na, nr, e, m, num_games, story_prompts, system_prompt, intermediate_results, results_path, exp_type):
    scores_list = []
    rewards_by_story = {story: [] for story in story_prompts}

    with open(f"records_{exp_name}.txt", "w", encoding="utf-8") as records_file, \
         open(f"game_results_{exp_name}.csv", "w", newline="", encoding="utf-8") as results_file:

        writer = csv.writer(results_file)
        writer.writerow(CSV_HEADER)

        for game_index in range(1, num_games + 1):
            if game_index <= len(intermediate_results):
                print(f"Skipping game {game_index} as it has already been run.")
                continue

            print(f"\n=== Game {game_index} for experiment {exp_name} ===")
            score, agent_results = run_single_game_random_story(
                game_index, system_prompt, na, nr, e, m, writer, records_file, story_prompts, exp_type
            )
            intermediate_results.append((game_index, score, agent_results))
            save_intermediate_results(intermediate_results, results_path)

            scores_list.append(score)
            for _, story_label, reward in agent_results:
                rewards_by_story[story_label].append(reward)

    return scores_list, rewards_by_story


def load_all_story_prompts():
    story_prompts = {}
    for story_file in sorted(glob.glob("stories/*.txt")):
        story_name = os.path.splitext(os.path.basename(story_file))[0]
        with open(story_file, "r", encoding="utf-8") as f:
            content = f.read()
        if story_name not in ["maxreward", "noinstruct"]:
            content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + content
        story_prompts[story_name] = content
    return story_prompts
