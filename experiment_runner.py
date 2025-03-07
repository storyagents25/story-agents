# -----------------------------
# Global Settings and Imports
# -----------------------------

import os
import sys
import glob
import itertools
import statistics
from agent_utils import (
    run_single_game,
    run_single_game_random_story,
    load_intermediate_results,
    prepare_experiment,
    get_system_prompt,
    save_intermediate_results,
    compute_and_print_statistics,
)

def run_same_story_experiment(is_bad_apple, story_index, num_rounds_list, endowment_list, multiplier_list, num_games, num_agents_list, exp_type):
    """Runs the same story experiment using SLURM array index."""
    story_index = story_index
    story_files = sorted(glob.glob("stories/*.txt"))

    if story_index >= len(story_files):
        print("Invalid story index. Exiting.")
        sys.exit(1)
    selected_story_file = story_files[story_index]
    story_name = os.path.splitext(os.path.basename(selected_story_file))[0]
    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        exp_name = f"{'bad_apple' if is_bad_apple else 'same_story'}_{story_name}_ag{na}_ro{nr}_end{e}_mult{m}_temp0.8"
        num_dummy_agents = 1 if is_bad_apple else 0
        
        print(f"\n\n######################\nRunning experiment: {exp_name}\n######################\n")
    
        intermediate_results, intermediate_results_filename = load_intermediate_results(exp_name, is_dict=True)

        records_file, game_results_file, csv_writer = prepare_experiment(exp_name, ["Game", "PromptType", "Round", "AgentName", "Contribution", "RoundPayoff", "CumulativePayoff", "CollaborationScore"])
        
        # If intermediate results exist, re-populate the CSV file with final rows from previous games.
        if intermediate_results:
            for prompt_label, score_list in intermediate_results.items():
                for game_index, score in enumerate(score_list, start=1):
                    csv_writer.writerow([
                        game_index,
                        prompt_label,
                        "final",
                        "All",
                        "",
                        "",
                        "",
                        f"{score:.2f}"
                    ])
        
        # Build a dynamic system prompt using the current parameters.
        system_prompt_story = get_system_prompt(na, e, m)

        prompt_categories = dict()            
        with open(selected_story_file, "r", encoding="utf-8") as f:
            story_content = f.read()
            
        if story_name not in ["maxreward", "noinstruct"]:
            story_content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + story_content
        
        prompt_categories[story_name] = system_prompt_story.replace("STORY", story_content)        
        # Ensure all prompt labels are present in intermediate_results.
        for prompt_label in prompt_categories:
            if prompt_label not in intermediate_results:
                intermediate_results[prompt_label] = []
        
        # Run num_games for every story prompt.
        scores_by_prompt = {}
        for prompt_label, prompt_text in prompt_categories.items():
            scores_by_prompt[prompt_label] = intermediate_results[prompt_label][:]
            print(f"\n=== Running Games with prompt: {prompt_label} for experiment {exp_name} ===")
            # Determine how many games have already been run for this prompt.
            games_already_run = len(intermediate_results[prompt_label])
            for game in range(games_already_run + 1, num_games + 1):
                print(f"\n=== Game {game} ({prompt_label}) for experiment {exp_name} ===")
                score = run_single_game(game, prompt_label, prompt_text, na, nr, e, m, csv_writer, records_file, num_dummy_agents, exp_type)
                scores_by_prompt[prompt_label].append(score)
                intermediate_results[prompt_label].append(score)
                # Save intermediate results after each game.
                save_intermediate_results(intermediate_results, intermediate_results_filename)

        # Compute and print statistics for each prompt category.
        compute_and_print_statistics(scores_by_prompt, exp_name)       
        game_results_file.close()
        records_file.close()

def run_different_story_experiment(num_rounds_list, endowment_list, multiplier_list, num_games, num_agents_list, exp_type):
    """Runs the different story experiment where each agent has a unique story."""
    exp_name = "different_story"
    
    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        exp_name = f"different_story_ag{na}_ro{nr}_end{e}_mult{m}_temp0.8"
        print(f"\n\n######################\nRunning experiment: {exp_name}\n######################\n")

        # Load all story files from the "stories" folder.
        story_prompts = {}
        for story_file in sorted(glob.glob("stories/*.txt")):
            story_name = os.path.splitext(os.path.basename(story_file))[0]
            with open(story_file, "r", encoding="utf-8") as f:
                story_content = f.read()
            # If the story is not a special case, prepend an influence message.
            if story_name not in ["maxreward", "noinstruct"]:
                story_content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + story_content
            story_prompts[story_name] = story_content

        # Build a base system prompt that includes a placeholder "STORY".
        system_prompt_story = get_system_prompt(na, e, m)

        intermediate_results, intermediate_results_filename = load_intermediate_results(exp_name, is_dict=False)

        records_file, game_results_file, csv_writer = prepare_experiment(exp_name, ["Game", "PromptType", "Round", "AgentName", "Contribution", "RoundPayoff", "CumulativePayoff", "CollaborationScore"])


        # Run num_games games.
        for game in range(1, num_games + 1):
            # Skip games that have already been run.
            if game <= len(intermediate_results):
                print(f"Skipping game {game} as it has already been run.")
                continue
            print(f"\n=== Game {game} for experiment {exp_name} ===")
            effective_score, agent_results = run_single_game_random_story(
                game, system_prompt_story, na, nr, e, m, csv_writer, records_file, story_prompts, exp_type
            )
            intermediate_results.append((game, effective_score, agent_results))
            # Save intermediate results after each game.
            save_intermediate_results(intermediate_results, intermediate_results_filename)


        # Compute and print statistics for each story.
        rewards_by_story = {story_label: [] for story_label in story_prompts.keys()}
        scores_list = []  # to collect overall effective collaboration scores
        for game_tuple in intermediate_results:
            _, effective_score, agent_results = game_tuple
            scores_list.append(effective_score)
            for _, story_label, reward in agent_results:
                rewards_by_story[story_label].append(reward)
        
        print(f"\n=== Rewards by Story for experiment {exp_name} ===")
        compute_and_print_statistics(rewards_by_story, exp_name)
        overall_mean = statistics.mean(scores_list) if scores_list else 0
        overall_stdev = statistics.stdev(scores_list) if len(scores_list) > 1 else 0
        print(f"\nOverall Effective Collaboration Score: Mean = {overall_mean:.2f}, SD = {overall_stdev:.2f}")

        # Close files for the experiment.
        game_results_file.close()
        records_file.close()
