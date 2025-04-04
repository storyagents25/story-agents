import os
import pickle
import random
import statistics
import csv
import re
from llm_model import Agent, DummyAgent, calculate_payoffs

def load_intermediate_results(exp_name, is_dict=True):
    """Loads intermediate results from a pickle file if it exists."""
    intermediate_results_filename = f"results_{exp_name}.pkl"
    if os.path.exists(intermediate_results_filename):
        with open(intermediate_results_filename, 'rb') as f:
            intermediate_results = pickle.load(f)
        print(f"Loaded intermediate results from {intermediate_results_filename}.")
    else:
        intermediate_results = {} if is_dict else []
    return intermediate_results, intermediate_results_filename

def save_intermediate_results(intermediate_results, intermediate_results_filename):
    """Saves intermediate results to a pickle file."""
    with open(intermediate_results_filename, 'wb') as f:
        pickle.dump(intermediate_results, f)

def get_system_prompt(na, e, m):
    """
    Generates the system prompt.
    
    Args:
        na (int): Number of agents.
        e (int): Endowment per round (tokens given to each agent).
        m (float): Contribution multiplier affecting payoffs.
    
    Returns:
        str: The formatted system prompt.
    """
    return (
        f"You are participating in a repeated public goods game with {na - 1} other agents. "
        f"In each round, you receive {e} tokens. You must decide how many tokens (an integer between 0 and {e}) "
        f"to contribute to a public pool. After everyone contributes, the total contribution is multiplied by {m} "
        f"and divided equally among all {na} agents. This means your payoff for a round is calculated as: \n\n"
        f"    {e} - (your contribution) + ({m} * total contributions / {na})\n\n"
        "Please respond **only** with the number of tokens you wish to contribute for this round, enclosed within `<TOKEN>` and `</TOKEN>` tags. "
        "For example: `<TOKEN>5</TOKEN>`. Do not provide any additional text, explanations, or summaries. "
        "STORY"
    )

def compute_and_print_statistics(scores_by_prompt, exp_name):
    """Computes and prints mean and standard deviation of scores for each prompt."""
    print(f"\n=== Comparison of Effective Collaboration Scores Over Games for experiment {exp_name} ===")
    for prompt_label, score_list in scores_by_prompt.items():
        mean_val = statistics.mean(score_list)
        stdev_val = statistics.stdev(score_list) if len(score_list) > 1 else 0
        print(f"{prompt_label}: Mean = {mean_val:.2f}, SD = {stdev_val:.2f}")

def extract_contribution(response_str):
    """
    Extracts a valid contribution from an agent's response.
    - First, looks for a contribution enclosed in <TOKEN>...</TOKEN>.
    - If found, extracts the first standalone integer from the match.
    """
    
    # Look for <TOKEN>...</TOKEN> first
    match = re.search(r"<TOKEN>(\d+)</TOKEN>", response_str)
    if match:
        return int(match.group(1))
    return None


def get_valid_contribution(agent, round_num, e, max_retries=5):
    """
    Repeatedly queries the agent for a valid contribution up to max_retries.
    If the response is invalid for more than max_retries, defaults to 0.
    """
    retries = 0
    while retries < max_retries:
        prompt = f"Round {round_num}: What is your contribution (0-{e})?"
        
        if retries > 0:
            prompt += " Your previous response was invalid. **Only provide a number inside `<TOKEN>...</TOKEN>`** with no extra text. Example: `<TOKEN>5</TOKEN>`."
        
        response_str = agent.chat(prompt).strip()
        print(f"{agent.name} (Story: {agent.story_label}) response (attempt {retries + 1}): {response_str}")

        contribution = extract_contribution(response_str)
        
        if contribution is not None:
            return contribution
        
        print(f"Warning: {agent.name} provided an invalid response. Retrying... ({retries + 1}/{max_retries})")
        retries += 1

    # If all retries fail, default to 0 and log the failure
    print(f"Error: {agent.name} failed to provide a valid response after {max_retries} attempts. Defaulting to 0.")
    return 0

def collect_contributions(agents, round_num, e):
    """Collects valid contributions from all agents."""
    contributions = []
    for agent in agents:       
        contribution = get_valid_contribution(agent, round_num, e)
        
        # Enforce valid contribution range
        available_tokens = e # Each agent gets `e` tokens every round
        if contribution > available_tokens:
            print(f"{agent.name} attempted to contribute {contribution} tokens but only has {available_tokens}. "
                    f"Reducing contribution to {available_tokens}.")
            contribution = available_tokens
        contribution = max(0, contribution)
        contributions.append(contribution)
    print(f"Round Contributions: {contributions}")
    return contributions

def calculate_rewards(contributions, agents, e, m, na, total_rewards, round_num):
    """Calculates payoffs and updates agent rewards."""
    round_total = sum(contributions)    
    # Calculate payoffs for the round.
    payoffs = calculate_payoffs(contributions, e, m, na)
    print(f"Round Payoffs: {payoffs}")

    for idx, agent in enumerate(agents):
        total_rewards[idx] += payoffs[idx]
        summary = (
            f"Round {round_num} Summary:\n"
            f" - Your contribution: {contributions[idx]}\n"
            f" - Total contributions: {round_total}\n"
            f" - Your payoff this round: {payoffs[idx]:.2f}\n"
            f" - Your cumulative reward: {total_rewards[idx]:.2f}"
        )
        agent.chat(summary)

    return payoffs, round_total, total_rewards


def log_round_results(csv_writer, agents, contributions, payoffs, total_rewards, game_index, prompt_label, round_num, exp_type):
    """Logs round results to CSV."""
    for idx, agent in enumerate(agents):
        story_or_prompt_label = agent.story_label if exp_type == "different_story" else prompt_label
        # Write per-round info to the CSV file.
        csv_writer.writerow([
            game_index,
            story_or_prompt_label,
            round_num,
            agent.name,
            contributions[idx],
            f"{payoffs[idx]:.2f}",
            f"{total_rewards[idx]:.2f}",
            "" # CollaborationScore left empty for per-round details.
        ])
def execute_game_rounds(agents, na, nr, e, m, csv_writer, records_file, game_index, prompt_label, exp_type, num_dummy_agents):
    """
    Executes a full game session consisting of multiple rounds where agents contribute to a shared pool.
    Args:
        agents (list): List of agent objects.
        na (int): Number of agents.
        nr (int): Number of rounds.
        e (int): Endowment per round.
        m (float): Multiplier for contributions.
        csv_writer (csv.writer): CSV writer object.
        records_file (file object): File for logging agent responses.
        game_index (int): Game identifier.
        prompt_label (str): Label for the prompt or story used.
        exp_type (str): "same_story" or "different_story" (to determine CSV formatting).
        num_dummy_agents: Number of dummy agents in the game
    Returns:
        effective_score (float): The overall collaboration score.
        total_rewards (list): Cumulative rewards for each agent.
    """
    total_rewards = [0 for _ in range(na)]
    total_game_contributions = 0

    print("\n=== Starting a New Game ===")
    for round_num in range(1, nr + 1):
        print(f"\n--- Round {round_num} ---")
        contributions = collect_contributions(agents, round_num, e)
        payoffs, round_total, total_rewards = calculate_rewards(contributions, agents, e, m, na, total_rewards, round_num)
        log_round_results(csv_writer, agents, contributions, payoffs, total_rewards, game_index, prompt_label, round_num, exp_type)
        total_game_contributions += round_total

    max_possible = (na - num_dummy_agents) * e * nr
    effective_score = total_game_contributions / max_possible
    print(f"\nEffective Collaboration Score: {effective_score:.2f}")

    # Write the final row with the collaboration score.
    csv_writer.writerow([
        game_index,
        prompt_label,
        "final",
        "All",
        "",
        "",
        "",
        f"{effective_score:.2f}"
    ])

    return effective_score, total_rewards


def run_single_game(game_index: int, prompt_label: str, system_prompt_used: str,
                    na: int, nr: int, e: int, m: float, csv_writer, records_file, num_dummy_agents, exp_type) -> float:
    """
    Run a single game (with nr rounds) using the given system prompt and experiment parameters.
    Returns the effective collaboration score for the game.
    """
    # Create new agents for this game.    
    agents = []
    for i in range(na):
        if i < num_dummy_agents:
            # Create a dummy agent
            agent = DummyAgent(f"Agent_{i+1}", system_prompt_used, records_file)
        else:
            # Create a standard LLM-based agent
            agent = Agent(f"Agent_{i+1}", system_prompt_used, records_file)
        agents.append(agent)
        
    for agent in agents:
        agent.story_label = prompt_label
    
    # Executes all rounds of the game, tracking contributions, payoffs, and collaboration scores.
    effective_score, _ = execute_game_rounds(
        agents, na, nr, e, m, csv_writer, records_file, game_index, prompt_label, exp_type, num_dummy_agents
    )
    return effective_score

def run_single_game_random_story(game_index: int, system_prompt_story: str, na: int, nr: int, e: int, m: float,
                                csv_writer, records_file, story_prompts: dict, exp_type) -> (float, list):
    """
    Run a single game where each agent gets a random story.
    Returns:
    - effective_score: overall collaboration score (total game contributions divided by maximum possible)
    - agent_results: list of tuples (agent_name, story_label, cumulative_reward) for each agent.
    """
    agents = []

    # Create agents with random story prompts.
    for i in range(na):
        chosen_label, chosen_story = random.choice(list(story_prompts.items()))
        # Insert the chosen story into the base system prompt.
        prompt_text = system_prompt_story.replace("STORY", chosen_story)
        agent = Agent(f"Agent_{i+1}", prompt_text, records_file)
        agent.story_label = chosen_label
        agents.append(agent)

    # Executes all rounds of the game, tracking contributions, payoffs, and collaboration scores.
    effective_score, total_rewards = execute_game_rounds(
        agents, na, nr, e, m, csv_writer, records_file, game_index, "All", exp_type, 0
    )

    # Prepare results: (agent_name, story_label, cumulative_reward)
    agent_results = [(agents[i].name, agents[i].story_label, total_rewards[i]) for i in range(na)]
    return effective_score, agent_results