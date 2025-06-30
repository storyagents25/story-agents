"""
Experiment orchestration and management for multi-pool story agents.
Handles experiment execution, checkpoint management, and result organization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import random
import pickle
import itertools
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import get_config
from agent_utils import (
    Agent, DummyAgent, create_agent_factory, get_system_prompt,
    load_all_story_prompts, get_story_by_index, RateLimiter
)
from game_engine import (
    build_pools, collect_contributions_single_pool, collect_contributions_multi_pool,
    calculate_payoffs_single_pool, calculate_payoffs_multi_pool,
    provide_feedback_single_pool, provide_feedback_multi_pool,
    log_round_results, log_final_score, setup_csv_file
)


def get_experiment_directory(exp_type: str, story_name: str, na: int, nr: int, e: int, 
                           m: float, pool_sizes: Optional[List[int]] = None) -> Path:
    """Generate standardized experiment directory path."""
    topology = "multi_pool" if pool_sizes else "single_pool"
    exp_dir = Path("experiments") / topology / exp_type
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_experiment_filename(file_type: str, exp_type: str, story_name: Optional[str], 
                          na: int, nr: int, e: int, m: float, 
                          pool_sizes: Optional[List[int]] = None, game_idx: Optional[int] = None) -> str:
    """Generate standardized filenames."""
    # Determine pool topology string
    if pool_sizes:
        if pool_sizes == [2]:
            pool_str = "multi_pool_2"
        else:
            raise ValueError("Only pool_sizes=[2] is supported")
    else:
        pool_str = "single_pool"
    
    # Build parameter string
    params = f"ag{na}_ro{nr}_end{e}_mult{m}"
    
    # Build filename based on file type 
    if file_type == "game_results":
        if exp_type == "different_story" or story_name is None:
            return f"game_results_{pool_str}_{exp_type}_{params}.csv"
        else:
            return f"game_results_{pool_str}_{exp_type}_{story_name}_{params}.csv"
    
    elif file_type == "game_records":
        if exp_type == "different_story" or story_name is None:
            base = f"game_records_{pool_str}_{exp_type}_{params}"
        else:
            base = f"game_records_{pool_str}_{exp_type}_{story_name}_{params}"
        
        if game_idx:
            return f"{base}_game{game_idx:03d}.txt"
        else:
            return f"{base}.txt"
    
    elif file_type == "checkpoint":
        if exp_type == "different_story" or story_name is None:
            return f"checkpoint_{pool_str}_{exp_type}_{params}.pkl"
        else:
            return f"checkpoint_{pool_str}_{exp_type}_{story_name}_{params}.pkl"
    
    else:
        return f"{file_type}.txt"


def load_checkpoint(exp_type: str, story_name: Optional[str], na: int, nr: int, e: int, 
                   m: float, pool_sizes: Optional[List[int]] = None) -> Tuple[Any, Path]:
    """Load checkpoint for resuming interrupted experiments."""
    exp_dir = get_experiment_directory(exp_type, story_name or "unknown", na, nr, e, m, pool_sizes)
    checkpoint_filename = get_experiment_filename("checkpoint", exp_type, story_name, na, nr, e, m, pool_sizes)
    checkpoint_path = exp_dir / checkpoint_filename
    
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return results, checkpoint_path
    
    # No existing results
    print(f"No existing checkpoint found. Starting fresh at: {checkpoint_path}")
    return {} if exp_type != "different_story" else [], checkpoint_path


def save_checkpoint(results: Any, checkpoint_path: Path):
    """Save checkpoint for experiment resumability."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(results, f)


def execute_single_pool_game(agents: List[Agent], na: int, nr: int, e: int, m: float,
                           exp_dir: Path, game_index: int, prompt_label: str, exp_type: str,
                           num_dummy_agents: int, csv_filename: str) -> float:
    """Execute a complete single-pool game."""
    total_rewards = [0.0 for _ in range(na)]
    total_game_contributions = 0
    
    csv_path = exp_dir / csv_filename
    csv_file, writer, _ = setup_csv_file(csv_path)
    
    try:
        print(f"\n=== Starting Single-Pool Game {game_index} ===")
        
        for round_num in range(1, nr + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Collect contributions
            contributions = collect_contributions_single_pool(agents, round_num, e)
            
            # Calculate payoffs
            payoffs = calculate_payoffs_single_pool(contributions, e, m, na)
            
            # Update total rewards
            for idx in range(na):
                total_rewards[idx] += payoffs[idx]
            
            # Provide feedback
            provide_feedback_single_pool(agents, contributions, payoffs, total_rewards, round_num)
            
            # Log results
            log_round_results(
                writer, agents, game_index, prompt_label, round_num,
                payoffs, total_rewards, exp_type, contributions_single=contributions
            )
            
            total_game_contributions += sum(contributions)
            print(f"Round Payoffs: {payoffs}")

        # Calculate and log final score
        max_possible = (na - num_dummy_agents) * e * nr
        collaboration_score = total_game_contributions / max_possible
        print(f"\nCollaboration Score: {collaboration_score:.4f}")

        log_final_score(writer, game_index, prompt_label, collaboration_score)
        
        return collaboration_score
        
    finally:
        csv_file.close()


def execute_multi_pool_game(agents: List[Agent], pools: Dict[str, List[int]], na: int, nr: int, e: int,
                          exp_dir: Path, game_index: int, prompt_label: str, exp_type: str,
                          csv_filename: str) -> float:
    """Execute a complete multi-pool game."""
    total_rewards = [0.0 for _ in range(na)]
    total_game_contributions = 0
    
    csv_path = exp_dir / csv_filename
    csv_file, writer, _ = setup_csv_file(csv_path)
    
    try:
        print(f"\n=== Starting Multi-Pool Game {game_index} ===")
        
        for round_num in range(1, nr + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Collect contributions from all pools
            pool_contributions = collect_contributions_multi_pool(agents, pools, e, round_num)
            
            # Calculate payoffs
            payoffs = calculate_payoffs_multi_pool(pool_contributions, pools, e)
            
            # Update total rewards
            for idx in range(na):
                total_rewards[idx] += payoffs[idx]
            
            # Provide feedback
            provide_feedback_multi_pool(agents, pool_contributions, pools, payoffs, total_rewards, round_num)
            
            # Log results
            log_round_results(
                writer, agents, game_index, prompt_label, round_num,
                payoffs, total_rewards, exp_type, pool_contributions=pool_contributions
            )
            
            total_game_contributions += sum(sum(v.values()) for v in pool_contributions.values())
            print(f"Round Payoffs: {payoffs}")

        # Calculate and log final score
        max_possible = na * e * nr  # All agents contribute to all pools
        collaboration_score = total_game_contributions / max_possible
        print(f"\nCollaboration Score: {collaboration_score:.4f}")

        log_final_score(writer, game_index, prompt_label, collaboration_score)
        
        return collaboration_score
        
    finally:
        csv_file.close()


def run_single_game(game_index: int, prompt_label: str, system_prompt: str,
                   na: int, nr: int, e: int, m: float, exp_dir: Path, exp_type: str,
                   story_name: Optional[str], num_dummy_agents: int = 0,
                   pools: Optional[Dict[str, List[int]]] = None,
                   pool_sizes: Optional[List[int]] = None,
                   config: Optional[Dict[str, Any]] = None) -> float:
    """Run a single complete game."""
    
    # Generate descriptive records filename 
    records_filename = get_experiment_filename("game_records", exp_type, story_name, na, nr, e, m, pool_sizes, game_index)
    records_path = exp_dir / records_filename
    
    # Generate descriptive CSV filename 
    csv_filename = get_experiment_filename("game_results", exp_type, story_name, na, nr, e, m, pool_sizes)
    
    # Create rate limiter for API calls
    rate_limiter = RateLimiter(calls_per_minute=50)  # Conservative rate limit
    agent_factory = create_agent_factory(config, rate_limiter)
    
    with open(records_path, "w", encoding="utf-8") as records_file:
        # Create agents for this game
        agents = []
        for i in range(na):
            is_dummy = i < num_dummy_agents
            agent = agent_factory(f"Agent_{i+1}", system_prompt, records_file, is_dummy)
            if hasattr(agent, 'story_label'):
                agent.story_label = prompt_label
            agents.append(agent)
        
        # Execute game based on pool configuration
        if pools:  # Multi-pool
            return execute_multi_pool_game(
                agents, pools, na, nr, e, exp_dir, game_index, prompt_label, exp_type, csv_filename
            )
        else:  # Single-pool
            return execute_single_pool_game(
                agents, na, nr, e, m, exp_dir, game_index, prompt_label, exp_type, num_dummy_agents, csv_filename
            )


def run_same_story_experiment(is_bad_apple: bool, story_index: int, num_rounds_list: List[int],
                            endowment_list: List[int], multiplier_list: List[float],
                            num_games: int, num_agents_list: List[int], exp_type: str,
                            pool_sizes: Optional[List[int]] = None, config: Optional[Dict[str, Any]] = None):
    """Run same story experiment where all agents receive identical story prompts."""
    
    if config is None:
        config = get_config()
    
    # Load story
    stories_dir = config["experiment"]["stories_dir"]
    story_name, story_content = get_story_by_index(story_index, stories_dir)
    
    num_dummy_agents = 1 if is_bad_apple else 0
    
    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        print(f"\n######################")
        print(f"Running {exp_type} experiment: {story_name}")
        print(f"Agents: {na}, Rounds: {nr}, Endowment: {e}, Multiplier: {m}")
        if pool_sizes:
            print(f"Pool sizes: {pool_sizes}")
        print(f"######################\n")

        # Get experiment directory
        exp_dir = get_experiment_directory(exp_type, story_name, na, nr, e, m, pool_sizes)
        
        # Create system prompt
        is_networked = pool_sizes is not None
        prompt_text = get_system_prompt(na, e, m, is_networked).replace("STORY", story_content)

        # Load existing results
        results, checkpoint_path = load_checkpoint(exp_type, story_name, na, nr, e, m, pool_sizes)
        
        if story_name not in results:
            results[story_name] = []

        scores = results[story_name][:]
        print(f"Experiment directory: {exp_dir}")
        print(f"Completed games: {len(scores)}")
                
        for game_index in range(len(scores) + 1, num_games + 1):
            print(f"\n=== Game {game_index}/{num_games} ({story_name}) ===")
            
            # Build pools if networked
            pools = build_pools(na, pool_sizes) if is_networked else None
            
            try:
                score = run_single_game(
                    game_index, story_name, prompt_text, na, nr, e, m, exp_dir, 
                    exp_type, story_name, num_dummy_agents, pools, pool_sizes, config
                )
                scores.append(score)
                results[story_name] = scores
                save_checkpoint(results, checkpoint_path)
                print(f"Game {game_index} completed. Score: {score:.4f}")
                
            except Exception as exc:
                print(f"Game {game_index} failed: {exc}")
                continue

        # Print statistics
        if scores:
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0
            print(f"\n=== Results for {story_name} ===")
            print(f"Mean Collaboration Score: {mean_score:.4f}")
            print(f"Standard Deviation: {std_score:.4f}")
            print(f"Games completed: {len(scores)}")


def run_different_story_experiment(num_rounds_list: List[int], endowment_list: List[int],
                                 multiplier_list: List[float], num_games: int, num_agents_list: List[int],
                                 exp_type: str, pool_sizes: Optional[List[int]] = None,
                                 config: Optional[Dict[str, Any]] = None):
    """Run different story experiment where each agent receives a random story."""
    
    if config is None:
        config = get_config()
    
    stories_dir = config["experiment"]["stories_dir"]
    story_prompts = load_all_story_prompts(stories_dir)
    
    for na, nr, e, m in itertools.product(num_agents_list, num_rounds_list, endowment_list, multiplier_list):
        print(f"\n######################")
        print(f"Running different_story experiment")
        print(f"Agents: {na}, Rounds: {nr}, Endowment: {e}, Multiplier: {m}")
        if pool_sizes:
            print(f"Pool sizes: {pool_sizes}")
        print(f"######################\n")

        # Get experiment directory
        exp_dir = get_experiment_directory("different_story", None, na, nr, e, m, pool_sizes)
        
        # Create system prompt template
        is_networked = pool_sizes is not None
        system_prompt_template = get_system_prompt(na, e, m, is_networked)

        # Load existing results
        results, checkpoint_path = load_checkpoint("different_story", None, na, nr, e, m, pool_sizes)
        
        scores_list = []
        rewards_by_story = {story: [] for story in story_prompts}
        
        print(f"Experiment directory: {exp_dir}")
        print(f"Completed games: {len(results)}")

        for game_index in range(len(results) + 1, num_games + 1):
            print(f"\n=== Game {game_index}/{num_games} (different_story) ===")
            
            # Build pools if networked
            pools = build_pools(na, pool_sizes) if pool_sizes else None
            
            try:
                # Create agents with random stories (handled in run_single_game_random_story)
                score, agent_results = run_single_game_random_story(
                    game_index, system_prompt_template, na, nr, e, m, exp_dir,
                    exp_type, story_prompts, pools, pool_sizes, config
                )
                
                # Store results
                results.append((game_index, score, agent_results))
                save_checkpoint(results, checkpoint_path)
                
                scores_list.append(score)
                for _, story_label, reward in agent_results:
                    rewards_by_story[story_label].append(reward)
                    
                print(f"Game {game_index} completed. Score: {score:.4f}")
                
            except Exception as exc:
                print(f"Game {game_index} failed: {exc}")
                continue

        # Print statistics
        if scores_list:
            print(f"\n=== Overall Results ===")
            print(f"Mean Collaboration Score: {statistics.mean(scores_list):.4f}")
            print(f"Standard Deviation: {statistics.stdev(scores_list):.4f}")
            
            print(f"\n=== Rewards by Story ===")
            for story, rewards in rewards_by_story.items():
                if rewards:
                    mean_reward = statistics.mean(rewards)
                    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0
                    print(f"{story}: Mean = {mean_reward:.2f}, SD = {std_reward:.2f}")


def run_single_game_random_story(game_index: int, system_prompt_template: str, na: int, nr: int, e: int, m: float,
                                exp_dir: Path, exp_type: str, story_prompts: Dict[str, str],
                                pools: Optional[Dict[str, List[int]]] = None,
                                pool_sizes: Optional[List[int]] = None,
                                config: Optional[Dict[str, Any]] = None) -> Tuple[float, List[Tuple[str, str, float]]]:
    """Run a single game where each agent gets a random story."""
    
    # Generate filenames
    records_filename = get_experiment_filename("game_records", exp_type, None, na, nr, e, m, pool_sizes, game_index)
    records_path = exp_dir / records_filename
    csv_filename = get_experiment_filename("game_results", exp_type, None, na, nr, e, m, pool_sizes)
    
    # Create rate limiter and agent factory
    rate_limiter = RateLimiter(calls_per_minute=50)
    agent_factory = create_agent_factory(config, rate_limiter)
    
    with open(records_path, "w", encoding="utf-8") as records_file:
        agents = []
        agent_stories = []

        # Create agents with random story prompts
        for i in range(na):
            chosen_label, chosen_story = random.choice(list(story_prompts.items()))
            prompt_text = system_prompt_template.replace("STORY", chosen_story)
            agent = agent_factory(f"Agent_{i+1}", prompt_text, records_file, False)
            agent.story_label = chosen_label
            agents.append(agent)
            agent_stories.append(chosen_label)

        print(f"Agent stories: {agent_stories}")

        # Execute game
        if pools:  # Multi-pool
            collaboration_score = execute_multi_pool_game(
                agents, pools, na, nr, e, exp_dir, game_index, "All", exp_type, csv_filename
            )
        else:  # Single-pool
            collaboration_score = execute_single_pool_game(
                agents, na, nr, e, m, exp_dir, game_index, "All", exp_type, 0, csv_filename
            )

        # Prepare results: (agent_name, story_label, cumulative_reward)
        agent_results = []
        for i, agent in enumerate(agents):
            # Get final cumulative reward (would need to track this properly)
            # For now, using a placeholder - should be implemented properly
            final_reward = 0.0  # This should be tracked during game execution
            agent_results.append((agent.name, agent.story_label, final_reward))
        
    return collaboration_score, agent_results