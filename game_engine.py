"""
Core game mechanics and pool management for multi-pool story agents experiments.
Handles both single-pool and multi-pool configurations.
"""

import csv
import random
from typing import List, Dict, Union
from pathlib import Path

from agent_utils import Agent, DummyAgent, get_valid_contribution


# CSV Header for unified data format
CSV_HEADER = [
    "Game", "PromptType", "Round", "AgentName", 
    "Contribution",      # Legacy column: total contribution amount
    "GlobalContrib",     # Global pool contribution
    "LocalContrib",      # Sum of local pool contributions  
    "TotalContrib",      # Total across all pools (Global + Local)
    "RoundPayoff", "CumulativePayoff", "CollaborationScore"
]


def build_pools(num_agents: int, pool_sizes: List[int] = None) -> Dict[str, List[int]]:
    """
    Build pool structure for experiments.
    
    Single-pool: one global pool containing all agents
    Multi-pool: global pool + local pools (subsets of agents)
    
    Args:
        num_agents: Total number of agents
        pool_sizes: List of local pool sizes (None for single-pool)
        
    Returns:
        Dictionary mapping pool_id to list of agent indices
    """
    pools = {"global": list(range(num_agents))}
    
    if pool_sizes:  # Multi-pool case
        if pool_sizes != [2]:
            raise ValueError("Only pool_sizes=[2] is currently supported")
        
        if num_agents != 4:
            raise ValueError("Multi-pool configuration currently only supports 4 agents")
        
        # Create non-overlapping pairs
        indices = list(range(num_agents))
        random.shuffle(indices)
        
        for i in range(0, num_agents, 2):
            if i + 1 < num_agents:
                pair = [indices[i], indices[i + 1]]
                pool_id = f"group_2_{i//2}"
                pools[pool_id] = pair
    
    return pools


def calculate_payoffs_single_pool(contributions: List[float], e: float, m: float, na: int) -> List[float]:
    """
    Calculate payoffs for single-pool game.
    
    Args:
        contributions: List of contributions from each agent
        e: Token endowment per round
        m: Multiplier for shared pool
        na: Number of agents
        
    Returns:
        List of payoffs for each agent
    """
    total = sum(contributions)
    shared_bonus = m * total / na
    return [e - c + shared_bonus for c in contributions]


def calculate_payoffs_multi_pool(
    contributions: Dict[str, Dict[int, int]], 
    pools: Dict[str, List[int]], 
    e: int
) -> List[float]:
    """
    Calculate payoffs for multi-pool game using the formula:
    π_i = Σ_{p: i ∈ M_p} (m * T_p / |M_p|) + (e - Σ_p t_{i,p})
    
    Args:
        contributions: Nested dict {pool_id: {agent_idx: contribution}}
        pools: Pool structure {pool_id: [agent_indices]}
        e: Token endowment per round
        
    Returns:
        List of payoffs for each agent
    """
    # Determine number of agents from pools
    all_agent_indices = set()
    for pool_members in pools.values():
        all_agent_indices.update(pool_members)
    n = len(all_agent_indices)
    
    payoffs = [0.0] * n
    m = 1.5  # Hardcoded multiplier for now
    
    # First term: Σ_{p: i ∈ M_p} (m * T_p / |M_p|)
    for pid, pool_contributions in contributions.items():
        members = list(pool_contributions.keys())
        pool_total = sum(pool_contributions.values())
        pool_size = len(members)
        
        if pool_size > 0:
            bonus_per_member = m * pool_total / pool_size
            for agent_idx in members:
                payoffs[agent_idx] += bonus_per_member
    
    # Second term: (e - Σ_p t_{i,p})
    for i in range(n):
        total_contributed = sum(contributions[pid].get(i, 0) for pid in contributions)
        kept_tokens = e - total_contributed
        payoffs[i] += kept_tokens
    
    return payoffs


def collect_contributions_single_pool(agents: List[Union[Agent, DummyAgent]], round_num: int, e: int) -> List[int]:
    """
    Collect contributions from all agents for single-pool game.
    
    Args:
        agents: List of agent instances
        round_num: Current round number
        e: Token endowment per round
        
    Returns:
        List of contributions
    """
    contributions = []
    for agent in agents:       
        contribution = get_valid_contribution(agent, round_num, e)
        
        # Enforce valid contribution range
        if contribution > e:
            print(f"{agent.name} attempted to contribute {contribution} tokens but only has {e}. "
                    f"Reducing contribution to {e}.")
            contribution = e
        contribution = max(0, contribution)
        contributions.append(contribution)
    
    print(f"Round {round_num} Contributions: {contributions}")
    return contributions


def collect_contributions_multi_pool(
    agents: List[Union[Agent, DummyAgent]], 
    pools: Dict[str, List[int]], 
    e: int,
    round_num: int
) -> Dict[str, Dict[int, int]]:
    """
    Collect contributions from agents for multi-pool games.
    
    Args:
        agents: List of agent instances
        pools: Pool structure
        e: Token endowment per round
        round_num: Current round number
        
    Returns:
        Nested dict {pool_id: {agent_idx: contribution}}
    """
    remaining = {i: e for i in range(len(agents))}
    contributions = {pid: {} for pid in pools}
    
    # Randomize pool order (agents don't know order)
    pool_order = random.sample(list(pools.keys()), len(pools))
    
    for pid in pool_order:
        # Randomize agent order within pool
        members = random.sample(pools[pid], len(pools[pid]))
        for i in members:
            others = [agents[j].name for j in pools[pid] if j != i]
            pool_context = f"Pool '{pid}' with {others}. You have {remaining[i]} tokens remaining. "
            
            contribution = get_valid_contribution(
                agent=agents[i], 
                round_num=round_num,
                e=remaining[i], 
                pool_context=pool_context
            )
            contribution = max(0, min(contribution, remaining[i]))
            remaining[i] -= contribution
            contributions[pid][i] = contribution
    
    print(f"Round {round_num} Multi-pool Contributions: {contributions}")
    return contributions


def provide_feedback_single_pool(
    agents: List[Union[Agent, DummyAgent]], 
    contributions: List[int], 
    payoffs: List[float],
    total_rewards: List[float],
    round_num: int
):
    """
    Provide round feedback to agents in single-pool game.
    
    Args:
        agents: List of agent instances
        contributions: List of contributions
        payoffs: List of round payoffs
        total_rewards: List of cumulative rewards
        round_num: Current round number
    """
    round_total = sum(contributions)
    
    for idx, agent in enumerate(agents):
        summary = (
            f"Round {round_num} Summary:\\n"
            f" - Your contribution: {contributions[idx]}\\n"
            f" - Total contributions: {round_total}\\n"
            f" - Your payoff this round: {payoffs[idx]:.2f}\\n"
            f" - Your cumulative reward: {total_rewards[idx]:.2f}"
        )
        agent.chat(summary)


def provide_feedback_multi_pool(
    agents: List[Union[Agent, DummyAgent]],
    contributions: Dict[str, Dict[int, int]],
    pools: Dict[str, List[int]],
    payoffs: List[float],
    total_rewards: List[float],
    round_num: int
):
    """
    Provide round feedback to agents in multi-pool game.
    
    Args:
        agents: List of agent instances
        contributions: Multi-pool contributions
        pools: Pool structure
        payoffs: List of round payoffs
        total_rewards: List of cumulative rewards
        round_num: Current round number
    """
    for i, agent in enumerate(agents):
        msg = [f"Round {round_num} Results:"]
        
        for pid, pool_contribs in contributions.items():
            if i in pool_contribs:
                others_in_pool = [j for j in pools[pid] if j != i]
                other_contributions = [
                    f"{agents[j].name}→{pool_contribs[j]}" 
                    for j in others_in_pool if j in pool_contribs
                ]
                msg.append(f"Pool '{pid}': You→{pool_contribs[i]}, " + ", ".join(other_contributions))
                
        msg.append(f"Round payoff: {payoffs[i]:.2f}, Cumulative: {total_rewards[i]:.2f}")
        agent.chat("\\n".join(msg))


def log_round_results(
    csv_writer,
    agents: List[Union[Agent, DummyAgent]],
    game_index: int,
    prompt_label: str,
    round_num: int,
    payoffs: List[float],
    total_rewards: List[float],
    exp_type: str,
    # Single-pool specific
    contributions_single: List[int] = None,
    # Multi-pool specific  
    pool_contributions: Dict[str, Dict[int, int]] = None
):
    """
    Log round results to CSV in unified format supporting both pool topologies.
    
    Args:
        csv_writer: CSV writer instance
        agents: List of agent instances
        game_index: Current game number
        prompt_label: Experiment/story label
        round_num: Current round number
        payoffs: Round payoffs
        total_rewards: Cumulative rewards
        exp_type: Experiment type
        contributions_single: Single-pool contributions (optional)
        pool_contributions: Multi-pool contributions (optional)
    """
    for idx, agent in enumerate(agents):
        # Determine story label based on experiment type
        if exp_type == "different_story" and hasattr(agent, 'story_label'):
            story_label = agent.story_label
        else:
            story_label = prompt_label
            
        if pool_contributions:  # Multi-pool case
            global_contrib = pool_contributions.get("global", {}).get(idx, 0)
            local_contrib = sum(
                pool_contributions[pid].get(idx, 0) 
                for pid in pool_contributions 
                if pid != "global"
            )
            total_contrib = global_contrib + local_contrib
            contribution = total_contrib  # For backward compatibility
            
        else:  # Single-pool case
            contribution = contributions_single[idx] if contributions_single else 0
            global_contrib = contribution  # In single-pool, all goes to "global"
            local_contrib = 0
            total_contrib = contribution
            
        csv_writer.writerow([
            game_index,
            story_label, 
            round_num,
            agent.name,
            contribution,      # Backward compatibility column
            global_contrib,    
            local_contrib,     
            total_contrib,     
            f"{payoffs[idx]:.2f}",
            f"{total_rewards[idx]:.2f}",
            ""  # CollaborationScore (empty for per-round rows)
        ])


def log_final_score(csv_writer, game_index: int, prompt_label: str, collaboration_score: float):
    """
    Log the final collaboration score row.
    
    Args:
        csv_writer: CSV writer instance
        game_index: Game number
        prompt_label: Experiment/story label
        collaboration_score: Final collaboration score
    """
    csv_writer.writerow([
        game_index,
        prompt_label,
        "final",
        "All",
        "",  # Contribution
        "",  # GlobalContrib  
        "",  # LocalContrib
        "",  # TotalContrib
        "",  # RoundPayoff
        "",  # CumulativePayoff
        f"{collaboration_score:.4f}"  # CollaborationScore
    ])


def setup_csv_file(csv_path: Path) -> tuple:
    """
    Setup CSV file for logging results.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (file_handle, csv_writer, file_existed)
    """
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    
    # Write header if new file
    if not csv_exists:
        writer.writerow(CSV_HEADER)
    
    return csv_file, writer, csv_exists