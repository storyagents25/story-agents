"""
Agent classes and utility functions for multi-pool story agents experiments.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import re
import glob
import time
from typing import List, Dict, Any, TextIO, Optional

from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

from llm_model import ModelFactory, RateLimiter

def log_records(message: str, records_file: TextIO) -> None:
    """
    Helper function to log a message to records file.
    
    Args:
        message: Message to log
        records_file: File handle for logging
    """
    records_file.write(message + "\n")
    records_file.flush()


class Agent:
    """
    LLM-powered agent for Public Goods Game participation.
    
    Features:
    - Configurable LLM backend (OpenAI or Llama)
    - Maintains conversation history across rounds
    - Logs all interactions for analysis
    - Handles contribution decisions with validation
    """
    
    def __init__(self, name: str, system_message: str, records_file: TextIO, 
                 config: Optional[Dict[str, Any]] = None, rate_limiter: Optional[RateLimiter] = None):
        self.name = name
        self.history: List[BaseMessage] = [SystemMessage(content=system_message)]
        self.records_file = records_file
        self.rate_limiter = rate_limiter
        
        # Additional attributes for experiment tracking
        self.story_label: Optional[str] = None
        
        # Create LLM instance
        self.llm = ModelFactory.create_model(config)
        
        # Log agent creation
        log_records(f"CREATING AGENT: {self.name}", records_file)
        log_records(f"System Prompt for {self.name}: {system_message}", records_file)

    def chat(self, message: str) -> str:
        """
        Send message to agent and get response.
        
        Args:
            message: Human message to send to agent
            
        Returns:
            Agent's response content
        """
        log_records(f"{self.name} receives HUMAN message: {message}", self.records_file)
        self.history.append(HumanMessage(content=message))
        
        # Apply rate limiting if configured
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        
        try:
            response = self.llm.invoke(self.history)
        except Exception as err:
            error_msg = f"[ERROR]: {err}"
            log_records(f"{self.name} ERROR: {err}", self.records_file)
            return error_msg

        # Store response in conversation history
        self.history.append(response)
        log_records(f"{self.name} responds ASSISTANT: {response.content}", self.records_file)
        
        # Brief pause to help avoid overwhelming APIs
        time.sleep(0.1)
        return response.content


class DummyAgent:
    """
    Dummy agent for robustness testing that always contributes 0.
    Simulates a persistent free-rider in the system.
    """
    
    def __init__(self, name: str, system_message: str, records_file: TextIO):
        self.name = name
        self.history: List[BaseMessage] = [SystemMessage(content=system_message)]
        self.records_file = records_file
        self.story_label: Optional[str] = None
        
        log_records(f"CREATING DUMMY AGENT: {self.name}", records_file)
        log_records(f"(Dummy) System Prompt for {self.name}: {system_message}", records_file)

    def chat(self, message: str) -> str:
        """
        Dummy agent always responds with 0 contribution.
        
        Args:
            message: Human message (logged but ignored)
            
        Returns:
            Always returns "<TOKEN>0</TOKEN>"
        """
        log_records(f"{self.name} (dummy) receives HUMAN message: {message}", self.records_file)
        response_content = "<TOKEN>0</TOKEN>"
        log_records(f"{self.name} (dummy) responds ASSISTANT: {response_content}", self.records_file)
        time.sleep(0.01)
        return response_content


def get_system_prompt(na: int, e: int, m: float, is_networked: bool = False) -> str:
    """
    Generate system prompt for agents based on game configuration.
    
    Args:
        na: Number of agents
        e: Token endowment per round
        m: Multiplier for shared pool returns
        is_networked: True for multi-pool, False for single-pool
        
    Returns:
        System prompt template with STORY placeholder
    """
    if is_networked:
        base = (
            f"You are participating in a repeated networked public goods game with {na} agents. "
            f"In each round, you receive {e} tokens. "
            "You will be prompted sequentially for contributions to each shared pool - some with all agents (global) and some with smaller subsets. "
            f"After everyone contributes, each pool's total is multiplied by {m} and divided equally among its members. "
            "Respond only with <TOKEN>k</TOKEN> where k is your integer contribution to the current pool."
        )
    else:
        base = (
            f"You are participating in a repeated public goods game with {na - 1} other agents. "
            f"In each round, you receive {e} tokens. You must decide how many tokens (an integer between 0 and {e}) "
            f"to contribute to a public pool. After everyone contributes, the total contribution is multiplied by {m} "
            f"and divided equally among all {na} agents. This means your payoff for a round is calculated as: \\n\\n"
            f"    {e} - (your contribution) + ({m} * total contributions / {na})\\n\\n"
            "Please respond **only** with the number of tokens you wish to contribute for this round, enclosed within `<TOKEN>` and `</TOKEN>` tags. "
            "For example: `<TOKEN>5</TOKEN>`. Do not provide any additional text, explanations, or summaries. "
        )
    return base + "STORY"


def extract_contribution(response_str: str) -> Optional[int]:
    """
    Extract valid contribution from agent's response.
    
    Args:
        response_str: Agent's response string
        
    Returns:
        Extracted contribution as integer, or None if invalid
    """
    match = re.search(r"<TOKEN>(\d+)</TOKEN>", response_str)
    if match:
        return int(match.group(1))
    return None


def get_valid_contribution(agent: Agent, round_num: int, e: int, max_retries: int = 5, 
                          pool_context: str = "") -> int:
    """
    Get valid contribution from agent with retries.
    
    Args:
        agent: Agent instance
        round_num: Current round number
        e: Maximum tokens available
        max_retries: Maximum retry attempts
        pool_context: Additional context about current pool
        
    Returns:
        Valid contribution (0 to e)
    """
    retries = 0
    while retries < max_retries:
        prompt = f"Round {round_num}: {pool_context}What is your contribution (0-{e})?"
        
        if retries > 0:
            prompt += " Your previous response was invalid. **Only provide a number inside `<TOKEN>...</TOKEN>`** with no extra text. Example: `<TOKEN>5</TOKEN>`."
        
        response_str = agent.chat(prompt).strip()
        print(f"{agent.name} response (attempt {retries + 1}): {response_str}")

        contribution = extract_contribution(response_str)
        
        if contribution is not None and 0 <= contribution <= e:
            return contribution
        
        print(f"Warning: {agent.name} provided an invalid response. Retrying... ({retries + 1}/{max_retries})")
        retries += 1

    print(f"Error: {agent.name} failed to provide a valid response after {max_retries} attempts. Defaulting to 0.")
    return 0


def load_all_story_prompts(stories_dir: str = "stories") -> Dict[str, str]:
    """
    Load all story prompts from files.
    
    Args:
        stories_dir: Directory containing story .txt files
        
    Returns:
        Dictionary mapping story names to content
    """
    story_prompts = {}
    story_files = sorted(glob.glob(f"{stories_dir}/*.txt"))
    
    if not story_files:
        raise FileNotFoundError(f"No .txt files found in '{stories_dir}' directory")
    
    for story_file in story_files:
        story_name = Path(story_file).stem
        try:
            with open(story_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            # Add prefix for most stories (except special ones)
            if story_name not in ["maxreward", "noinstruct"]:
                content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + content
            
            story_prompts[story_name] = content
            
        except Exception as e:
            print(f"Warning: Could not load story file '{story_file}': {e}")
    
    return story_prompts


def get_story_by_index(story_index: int, stories_dir: str = "stories") -> tuple[str, str]:
    """
    Get story name and content by index.
    
    Args:
        story_index: Index of story (0-based)
        stories_dir: Directory containing story files
        
    Returns:
        Tuple of (story_name, story_content)
    """
    story_files = sorted(glob.glob(f"{stories_dir}/*.txt"))
    
    if not story_files:
        raise FileNotFoundError(f"No .txt files found in '{stories_dir}' directory")
    
    if story_index >= len(story_files):
        raise IndexError(f"Story index {story_index} out of range. Found {len(story_files)} stories.")
    
    selected_story_file = story_files[story_index]
    story_name = Path(selected_story_file).stem
    
    with open(selected_story_file, "r", encoding="utf-8") as f:
        story_content = f.read().strip()
    
    # Add prefix for most stories
    if story_name not in ["maxreward", "noinstruct"]:
        story_content = "Your behavior is influenced by the following bedtime story your mother read to you every night: " + story_content
    
    return story_name, story_content


def create_agent_factory(config: Optional[Dict[str, Any]] = None, 
                        rate_limiter: Optional[RateLimiter] = None):
    """
    Create agent factory function with shared configuration.
    
    Args:
        config: Configuration dictionary
        rate_limiter: Shared rate limiter instance
        
    Returns:
        Function to create agents with consistent configuration
    """
    def create_agent(name: str, system_message: str, records_file: TextIO, 
                    is_dummy: bool = False) -> Agent:
        if is_dummy:
            return DummyAgent(name, system_message, records_file)
        else:
            return Agent(name, system_message, records_file, config, rate_limiter)
    
    return create_agent


# Statistics and analysis utilities
def compute_collaboration_score(total_contributions: float, max_possible: float) -> float:
    """
    Compute collaboration score as fraction of maximum possible contributions.
    
    Args:
        total_contributions: Sum of all contributions across all rounds
        max_possible: Maximum possible contributions
        
    Returns:
        Collaboration score between 0 and 1
    """
    if max_possible == 0:
        return 0.0
    return min(1.0, total_contributions / max_possible)


def compute_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    import statistics
    
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values)
    }