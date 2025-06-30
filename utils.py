"""
Additional utility functions for multi-pool story agents experiments.
Contains helper functions for file management, data processing, and analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("multi_pool_experiments")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_experiment_setup(stories_dir: str = "stories", 
                            required_stories: Optional[List[str]] = None) -> bool:
    """
    Validate that the experiment environment is properly set up.
    
    Args:
        stories_dir: Directory containing story files
        required_stories: Optional list of required story names
        
    Returns:
        True if setup is valid, False otherwise
    """
    logger = logging.getLogger("multi_pool_experiments")
    
    # Check stories directory
    if not Path(stories_dir).exists():
        logger.error(f"Stories directory '{stories_dir}' does not exist")
        return False
    
    # Check for story files
    story_files = list(Path(stories_dir).glob("*.txt"))
    if not story_files:
        logger.error(f"No .txt story files found in '{stories_dir}'")
        return False
    
    logger.info(f"Found {len(story_files)} story files")
    
    # Check for required stories
    if required_stories:
        existing_stories = [f.stem for f in story_files]
        missing_stories = [s for s in required_stories if s not in existing_stories]
        if missing_stories:
            logger.error(f"Missing required stories: {missing_stories}")
            return False
    
    # Check environment variables
    required_env_vars = ["LLAMA_API_URL", "LLAMA_API_KEY"]
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_env_vars:
        logger.warning(f"Missing environment variables: {missing_env_vars}")
    
    logger.info("Experiment setup validation completed")
    return True


def cleanup_incomplete_games(experiment_dir: str, min_files_per_game: int = 2) -> None:
    """
    Clean up incomplete game files that may have been left by interrupted runs.
    
    Args:
        experiment_dir: Directory containing experiment files
        min_files_per_game: Minimum files expected per complete game
    """
    logger = logging.getLogger("multi_pool_experiments")
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        return
    
    # Find incomplete game record files
    record_files = list(exp_path.glob("**/game_records_*_game*.txt"))
    incomplete_files = []
    
    for record_file in record_files:
        # Check if file is very small (likely incomplete)
        if record_file.stat().st_size < 1000:  # Less than 1KB
            incomplete_files.append(record_file)
    
    if incomplete_files:
        logger.info(f"Found {len(incomplete_files)} potentially incomplete game files")
        for file_path in incomplete_files:
            logger.info(f"Removing incomplete file: {file_path}")
            file_path.unlink()


def merge_experiment_results(source_dirs: List[str], target_dir: str) -> None:
    """
    Merge results from multiple experiment directories.
    
    Args:
        source_dirs: List of source directories
        target_dir: Target directory for merged results
    """
    logger = logging.getLogger("multi_pool_experiments")
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            continue
        
        # Copy CSV files
        csv_files = list(source_path.glob("**/*.csv"))
        for csv_file in csv_files:
            rel_path = csv_file.relative_to(source_path)
            target_file = target_path / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            if target_file.exists():
                # Merge CSV files
                df1 = pd.read_csv(target_file)
                df2 = pd.read_csv(csv_file)
                merged_df = pd.concat([df1, df2], ignore_index=True)
                merged_df.to_csv(target_file, index=False)
                logger.info(f"Merged CSV: {csv_file} -> {target_file}")
            else:
                shutil.copy2(csv_file, target_file)
                logger.info(f"Copied CSV: {csv_file} -> {target_file}")


def generate_experiment_summary(experiment_dir: str) -> Dict[str, Any]:
    """
    Generate a summary of completed experiments.
    
    Args:
        experiment_dir: Directory containing experiment results
        
    Returns:
        Dictionary with experiment summary statistics
    """
    logger = logging.getLogger("multi_pool_experiments")
    exp_path = Path(experiment_dir)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_path),
        "topologies": {},
        "total_games": 0,
        "total_csv_files": 0
    }
    
    if not exp_path.exists():
        return summary
    
    # Scan for different topologies
    for topology in ["single_pool", "multi_pool"]:
        topology_path = exp_path / topology
        if not topology_path.exists():
            continue
        
        topology_summary = {
            "experiment_types": {},
            "csv_files": 0,
            "games": 0
        }
        
        # Scan experiment types
        for exp_type in ["same_story", "different_story", "bad_apple"]:
            exp_type_path = topology_path / exp_type
            if not exp_type_path.exists():
                continue
            
            csv_files = list(exp_type_path.glob("*.csv"))
            record_files = list(exp_type_path.glob("**/game_records_*.txt"))
            
            exp_type_summary = {
                "csv_files": len(csv_files),
                "record_files": len(record_files),
                "stories": []
            }
            
            # Analyze CSV files for game counts
            total_games = 0
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if "Game" in df.columns:
                        unique_games = df["Game"].nunique()
                        total_games += unique_games
                except Exception as e:
                    logger.warning(f"Error reading {csv_file}: {e}")
            
            exp_type_summary["games"] = total_games
            topology_summary["experiment_types"][exp_type] = exp_type_summary
            topology_summary["csv_files"] += len(csv_files)
            topology_summary["games"] += total_games
        
        summary["topologies"][topology] = topology_summary
        summary["total_csv_files"] += topology_summary["csv_files"]
        summary["total_games"] += topology_summary["games"]
    
    logger.info(f"Experiment summary generated: {summary['total_games']} total games")
    return summary


def save_experiment_summary(experiment_dir: str, output_file: str = "experiment_summary.json") -> None:
    """
    Save experiment summary to JSON file.
    
    Args:
        experiment_dir: Directory containing experiment results
        output_file: Output JSON file name
    """
    summary = generate_experiment_summary(experiment_dir)
    
    output_path = Path(experiment_dir) / output_file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {output_path}")


def check_experiment_progress(experiment_dir: str, expected_games: Dict[str, int]) -> Dict[str, float]:
    """
    Check progress of ongoing experiments.
    
    Args:
        experiment_dir: Directory containing experiment results
        expected_games: Dictionary mapping experiment names to expected game counts
        
    Returns:
        Dictionary with progress percentages
    """
    summary = generate_experiment_summary(experiment_dir)
    progress = {}
    
    for topology, topo_data in summary["topologies"].items():
        for exp_type, exp_data in topo_data["experiment_types"].items():
            exp_key = f"{topology}_{exp_type}"
            completed_games = exp_data["games"]
            expected = expected_games.get(exp_key, 100)  # Default expectation
            
            progress[exp_key] = min(100.0, (completed_games / expected) * 100)
    
    return progress


def estimate_remaining_time(experiment_dir: str, expected_games: Dict[str, int], 
                          avg_time_per_game: float = 300) -> Dict[str, float]:
    """
    Estimate remaining time for ongoing experiments.
    
    Args:
        experiment_dir: Directory containing experiment results
        expected_games: Dictionary mapping experiment names to expected game counts
        avg_time_per_game: Average time per game in seconds
        
    Returns:
        Dictionary with estimated remaining time in hours
    """
    progress = check_experiment_progress(experiment_dir, expected_games)
    remaining_time = {}
    
    for exp_key, progress_pct in progress.items():
        if progress_pct < 100:
            expected = expected_games.get(exp_key, 100)
            completed = int((progress_pct / 100) * expected)
            remaining_games = expected - completed
            remaining_seconds = remaining_games * avg_time_per_game
            remaining_time[exp_key] = remaining_seconds / 3600  # Convert to hours
        else:
            remaining_time[exp_key] = 0.0
    
    return remaining_time


def create_experiment_config_template(output_file: str = "experiment_config.json") -> None:
    """
    Create a template configuration file for experiments.
    
    Args:
        output_file: Output configuration file name
    """
    config_template = {
        "llm_choice": "llama",
        "llama": {
            "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "temperature": 0.6,
            "max_tokens": 500,
            "api_url": "https://api.deepinfra.com/v1/openai/chat/completions",
            "api_key": "YOUR_API_KEY_HERE"
        },
        "experiment": {
            "stories_dir": "stories",
            "results_dir": "experiments",
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "request_timeout": 60
        },
        "experiment_parameters": {
            "same_story": {
                "num_games": 100,
                "num_agents_list": [4, 16, 32],
                "num_rounds": 5,
                "endowment": 10,
                "multiplier": 1.5
            },
            "different_story": {
                "num_games": 400,
                "num_agents_list": [4],
                "num_rounds": 5,
                "endowment": 10,
                "multiplier": 1.5
            },
            "bad_apple": {
                "num_games": 100,
                "num_agents_list": [4],
                "num_rounds": 5,
                "endowment": 10,
                "multiplier": 1.5
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"Configuration template created: {output_file}")
    print("Please update the API key and other settings as needed.")