#!/usr/bin/env python3
"""
Supports both single-pool and multi-pool configurations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import os
import time
from datetime import datetime

from config import load_config, validate_config, setup_logging, print_config
from experiment_runner import (
    run_same_story_experiment,
    run_different_story_experiment
)
from llm_model import test_model_connection
from utils import (
    validate_experiment_setup,
    cleanup_incomplete_games,
    save_experiment_summary
)


def parse_arguments():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Run multi-pool story agents experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single story, single-pool (quick test)
  python main.py --exp_type same_story --story_index 0 --pool_type single --num_games 2

  # Multi-pool experiment
  python main.py --exp_type same_story --story_index 0 --pool_type multi --pool_sizes 2

  # Heterogeneous experiment
  python main.py --exp_type different_story --pool_type single

  # Full production run with custom parameters
  python main.py --exp_type same_story --story_index 5 --num_games 100 --num_agents 4 16 32

  # Test connection and exit
  python main.py --test_connection --verbose
        """
    )
    
    # Core experiment parameters
    parser.add_argument(
        "--exp_type",
        type=str,
        choices=["same_story", "different_story", "bad_apple"],
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--story_index",
        type=int,
        default=0,
        help="Index of story to use (0-11 for same_story/bad_apple)"
    )
    
    # Pool configuration
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Pool configuration: single or multi-pool"
    )
    
    parser.add_argument(
        "--pool_sizes",
        type=int,
        nargs="*",
        default=None,
        help="Pool sizes for multi-pool experiments (e.g., 2)"
    )
    
    # Game parameters
    parser.add_argument(
        "--num_games",
        type=int,
        default=None,
        help="Number of games to run (overrides default)"
    )
    
    parser.add_argument(
        "--num_agents",
        type=int,
        nargs="*",
        default=None,
        help="Number of agents (overrides default)"
    )
    
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of rounds per game"
    )
    
    parser.add_argument(
        "--endowment",
        type=int,
        default=10,
        help="Token endowment per round"
    )
    
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.5,
        help="Multiplier for shared pool returns"
    )
    
    # Configuration
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    # Testing and debugging
    parser.add_argument(
        "--test_connection",
        action="store_true",
        help="Test model connection and exit"
    )
    
    parser.add_argument(
        "--validate_setup",
        action="store_true",
        help="Validate experiment setup and exit"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up incomplete experiment files"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate experiment summary and exit"
    )
    
    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    # Performance tuning
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=None,
        help="API rate limit (calls per minute)"
    )
    
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=None,
        help="Number of retry attempts for failed API calls"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check that we have an action to perform
    if not any([args.exp_type, args.test_connection, args.validate_setup, 
                args.cleanup, args.summary]):
        errors.append("Must specify --exp_type or one of the utility options")
    
    # Validate story index
    if args.exp_type in ["same_story", "bad_apple"] and args.story_index < 0:
        errors.append("Story index must be non-negative")
    
    # Validate pool configuration
    if args.pool_type == "multi" and not args.pool_sizes:
        args.pool_sizes = [2]  # Default for multi-pool
    
    # Validate numeric parameters
    if args.num_rounds < 1:
        errors.append("Number of rounds must be at least 1")
    
    if args.endowment < 1:
        errors.append("Endowment must be at least 1")
    
    if args.multiplier <= 0:
        errors.append("Multiplier must be positive")
    
    if args.num_games and args.num_games < 1:
        errors.append("Number of games must be at least 1")
    
    if args.num_agents and any(n < 1 for n in args.num_agents):
        errors.append("Number of agents must be at least 1")
    
    return errors


def setup_experiment_environment(args, config):
    """Set up the experiment environment and logging."""
    # Update config with command line overrides
    if args.rate_limit:
        config["experiment"]["rate_limit_per_minute"] = args.rate_limit
    
    if args.retry_attempts:
        config["experiment"]["retry_attempts"] = args.retry_attempts
    
    if args.log_file:
        config["logging"]["log_file"] = args.log_file
    
    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    elif args.quiet:
        config["logging"]["level"] = "WARNING"
    
    # Setup logging
    logger = setup_logging(config)
    
    # Create necessary directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    return logger


def main():
    """Main execution function."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        print("Argument validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Load and validate configuration
    try:
        config = load_config(args.config_file)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    if not validate_config(config):
        print("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    # Setup environment and logging
    logger = setup_experiment_environment(args, config)
    
    # Print configuration if verbose
    if args.verbose:
        print_config(config, hide_sensitive=False)
    
    # Handle utility operations
    if args.test_connection:
        logger.info("Testing model connection...")
        if test_model_connection(config, verbose=True):
            print("✓ Model connection test successful")
            sys.exit(0)
        else:
            print("✗ Model connection test failed")
            sys.exit(1)
    
    if args.validate_setup:
        logger.info("Validating experiment setup...")
        if validate_experiment_setup(config["experiment"]["stories_dir"]):
            print("✓ Experiment setup validation successful")
            sys.exit(0)
        else:
            print("✗ Experiment setup validation failed")
            sys.exit(1)
    
    if args.cleanup:
        logger.info("Cleaning up incomplete experiment files...")
        cleanup_incomplete_games(config["experiment"]["results_dir"])
        print("✓ Cleanup completed")
        sys.exit(0)
    
    if args.summary:
        logger.info("Generating experiment summary...")
        save_experiment_summary(config["experiment"]["results_dir"])
        print("✓ Summary generated")
        sys.exit(0)
    
    # Validate required experiment parameters
    if not args.exp_type:
        print("Error: --exp_type is required for running experiments")
        sys.exit(1)
    
    # Set up pool configuration
    pool_sizes = None
    if args.pool_type == "multi":
        pool_sizes = args.pool_sizes or [2]  # Default to pairs
        logger.info(f"Running multi-pool experiment with pool sizes: {pool_sizes}")
    else:
        logger.info("Running single-pool experiment")
    
    # Set default parameters based on experiment type
    if args.exp_type in ["same_story", "bad_apple"]:
        default_num_games = 100
        default_num_agents = [4, 16, 32] if args.exp_type == "same_story" else [4]
    else:  # different_story
        default_num_games = 400
        default_num_agents = [4]
    
    # Override defaults with command line arguments
    num_games = args.num_games or default_num_games
    num_agents_list = args.num_agents or default_num_agents
    
    # Single values for this run
    num_rounds_list = [args.num_rounds]
    endowment_list = [args.endowment]
    multiplier_list = [args.multiplier]
    
    # Log experiment configuration
    logger.info("="*50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*50)
    logger.info(f"Experiment type: {args.exp_type}")
    logger.info(f"Pool type: {args.pool_type}")
    logger.info(f"Pool sizes: {pool_sizes}")
    logger.info(f"Story index: {args.story_index}")
    logger.info(f"Number of games: {num_games}")
    logger.info(f"Agent counts: {num_agents_list}")
    logger.info(f"Rounds per game: {args.num_rounds}")
    logger.info(f"Token endowment: {args.endowment}")
    logger.info(f"Pool multiplier: {args.multiplier}")
    logger.info(f"Configuration file: {args.config_file or 'Environment variables'}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)
    
    try:
        # Run the appropriate experiment
        if args.exp_type == "same_story":
            run_same_story_experiment(
                is_bad_apple=False,
                story_index=args.story_index,
                num_rounds_list=num_rounds_list,
                endowment_list=endowment_list,
                multiplier_list=multiplier_list,
                num_games=num_games,
                num_agents_list=num_agents_list,
                exp_type="same_story",
                pool_sizes=pool_sizes,
                config=config
            )
        
        elif args.exp_type == "bad_apple":
            run_same_story_experiment(
                is_bad_apple=True,
                story_index=args.story_index,
                num_rounds_list=num_rounds_list,
                endowment_list=endowment_list,
                multiplier_list=multiplier_list,
                num_games=num_games,
                num_agents_list=num_agents_list,
                exp_type="bad_apple",
                pool_sizes=pool_sizes,
                config=config
            )
        
        elif args.exp_type == "different_story":
            run_different_story_experiment(
                num_rounds_list=num_rounds_list,
                endowment_list=endowment_list,
                multiplier_list=multiplier_list,
                num_games=num_games,
                num_agents_list=num_agents_list,
                exp_type="different_story",
                pool_sizes=pool_sizes,
                config=config
            )
        
        # Calculate total runtime
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("="*50)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"Total runtime: {total_time/3600:.2f} hours")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        
        print(f"✓ Experiment '{args.exp_type}' completed successfully!")
        print(f"  Total runtime: {total_time/3600:.2f} hours")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        print("\n⚠ Experiment interrupted by user.")
        
        # Save partial results summary
        try:
            save_experiment_summary(config["experiment"]["results_dir"], 
                                  "experiment_summary_partial.json")
            print("Partial results summary saved.")
        except Exception as e:
            logger.error(f"Failed to save partial summary: {e}")
        
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        print(f"✗ Experiment failed with error: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        # Save error summary
        try:
            save_experiment_summary(config["experiment"]["results_dir"], 
                                  "experiment_summary_error.json")
        except Exception as summary_error:
            logger.error(f"Failed to save error summary: {summary_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()