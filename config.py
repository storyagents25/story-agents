"""
Configuration management for multi-pool story agents experiments.
Handles environment variables, defaults, validation, and logging setup.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import json
import logging
from typing import Dict, Any, Optional


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_file: Optional path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    config = {
        "llm_choice": os.getenv("LLM_CHOICE", "llama"),
        "llama": {
            "model_name": os.getenv("LLAMA_MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            "temperature": float(os.getenv("LLAMA_TEMP", "0.6")),
            "max_tokens": int(os.getenv("LLAMA_MAX_TOKENS", "500")),
            "api_url": os.getenv("LLAMA_API_URL", "https://api.deepinfra.com/v1/openai/chat/completions"),
            "api_key": os.getenv("LLAMA_API_KEY")
        },
        "openai": {
            "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            "temperature": float(os.getenv("OPENAI_TEMP", "1.0")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "experiment": {
            "stories_dir": os.getenv("STORIES_DIR", "stories"),
            "results_dir": os.getenv("RESULTS_DIR", "experiments"),
            "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "3")),
            "retry_delay": float(os.getenv("RETRY_DELAY", "1.0")),
            "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "60")),
            "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "50")),
            "api_delay": float(os.getenv("API_DELAY", "0.1"))
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": os.getenv("LOG_FILE", None)
        }
    }
    
    # Override with config file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Recursively update nested dictionaries
                _deep_update(config, file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    return config


def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update nested dictionaries."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Check LLM choice
    if config["llm_choice"] not in ["llama", "openai"]:
        logger.error(f"Invalid LLM choice '{config['llm_choice']}'. Must be 'llama' or 'openai'.")
        return False
    
    # Check API configuration
    llm_config = config[config["llm_choice"]]
    
    if config["llm_choice"] == "llama":
        if not llm_config.get("api_url"):
            logger.error("LLAMA_API_URL not set.")
            return False
        if not llm_config.get("api_key"):
            logger.error("LLAMA_API_KEY not set.")
            return False
    
    elif config["llm_choice"] == "openai":
        if not llm_config.get("api_key"):
            logger.error("OPENAI_API_KEY not set.")
            return False
    
    # Check stories directory
    stories_dir = config["experiment"]["stories_dir"]
    if not os.path.exists(stories_dir):
        logger.error(f"Stories directory '{stories_dir}' does not exist.")
        return False
    
    # Check for story files
    story_files = [f for f in os.listdir(stories_dir) if f.endswith('.txt')]
    if len(story_files) == 0:
        logger.error(f"No .txt story files found in '{stories_dir}'.")
        return False
    
    logger.info(f"Found {len(story_files)} story files")
    
    # Validate numeric parameters
    try:
        if llm_config["temperature"] < 0 or llm_config["temperature"] > 2:
            logger.warning(f"Temperature {llm_config['temperature']} outside typical range [0, 2].")
        
        if llm_config["max_tokens"] < 1:
            logger.error(f"max_tokens must be positive, got {llm_config['max_tokens']}.")
            return False
        
        exp_config = config["experiment"]
        if exp_config["retry_attempts"] < 1:
            logger.error("retry_attempts must be at least 1.")
            return False
        
        if exp_config["request_timeout"] < 1:
            logger.error("request_timeout must be at least 1 second.")
            return False
            
    except (ValueError, KeyError) as e:
        logger.error(f"Error validating numeric parameters: {e}")
        return False
    
    return True


def get_config() -> Dict[str, Any]:
    """Get the current configuration (convenience function)."""
    return load_config()


def print_config(config: Dict[str, Any], hide_sensitive: bool = True) -> None:
    """Print configuration in a readable format."""
    print("Current Configuration:")
    print(f"  LLM Choice: {config['llm_choice']}")
    
    llm_config = config[config["llm_choice"]]
    print(f"  Model: {llm_config['model_name']}")
    print(f"  Temperature: {llm_config['temperature']}")
    print(f"  Max Tokens: {llm_config['max_tokens']}")
    
    if config["llm_choice"] == "llama":
        print(f"  API URL: {llm_config['api_url']}")
        api_key = llm_config.get('api_key', '')
        if api_key and hide_sensitive:
            print(f"  API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else '***'}")
        elif api_key:
            print(f"  API Key: {api_key}")
        else:
            print("  API Key: Not set")
    
    exp_config = config["experiment"]
    print(f"  Stories Directory: {exp_config['stories_dir']}")
    print(f"  Results Directory: {exp_config['results_dir']}")
    print(f"  Retry Attempts: {exp_config['retry_attempts']}")
    print(f"  Rate Limit: {exp_config['rate_limit_per_minute']} calls/minute")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging based on configuration."""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_file = log_config.get("log_file")
    
    logger = logging.getLogger("multi_pool_experiments")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger


def save_config(config: Dict[str, Any], output_file: str) -> None:
    """Save configuration to JSON file."""
    # Remove sensitive information before saving
    safe_config = json.loads(json.dumps(config))  # Deep copy
    
    if "llama" in safe_config and "api_key" in safe_config["llama"]:
        safe_config["llama"]["api_key"] = "REDACTED"
    if "openai" in safe_config and "api_key" in safe_config["openai"]:
        safe_config["openai"]["api_key"] = "REDACTED"
    
    with open(output_file, 'w') as f:
        json.dump(safe_config, f, indent=2)
    
    print(f"Configuration saved to: {output_file}")


def create_config_template(output_file: str = "experiment_config_template.json") -> None:
    """Create a template configuration file."""
    template = {
        "llm_choice": "llama",
        "llama": {
            "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "temperature": 0.6,
            "max_tokens": 500,
            "api_url": "https://api.deepinfra.com/v1/openai/chat/completions",
            "api_key": "YOUR_DEEPINFRA_API_KEY_HERE"
        },
        "openai": {
            "model_name": "gpt-4o-mini",
            "temperature": 1.0,
            "max_tokens": 500,
            "api_key": "YOUR_OPENAI_API_KEY_HERE"
        },
        "experiment": {
            "stories_dir": "stories",
            "results_dir": "experiments",
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "request_timeout": 60,
            "rate_limit_per_minute": 50,
            "api_delay": 0.1
        },
        "logging": {
            "level": "INFO",
            "log_file": null
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Configuration template created: {output_file}")
    print("Please update the API keys and other settings as needed.")


# Default configuration for reference
DEFAULT_CONFIG = {
    "llm_choice": "llama",
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "temperature": 0.6,
        "max_tokens": 500,
        "api_url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "api_key": None
    },
    "openai": {
        "model_name": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 500,
        "api_key": None
    },
    "experiment": {
        "stories_dir": "stories",
        "results_dir": "experiments",
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "request_timeout": 60,
        "rate_limit_per_minute": 50,
        "api_delay": 0.1
    },
    "logging": {
        "level": "INFO",
        "log_file": None
    }
}


if __name__ == "__main__":
    # Test configuration loading and validation
    config = load_config()
    print_config(config)
    
    if validate_config(config):
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration validation failed")
    
    # Create template if needed
    create_config_template()