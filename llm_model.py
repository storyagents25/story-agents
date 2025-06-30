"""
LLM interface classes for multi-pool story agents experiments.
Supports both Llama and OpenAI models.
Includes comprehensive error handling, retry logic, and rate limiting.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

from config import get_config


class LlamaModel:
    """
    Llama model interface with enhanced error handling and retry logic.
    Supports OpenAI-compatible APIs.
    """
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int, 
                 api_url: str, api_key: str = None, timeout: int = 60, 
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"{__name__}.LlamaModel")
        
        if not self.api_key:
            raise ValueError("LLAMA_API_KEY must be set in environment or passed as parameter")
        
        self.logger.info(f"Initialized Llama model: {model_name}")

    def _build_messages(self, history: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain message history to API format."""
        messages = []
        for msg in history:
            if hasattr(msg, "content"):
                content = msg.content
            else:
                content = str(msg)
            
            # Map LangChain message types to API roles
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                raw = getattr(msg, "role", None)
                role = raw if raw in ("system", "user", "assistant") else "user"

            messages.append({"role": role, "content": content})
        
        return messages

    def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API request with comprehensive retry logic."""
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        } if self.api_key else {"Content-Type": "application/json"}

        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making API request (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.api_url, 
                    json=data, 
                    headers=headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                error_msg = f"Request timed out (attempt {attempt + 1}/{self.max_retries})"
                self.logger.warning(error_msg)
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"API request timed out after {self.max_retries} attempts")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else None
                
                if status_code == 429:  # Rate limit
                    error_msg = f"Rate limited (attempt {attempt + 1}/{self.max_retries})"
                    self.logger.warning(error_msg)
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Rate limited after {self.max_retries} attempts: {e}")
                    # Longer wait for rate limits
                    wait_time = self.retry_delay * (3 ** attempt)
                    self.logger.info(f"Waiting {wait_time:.1f}s due to rate limit")
                    time.sleep(wait_time)
                    
                elif status_code == 503:  # Service unavailable
                    error_msg = f"Service unavailable (attempt {attempt + 1}/{self.max_retries})"
                    self.logger.warning(error_msg)
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Service unavailable after {self.max_retries} attempts: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    
                elif status_code in [400, 401, 403]:  # Client errors - don't retry
                    raise RuntimeError(f"Client error (HTTP {status_code}): {e}")
                    
                else:  # Other HTTP errors
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"HTTP error after {self.max_retries} attempts: {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                error_msg = f"Connection error (attempt {attempt + 1}/{self.max_retries})"
                self.logger.warning(error_msg)
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Connection error after {self.max_retries} attempts: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                error_msg = f"Request error (attempt {attempt + 1}/{self.max_retries})"
                self.logger.warning(error_msg)
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Request error after {self.max_retries} attempts: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

        # Should not reach here, but just in case
        raise RuntimeError(f"Unexpected error after {self.max_retries} attempts: {last_exception}")

    def invoke(self, history: List[BaseMessage]) -> AIMessage:
        """
        Invoke the model with message history.
        
        Args:
            history: List of LangChain messages
            
        Returns:
            AIMessage with model response
        """
        messages = self._build_messages(history)
        
        try:
            response_data = self._make_request(messages)
            content = response_data["choices"][0]["message"]["content"]
            
            self.logger.debug(f"Successful API response: {len(content)} characters")
            return AIMessage(content=content)
            
        except (ValueError, KeyError, IndexError) as parse_err:
            self.logger.error(f"Failed to parse API response: {parse_err}")
            raise RuntimeError(f"Failed to parse API response: {parse_err}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_url": self.api_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }


class ModelFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_model(config: Optional[Dict[str, Any]] = None) -> Union[LlamaModel, ChatOpenAI]:
        """
        Create appropriate LLM instance based on configuration.
        
        Args:
            config: Configuration dictionary (if None, loads from environment)
            
        Returns:
            LLM instance
        """
        if config is None:
            config = get_config()
        
        llm_choice = config["llm_choice"]
        logger = logging.getLogger(f"{__name__}.ModelFactory")
        
        if llm_choice == "llama":
            llama_cfg = config["llama"]
            exp_cfg = config["experiment"]
            
            logger.info(f"Creating Llama model: {llama_cfg['model_name']}")
            
            return LlamaModel(
                model_name=llama_cfg["model_name"],
                temperature=llama_cfg["temperature"],
                max_tokens=llama_cfg["max_tokens"],
                api_url=llama_cfg["api_url"],
                api_key=llama_cfg["api_key"],
                timeout=exp_cfg.get("request_timeout", 60),
                max_retries=exp_cfg.get("retry_attempts", 3),
                retry_delay=exp_cfg.get("retry_delay", 1.0)
            )
            
        elif llm_choice == "openai":
            openai_cfg = config["openai"]
            
            logger.info(f"Creating OpenAI model: {openai_cfg['model_name']}")
            
            return ChatOpenAI(
                model_name=openai_cfg["model_name"],
                temperature=openai_cfg["temperature"],
                max_tokens=openai_cfg["max_tokens"],
                openai_api_key=openai_cfg["api_key"]
            )
            
        else:
            raise ValueError(f"Unsupported LLM choice: {llm_choice}")


class RateLimiter:
    """
    Simple rate limiter for API calls to prevent overwhelming the API.
    """
    
    def __init__(self, calls_per_minute: int = 60, buffer_time: float = 0.1):
        self.calls_per_minute = calls_per_minute
        self.buffer_time = buffer_time
        self.call_times = []
        self.logger = logging.getLogger(f"{__name__}.RateLimiter")
        
        self.logger.info(f"Rate limiter initialized: {calls_per_minute} calls/minute")
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.calls_per_minute:
            # Need to wait
            oldest_call = min(self.call_times)
            wait_time = 60 - (now - oldest_call) + self.buffer_time
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        self.call_times.append(now)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        recent_calls = [t for t in self.call_times if now - t < 60]
        
        return {
            "calls_last_minute": len(recent_calls),
            "calls_per_minute_limit": self.calls_per_minute,
            "utilization_pct": (len(recent_calls) / self.calls_per_minute) * 100
        }


def test_model_connection(config: Optional[Dict[str, Any]] = None, 
                         verbose: bool = True) -> bool:
    """
    Test model connection with a simple prompt.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print detailed output
        
    Returns:
        True if connection successful, False otherwise
    """
    logger = logging.getLogger(f"{__name__}.test_connection")
    
    try:
        if verbose:
            print("Testing model connection...")
        
        model = ModelFactory.create_model(config)
        test_message = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say 'Hello' if you can hear me.")
        ]
        
        start_time = time.time()
        response = model.invoke(test_message)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if verbose:
            print(f"  Model test successful!")
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Response: {response.content[:100]}...")
            
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                print(f"  Model: {info['model_name']}")
                print(f"  Temperature: {info['temperature']}")
        
        logger.info(f"Model connection test successful in {response_time:.2f}s")
        return True
        
    except Exception as e:
        error_msg = f"Model test failed: {e}"
        logger.error(error_msg)
        if verbose:
            print(f"✗ {error_msg}")
        return False


def benchmark_model_performance(config: Optional[Dict[str, Any]] = None, 
                              num_requests: int = 5) -> Dict[str, float]:
    """
    Benchmark model performance with multiple requests.
    
    Args:
        config: Configuration dictionary
        num_requests: Number of test requests to make
        
    Returns:
        Performance statistics
    """
    logger = logging.getLogger(f"{__name__}.benchmark")
    
    try:
        model = ModelFactory.create_model(config)
        response_times = []
        
        print(f"Benchmarking model performance with {num_requests} requests...")
        
        for i in range(num_requests):
            test_message = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Count from 1 to {i+3}.")
            ]
            
            start_time = time.time()
            response = model.invoke(test_message)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"  Request {i+1}: {response_time:.2f}s")
        
        # Calculate statistics
        stats = {
            "mean_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_time": sum(response_times)
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Mean response time: {stats['mean_response_time']:.2f}s")
        print(f"  Min response time: {stats['min_response_time']:.2f}s")
        print(f"  Max response time: {stats['max_response_time']:.2f}s")
        print(f"  Total time: {stats['total_time']:.2f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test script when run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM model connection")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--requests", type=int, default=5, help="Number of benchmark requests")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = get_config() if not args.config else get_config(args.config)
    
    # Test connection
    if test_model_connection(config):
        if args.benchmark:
            benchmark_model_performance(config, args.requests)
    else:
        sys.exit(1)