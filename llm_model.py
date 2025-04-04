import os
import time
import requests
from typing import List, Dict, Any, TextIO
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

# -----------------------------
# Central Configuration
# -----------------------------
config: Dict[str, Any] = {
    "llm_choice": os.getenv("LLM_CHOICE", "llama"),
    "llama": {
        "model_name": os.getenv("LLAMA_MODEL_NAME", "meta-llama-3.3-70b-instruct-fp8"),
        "temperature": float(os.getenv("LLAMA_TEMP", 0.6)),
        "max_tokens": int(os.getenv("LLAMA_MAX_TOKENS", 500)),
        "api_url": os.getenv("LLAMA_API_URL")
    },
    "openai": {
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        "temperature": float(os.getenv("OPENAI_TEMP", 1.0)),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", 10)),
        "api_key": os.getenv("OPENAI_API_KEY")
    },
}

# -----------------------------
# Llama Helper Class
# -----------------------------
class Llama:
    def __init__(self, model_name: str, temperature: float, max_tokens: int, api_url: str) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = api_url

    def invoke(self, history: List[BaseMessage]) -> AIMessage:
        # Build messages in a robust way.
        messages: List[Dict[str, str]] = []
        for msg in history:
            # Try to extract role and content, whether msg is an object or dict.
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", str(msg))
            else:
                role = "unknown"
                content = str(msg)
            messages.append({"role": role, "content": content})

        data: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        try:
            response = requests.post(self.api_url, json=data, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Llama API request timed out.")
        except requests.exceptions.HTTPError as http_err:
            raise RuntimeError(f"Llama API HTTP error: {http_err}")
        except requests.exceptions.RequestException as err:
            raise RuntimeError(f"Llama API connection error: {err}")

        try:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as parse_err:
            raise RuntimeError(f"Failed to parse Llama API response: {parse_err}")

        return AIMessage(content=content)

def log_records(message: str, records_file: TextIO) -> None:
    """
    Helper function to log a message to records file so we can keep track of all history.
    """
    records_file.write(message + "\n")
    records_file.flush() # make sure it's written immediately

def calculate_payoffs(contributions: List[float], e: float, m: float, na: int) -> List[float]:
    """
    Given a list of contributions from each agent, compute the payoff for each agent.
    """
    total: float = sum(contributions)
    shared_bonus: float = m * total / na
    return [e - c + shared_bonus for c in contributions]

# -----------------------------
# Agent Class
# -----------------------------
class Agent:
    """
    A simple agent that uses an LLM backend determined by LLM_CHOICE.
    It maintains its own conversation history so that previous rounds and summaries
    can influence its responses.
    """
    def __init__(self, name: str, system_message: str, records_file: TextIO) -> None:
        self.name = name
        # Start the conversation with a system message.
        self.history: List[BaseMessage] = [SystemMessage(content=system_message)]
        self.records_file = records_file

        # Log the creation of this agent and its system prompt to records.
        log_records(f"CREATING AGENT: {self.name}", records_file)
        log_records(f"System Prompt for {self.name}: {system_message}", records_file)

        # Create the appropriate LLM instance
        llm_choice = config["llm_choice"]
        if llm_choice == "llama":
            llama_cfg = config["llama"]
            self.llm = Llama(
                model_name=llama_cfg["model_name"],
                temperature=llama_cfg["temperature"],
                max_tokens=llama_cfg["max_tokens"],
                api_url=llama_cfg["api_url"]
            )
        else:
            openai_cfg = config["openai"]
            self.llm = ChatOpenAI(
                model_name=openai_cfg["model_name"],
                temperature=openai_cfg["temperature"],
                max_tokens=openai_cfg["max_tokens"],
                openai_api_key=openai_cfg["api_key"]
            )

    def chat(self, message: str) -> str:
        """
        Append a human message, call the LLM, append the assistant's reply,
        log everything, and return the response content.
        """
        log_records(f"{self.name} receives HUMAN message: {message}", self.records_file)
        self.history.append(HumanMessage(content=message))
        try:
            response = self.llm.invoke(self.history) # Use the pre-created ChatOpenAI instance.
        except RuntimeError as err:
            log_records(f"{self.name} ERROR: {err}", self.records_file) 
            return f"[ERROR]: {err}"

        # Store the response in the conversation history
        self.history.append(response)
        # Brief pause to help avoid rate limits
        log_records(f"{self.name} responds ASSISTANT: {response.content}", self.records_file)       
        time.sleep(0.01)
        return response.content

# -----------------------------
# DummyAgent Class
# -----------------------------
class DummyAgent:
    """
    A 'dummy' agent that does NOT connect to an LLM and always contributes 0.
    """
    def __init__(self, name: str, system_message: str, records_file: TextIO) -> None:
        self.name = name
        self.history: List[BaseMessage] = [SystemMessage(content=system_message)]
        self.records_file = records_file

        log_records(f"CREATING DUMMY AGENT: {self.name}", records_file)
        log_records(f"(Dummy) System Prompt for {self.name}: {system_message}", records_file)

    def chat(self, message: str) -> str:
        """
        This agent always contributes "0".
        We still log the conversation but do not call any LLM.
        """
        log_records(f"{self.name} (dummy) receives HUMAN message: {message}", self.records_file)
        # We return "0" as a string to emulate the minimal integer-based response.
        response_content = "<TOKEN>0</TOKEN>"
        log_records(f"{self.name} (dummy) responds ASSISTANT: {response_content}", self.records_file)
        time.sleep(0.01)  # a brief pause, mirroring the normal agent
        return response_content