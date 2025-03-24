# -----------------------------
# Global Settings and Imports
# -----------------------------
import os
import time
import requests  # Needed for llama requests
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Set your backend here: choose "llama" or "openai"
LLM_CHOICE = "llama"

# For OpenAI backend (if used)
# SET YOUR OPENAI API KEY HERE
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
LLAMA_API_URL = os.getenv("LLAMA_API_URL")
# -----------------------------
# Llama Helper Class
# -----------------------------
# This class implements an LLM interface similar to ChatOpenAI,
# but calls the llama endpoint via a POST request.
class Llama:
    def __init__(self, model_name, temperature, max_tokens, openai_api_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, history):
        # Build messages in a robust way.
        messages = []
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

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        response = requests.post(LLAMA_API_URL, json=data)
        # Optionally add error checking for response status here
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        from langchain.schema import AIMessage
        return AIMessage(content=content)

# -----------------------------
# SET YOUR OPENAI API KEY HERE
# -----------------------------
# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


def log_records(message: str, records_file):
    """
    Helper function to log a message to records file so we can keep track of all history.
    """
    records_file.write(message + "\n")
    records_file.flush()  # make sure it's written immediately


def calculate_payoffs(contributions, e, m, na):
    """
    Given a list of contributions from each agent, compute the payoff for each agent.
    """
    total = sum(contributions)
    shared_bonus = m * total / na
    return [e - c + shared_bonus for c in contributions]


class Agent:
    """
    A simple agent that uses an LLM backend determined by LLM_CHOICE.
    It maintains its own conversation history so that previous rounds and summaries
    can influence its responses.
    """
    def __init__(self, name, system_message, records_file):
        self.name = name
        # Start the conversation with a system message.
        self.history = [SystemMessage(content=system_message)]

        # Log the creation of this agent and its system prompt to records.
        log_records(f"CREATING AGENT: {self.name}", records_file)
        log_records(f"System Prompt for {self.name}: {system_message}", records_file)
        self.records_file = records_file


        # Create the appropriate LLM instance
        if LLM_CHOICE == "llama":
            self.llm = Llama(
            model_name="meta-llama-3.3-70b-instruct-fp8",  # adjust as needed
            temperature=0.6,
            max_tokens=500,
        )
        else: 
            # Create one instance of ChatOpenAI per agent.
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini",  # adjust as needed
                temperature=1.0,
                max_tokens=10,
                openai_api_key=OPENAI_API_KEY
            )

    def chat(self, message: str) -> str:
        """
        Append a human message, call the LLM, append the assistant's reply,
        log everything, and return the response content.
        """
        log_records(f"{self.name} receives HUMAN message: {message}", self.records_file)
        self.history.append(HumanMessage(content=message))
        # Use the pre-created ChatOpenAI instance.
        response = self.llm.invoke(self.history)
        # Store the response in the conversation history
        self.history.append(response)
        log_records(f"{self.name} responds ASSISTANT: {response.content}", self.records_file)
        # Brief pause to help avoid rate limits
        time.sleep(0.01)
        return response.content
    
class DummyAgent:
    """
    A 'dummy' agent that does NOT connect to an LLM and always contributes 0.
    """
    def __init__(self, name, system_message, records_file):
        self.name = name
        self.history = [SystemMessage(content=system_message)]
        self.records_file = records_file

        log_records(f"CREATING DUMMY AGENT: {self.name}", records_file)
        log_records(f"(Dummy) System Prompt for {self.name}: {system_message}", records_file)
        # No LLM needed.

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