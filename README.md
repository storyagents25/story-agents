# **Story Agents - A Multi-Agent Public Goods Experiment**

## **Overview**
**Story Agents** is a multi-agent framework for studying how different storytelling influences cooperation in a repeated **Public Goods Game**. The framework allows experiments where **agents contribute tokens**, earn payoffs, and analyze the effects of different story assignments.

This project consists of **three main experiments**:
1. **Same Story Experiment**: All agents receive the same story from a set of 12 stories and play 100 games across different agent sizes ([4, 16, 32]).
2. **Different Story Experiment**: Each agent is assigned a random story, and 200 games are run with 16 agents per game.
3. **Robustness Experiment**: Similar to the Same Story Experiment but introduces a **dummy agent** who always contributes **0 tokens** to test the impact of non-cooperative players.

The study tracks the following metrics:
- **Contributions** per round
- **Round Payoffs**
- **Cumulative Payoffs**
- **Collaboration Score**

---

## **Features**
- Multi-agent framework with LLM-powered decision-making  
- Customizable storytelling influence on cooperation  
- Three experiment types to evaluate different scenarios  
- Automatic logging of results and intermediate saving for resuming experiments  
- Supports different model configurations (LLMs, rule-based, and dummy agents)   

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/storyagents25/story-agents
```
```bash
cd story-agents
```
## **2. Installation Dependencies**
You can set up the project using **Conda** or **Pip**.

### **Option 1: Using Conda**
Install dependencies using the provided **Conda environment file**:
```bash
conda env create -f environment.yml
```
```bash
conda activate story-agents
```

### **Option 2: Using Pip**
```bash
pip install -U langchain-community
```
```bash
pip install --upgrade langchain_openai -q
```
```bash
pip install numpy pandas matplotlib seaborn jupyterlab
```

## **Setup API Keys**
### **For OpenAI Models**
export OPENAI_API_KEY="your-api-key"
### **For LLAMA Model**
export LLAMA_API_URL="your-hosted-url"

## **Usage**
### **Running the Experiments**
You can run different experiments using the `main.py` script.

---

### **1. Same Story Experiment**
Runs **100 games per story** for agent sizes **[4, 16, 32]**.
To parallelly run the experiment for each story:
```bash
sbatch run_samestory.sh
```
Internally, this runs
```bash
python main.py --exp_type same_story --story_index <STORY_INDEX>
```
### **2. Different Story Experiment**
Assigns a random story to each agent and runs 200 games with 16 agents.
```bash
python main.py --exp_type different_story
```
### **3. Robustness Experiment (Bad Apple)**
Same as the same story experiment, but introduces one dummy agent who always contributes 0. <br>
To parallelly run the experiment for each story:
```bash
sbatch run_samestory.sh
```
Internally, this runs
```bash
python main.py --exp_type bad_apple --story_index <STORY_INDEX>
```
