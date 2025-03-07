#!/bin/bash
#SBATCH --array=0-11
#SBATCH --job-name="Story-Agents"
#SBATCH --container-mounts=/netscratch/$USER:/netscratch/$USER
#SBATCH --container-image=/netscratch/$USER/story_agents/pipeline_LLM.sqsh
#SBATCH --container-workdir=/netscratch/$USER/story_agents
#SBATCH --ntasks=1
#SBATCH --partition=V100-32GB,L40S,L40S-DSA,batch,A100-40GB,RTX3090,RTXA6000
#SBATCH --gpus=0
#SBATCH --mem=40GB
#SBATCH --verbose
#SBATCH --output=same_story_temp_%A_%a.txt

export LLAMA_API_URL="your-hosted-url"
python main.py --exp_type same_story --story_index $SLURM_ARRAY_TASK_ID