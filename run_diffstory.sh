export LLAMA_API_URL="your-hosted-url"
srun \
    --job-name="Story-Agents" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,"$(pwd)":"$(pwd)" \
    --container-image=/netscratch/$USER/story_agents/pipeline_LLM.sqsh \
    --container-workdir="$(pwd)" \
    --ntasks=1 \
    --partition=V100-32GB,L40S,L40S-DSA,batch,A100-40GB,RTX3090,RTXA6000 \
    --gpus=0 \
    --mem=40GB \
    --time=2-00:00:00 \
    --immediate=3500 \
    --verbose \
    --output=/dev/stdout \
    python main.py --exp_type different_story