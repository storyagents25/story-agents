
srun \
    --job-name="Story-Agents" \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,"$(pwd)":"$(pwd)" \
    --container-image=/netscratch/$USER/story_agents/pipeline_LLM.sqsh \
    --container-workdir="$(pwd)" \
    --ntasks=1 \
    --partition=V100-32GB,L40S,L40S-DSA,batch,A100-40GB,RTX3090,RTXA6000 \
    --gpus=0 \
    --mem=40GB \
    --time=1:00:00 \
    --immediate=3500 \
    --verbose \
    --output=/dev/stdout \
    pip install seaborn && python visualise_collaboration.py

