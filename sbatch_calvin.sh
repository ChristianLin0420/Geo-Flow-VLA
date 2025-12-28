#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch

#SBATCH --time=04:00:00                 # Adjust time limit as needed

#SBATCH --mem=0                         # all mem avail
#SBATCH --overcommit                    # allows more than one process per CPU
#SBATCH --dependency=singleton
#SBATCH --exclusive                     # exclusive node access
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32 # number of cores
#SBATCH --job-name=geo-flow-vla-calvin-trainD-phase1-train # customize your job name
#SBATCH --output=./logs/%x-%j/stdout.log      # !!!! Update log NAME Here
#SBATCH --error=./logs/%x-%j/stderr.log      # !!!! Update log NAME Here
#SBATCH --mail-type=BEGIN,END,FAIL # Adjust event types as needed
#SBATCH --mail-user=chrislin@nvidia.com

# Array of commands to execute
COMMANDS=(
    "pip install pyarrow pandas && bash scripts/train.sh phase1 --gpus all --config calvin_config --epochs 50 --batch-size 1024 --name phase1-world-model-bs1024-epochs50 data.num_workers=0 checkpoint.resume=./checkpoints/calvin/phase1/latest.pt"
    "pip install pyarrow pandas && bash scripts/train.sh phase1 --gpus all --config calvin_config --epochs 50 --batch-size 1024 --name phase1-world-model-bs1024-epochs50 data.num_workers=0 checkpoint.resume=./checkpoints/calvin/phase1/latest.pt"
    "pip install pyarrow pandas && bash scripts/train.sh phase1 --gpus all --config calvin_config --epochs 50 --batch-size 1024 --name phase1-world-model-bs1024-epochs50 data.num_workers=0 checkpoint.resume=./checkpoints/calvin/phase1/latest.pt"
    "pip install pyarrow pandas && bash scripts/train.sh phase1 --gpus all --config calvin_config --epochs 50 --batch-size 1024 --name phase1-world-model-bs1024-epochs50 data.num_workers=0 checkpoint.resume=./checkpoints/calvin/phase1/latest.pt"
    "pip install pyarrow pandas && bash scripts/train.sh phase1 --gpus all --config calvin_config --epochs 50 --batch-size 1024 --name phase1-world-model-bs1024-epochs50 data.num_workers=0 checkpoint.resume=./checkpoints/calvin/phase1/latest.pt"
)

# File to store the current index
INDEX_FILE="/home/chrislin/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_command_index.txt"

# If the index file doesn't exist, start from 0
if [[ ! -f $INDEX_FILE ]]; then
    CURRENT_INDEX=0
    echo "0" > $INDEX_FILE
else
    # If file exists, read the current index from the file
    CURRENT_INDEX=$(cat $INDEX_FILE)
fi

# Check if the index is within the range of the array
if [[ $CURRENT_INDEX -ge ${#COMMANDS[@]} ]]; then
    echo "All commands processed. Exiting."
    exit 0
fi

# Get the current command
CURRENT_COMMAND=${COMMANDS[$CURRENT_INDEX]}

# Execute the command, set the time out less than the 4 hours limit
echo "Running command: $CURRENT_COMMAND"

# Around line 48, BEFORE the srun command
# Get node list outside container where scontrol is available
if [ ! -z "$SLURM_JOB_NODELIST" ]; then
    export NODE_IPS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
    echo "Detected nodes from SLURM: $NODE_IPS"
fi

# Then in the srun command, pass it as an environment variable
timeout 3.95h srun --ntasks-per-node=1 --container-image="christianlin0420/geo-flow-vla:latest" \
    --container-mounts=/lustre:/lustre,/lustre/fsw/portfolios/edgeai/users/chrislin/projects/Geo-Flow-VLA:/workspace \
    --container-writable \
    bash -c "
export NODE_IPS='$NODE_IPS'
echo 'Starting training...'
cd /lustre/fsw/portfolios/edgeai/users/chrislin/projects/Geo-Flow-VLA;
$CURRENT_COMMAND"

# Command completed, move to the next command
CURRENT_INDEX=$((CURRENT_INDEX + 1))
echo "Command completed. Moving to index $CURRENT_INDEX."

# Save the updated index to the file
echo $CURRENT_INDEX > $INDEX_FILE

# If there are more commands, requeue with updated index
if [[ $CURRENT_INDEX -lt ${#COMMANDS[@]} ]]; then
    echo "Requeuing for next command..."
    scontrol requeue $SLURM_JOB_ID
else
    echo "All commands finished successfully."
    exit 0
fi