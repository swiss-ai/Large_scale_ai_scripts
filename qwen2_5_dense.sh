#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen2_5_14B
#SBATCH --output=/iopsstor/scratch/cscs/%u/slurmlogs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/slurmlogs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
# NOTE(tj.solergibert) Check the `Data` section in the README. Use `,` to specify multiple datasets e.g. "/path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C"
# DATASETS="/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-keeprobots-merge/dump-1-merged"

DATASET_NAME="climbmix"

# fineweb-edu-100B
if [ "$DATASET_NAME" == "fineweb-edu-100B" ]; then
	DATASETS="1.0 /iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B/fineweb-edu-100B_00002_tokens 1.0 /iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B/fineweb-edu-100B_00000_tokens 1.0 /iopsstor/scratch/cscs/anowak/datasets/megatron/llama_tokenized/fineweb-edu-100B/fineweb-edu-100B_00001_tokens"
fi

# nemotron-climbmix: hftokenizer + swiss-ai/Apertus-8B-2509 vocab
if [ "$DATASET_NAME" == "climbmix" ]; then
	NUM_FILES=100
	for (( i=0; i<$NUM_FILES; i++ ))
	do
		DATASETS+="1.0 /iopsstor/scratch/cscs/gfu/datasets/climbmix/hftokenized/part_${i}_text_document "
	done
fi

# dynamic experiment name
EXP_NAME_SUFFIX=""

# training configs
MBS=2 # Micro batch size
GBS=96 # Global batch size
TP=2 # Tensor parallelism (reduced: MLA reduces attention memory)
ETP=1
EP=1 # Expert parallelism (for MoE model only)
PP=2 # Pipeline parallelism
CP=1 # context parallelism

SEQ_LEN=4096 # Sequence length 
TRAINING_STEPS=10000
CHECKPOINT_STEPS=30


#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=false # Set to `true` to use mock data
###################

# Megatron source and dataset cache
MEGATRON_LM_DIR=/users/$USER/frameworks/Megatron-LM
# export PYTHONPATH=${MEGATRON_PATCH_DIR}:${MEGATRON_PATCH_DIR}/backends/megatron/Megatron-LM-250908:$PYTHONPATH

# Dataset cache
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

# Container image
IMAGE_ENV=/iopsstor/scratch/cscs/$USER/ce-images/alps-pytorch2512.toml

#### Checkpointing & Resuming ####
LOAD_CKPT=false # Set to `true` to load from checkpoint
AUTO_JOB_REQUEUE=false # Set to `true` to continuously submit jobs to Slurm until training is complete. Enable it once you are sure of the cost involved in running this experiment.
BACKUP_CODEBASE=false # Set to `true` to copy the codebase to the experiment folder and re-use it across runs

# Logging directories & artifacts
PROJECT_NAME=qwen2_5_14B_${DATASET_NAME}
EXP_NAME=qwen2_5_14B-${DATASET_NAME}-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz-${MBS}mbsz-${TP}tp-${PP}pp-${EP}ep-${ETP}etp-${CP}cp-${EXP_NAME_SUFFIX}
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
NSYS_DIR="/iopsstor/scratch/cscs/$USER/slurmlogs/nsys/"

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=/iopsstor/scratch/cscs/$USER/megatron-runs/$PROJECT_NAME/$EXP_NAME/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=25679
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args #### 
# Check megatron/training/arguments.py
TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--use-precision-aware-optimizer
	--main-grads-dtype bf16
)

# Based on the Qwen 2.5 14B model.
NETWORK_SIZE_ARGS=(
	--num-layers 48
	--hidden-size 5120
	--ffn-hidden-size 13824

	# GQA
	--num-attention-heads 40
	--group-query-attention
	--num-query-groups 8

	--max-position-embeddings 131072
	--position-embedding-type rope
	--rotary-base 1000000
	--use-rope-scaling
	--rope-scaling-factor 40
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--norm-epsilon 1e-5
	--swiglu
	--untie-embeddings-and-output-weights
	--attention-backend auto
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

# Recompute to reduce memory usage with minimal compute overhead
RECOMPUTE_ARGS=(
	--recompute-granularity selective
	--recompute-modules layernorm
)

TRAINING_ARGS=(
	# --use-mcore-models
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS

	# Evaluation during training
	--eval-interval 1000000	# disable
	--eval-iters 1
	
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer adam  # ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	# --exit-signal-handler
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

# NOTE(tj.solergibert) Check all the arguments in megatron/training/arguments.py#L1548 or https://github.com/NVIDIA/Megatron-LM/blob/0dd78ddcdb117ce4f2e9761449274d87af717674/megatron/training/arguments.py#L1548-L1606
LEARNING_RATE_ARGS=(
	--lr 0.0003
	--min-lr 0.00003  # x10 reduction
	--lr-decay-style WSD  # WSD schedule
	--lr-warmup-iters 30
	--lr-wsd-decay-style linear  # WSD schedule
	--lr-wsd-decay-iters 100  # WSD edcay will be a different run
)

# NOTE(tj.solergibert) Check the `Checkpointing` section in the README
CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--no-load-optim
	--async-save
)

if [ "$LOAD_CKPT" = false ]; then
	# If not loading from checkpoint, start fresh and ignore existing checkpoints in the directory
	CHECKPOINTING_ARGS=(
		--save $CKPT_DIR
		--save-interval $CHECKPOINT_STEPS
		--ckpt-format torch_dist
		--async-save
	)
fi

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--expert-tensor-parallel-size $ETP
	--expert-model-parallel-size $EP
	--context-parallel-size $CP
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
	--sequence-parallel
	# --tp-comm-overlap  # Requires TE > 2.8
	# --num-layers-per-virtual-pipeline-stage 4
	# --virtual-pipeline-model-parallel-size 2
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model swiss-ai/Apertus-8B-2509
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--reset-attention-mask  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 4
	--num-dataset-builder-threads 1
	# --goldfish-loss  # goldfish
	# --goldfish-k 50  # goldfish
	# --goldfish-h 50  # goldfish
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $DATASETS --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"


# Hugging Face Token
# export HF_TOKEN=''
export TRANSFORMERS_NO_SLOW_TOKENIZER=1 
# WANDB Logging
export WANDB_API_KEY=''

if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

mkdir -p "$NSYS_DIR"

if [ "$NSYS_PROFILER" = true ]; then
  NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' \
    --output=${NSYS_DIR}/nsys-${SLURM_JOB_ID}-rank${SLURM_PROCID} \
    --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
  TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi


# Save sbatch script
cp $0 $DEBUG_DIR

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Checkpoint Compute Environment
echo "Current Path: ${PWD}"
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR 
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(python -m pip list 2>/dev/null || python3 -m pip list 2>/dev/null || echo 'pip not available')" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 

# before you call python / srun
export LOCAL_CACHE_BASE=${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${LOCAL_CACHE_BASE}/triton_cache/${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_CACHE_BASE}/inductor_cache/${SLURM_PROCID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SRUN_ARGS=" \
	-lu \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--wait 60 \
	--jobid $SLURM_JOB_ID \
	--kill-on-bad-exit 1 \
	"

# srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task $SLURM_CPUS_PER_TASK --mpi=pmix \
	--distribution=block:block --reservation=PA-2338-RL \
	--network=disable_rdzv_get \
    --environment=$IMAGE_ENV \
	-lu bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi