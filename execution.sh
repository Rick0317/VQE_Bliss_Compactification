#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:30:00
#SBATCH --job-name=adapt_vqe_array
#SBATCH --output=adapt_vqe_array_%A_%a.out
#SBATCH --error=adapt_vqe_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca
#SBATCH --array=0-5%2  # Run 6 jobs (indices 0-5), max 2 at a time

# Load required modules
module load python
module load conda3
source ~/adapt-vqe-env/bin/activate

# Set up matplotlib config directory
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
mkdir -p $MPLCONFIGDIR

# Set memory and CPU optimizations for our ADAPT-VQE implementation
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Define parameter combinations for different jobs
# Format: "mol_file mol n_qubits n_electrons pool_type"
declare -a JOB_PARAMS=(
    "h4_sto-3g.pkl h4 8 4 uccsd"
    "lih_fer.bin lih 12 4 uccsd"
    "beh2_fer.bin beh2 14 4 uccsd"
    "h4_sto-3g.pkl h4 8 4 qubit_pool"
    "lih_fer.bin lih 12 4 qubit_pool"
    "beh2_fer.bin beh2 14 4 qubit_pool"
)

# Get parameters for this array job
if [ ${SLURM_ARRAY_TASK_ID} -lt ${#JOB_PARAMS[@]} ]; then
    PARAMS=${JOB_PARAMS[${SLURM_ARRAY_TASK_ID}]}
    read -r mol_file mol n_qubits n_electrons pool_type <<< "$PARAMS"
else
    echo "ERROR: Invalid array task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Print job information
echo "SLURM array job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo ""
echo "Running with parameters:"
echo "  Molecule file: $mol_file"
echo "  Molecule: $mol"
echo "  Qubits: $n_qubits"
echo "  Electrons: $n_electrons"
echo "  Pool type: $pool_type"
echo ""

echo "Testing environment..."
which python
python --version
python -c "import numpy as np; import qiskit; print('NumPy version:', np.__version__); print('Qiskit version:', qiskit.__version__)"

# Print available memory and CPU info
echo "Available memory:"
free -h
echo "CPU info:"
lscpu | grep "CPU(s):"

# Change to the root directory of the project
cd $SLURM_SUBMIT_DIR

# Run the ADAPT-VQE qubitwise script with specific parameters
echo "Starting ADAPT-VQE qubitwise optimization for $mol..."
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Create unique log filename for this array job
LOG_FILE="adapt_vqe_${mol}_${pool_type}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# Run with memory monitoring and specific parameters
python -u adapt_vqe_qiskit_qubitwise_bai.py "$mol_file" "$mol" "$n_qubits" "$n_electrons" "$pool_type" 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "ADAPT-VQE for $mol completed successfully at $(date)"
else
    echo "ADAPT-VQE for $mol failed with exit code $? at $(date)"
fi

# Print final memory usage
echo "Final memory usage:"
free -h

# Copy results to a timestamped directory specific to this job
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${mol}_${pool_type}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $RESULTS_DIR

# Copy output files - look for files with this specific molecule/pool combination
echo "Copying results for $mol with $pool_type pool..."
cp *${mol}*${pool_type}*.csv $RESULTS_DIR/ 2>/dev/null || echo "No CSV files found for $mol-$pool_type"
cp *${mol}*.json $RESULTS_DIR/ 2>/dev/null || echo "No JSON cache files found for $mol"
cp "$LOG_FILE" $RESULTS_DIR/ 2>/dev/null || echo "No log file found"

# Also copy any generic output files
cp *.csv $RESULTS_DIR/ 2>/dev/null || echo "No additional CSV files in root directory"
cp *.json $RESULTS_DIR/ 2>/dev/null || echo "No additional JSON cache files in root directory"

echo "Results copied to: $RESULTS_DIR"
echo "SLURM array job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
