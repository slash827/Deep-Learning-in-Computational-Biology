#!/bin/bash
# GCP VM Startup Script for RNA-Protein Binding Deep Learning Project
# This script sets up the environment and dependencies on VM boot

set -e

# Log everything
exec > >(tee /var/log/startup-script.log)
exec 2>&1

echo "🚀 Starting VM setup for RNA-Protein Binding Deep Learning..."
echo "Timestamp: $(date)"

# Update system
echo "📦 Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install essential tools
echo "🔧 Installing essential tools..."
apt-get install -y \
    git \
    wget \
    curl \
    htop \
    tmux \
    vim \
    unzip \
    tree \
    build-essential

# Verify NVIDIA drivers and CUDA
echo "🖥️ Verifying GPU setup..."
nvidia-smi || {
    echo "❌ NVIDIA drivers not found. Installing..."
    # This should already be installed on DL images, but just in case
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    apt-get update
    apt-get -y install cuda-drivers
}

# Verify PyTorch installation and GPU access
echo "🔥 Verifying PyTorch GPU support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available!')
"

# Create project directory
echo "📁 Setting up project directory..."
mkdir -p /home/rna_protein_project
cd /home/rna_protein_project

# Set permissions for the default user
chown -R ${USER:-jupyter}:${USER:-jupyter} /home/rna_protein_project

# Install additional Python packages that might be needed
echo "📦 Installing additional Python packages..."
pip3 install --upgrade pip
pip3 install \
    torch \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    wandb \
    transformers \
    scipy \
    jupyter \
    ipywidgets

# Create helpful aliases and environment setup
echo "⚙️ Setting up environment..."
cat << 'EOF' >> /home/${USER:-jupyter}/.bashrc

# RNA-Protein Deep Learning Project aliases
alias ll='ls -la'
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias activate_project='cd /home/rna_protein_project && source venv/bin/activate 2>/dev/null || echo "Virtual env not found"'

# Set project directory
export PROJECT_DIR="/home/rna_protein_project"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Auto-activate project environment
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
fi

EOF

# Create a virtual environment for the project
echo "🐍 Setting up Python virtual environment..."
cd /home/rna_protein_project
python3 -m venv venv
source venv/bin/activate

# Install requirements in virtual environment
pip install --upgrade pip
pip install \
    torch \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    wandb \
    transformers \
    scipy

# Create welcome script
cat << 'EOF' > /home/rna_protein_project/welcome.sh
#!/bin/bash
echo "🧬 Welcome to RNA-Protein Binding Deep Learning VM!"
echo "=================================================="
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python3 --version)"
echo "🔥 PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "🖥️ GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -1
echo ""
echo "📝 Quick commands:"
echo "  gpu          - Show GPU status"
echo "  gpuwatch     - Monitor GPU usage"
echo "  ll           - List files"
echo ""
echo "🚀 To start training:"
echo "  1. Upload your project code to this directory"
echo "  2. Activate virtual environment: source venv/bin/activate"
echo "  3. Run your training script"
echo ""
echo "💡 Remember to stop the VM when done to save costs!"
echo "=================================================="
EOF

chmod +x /home/rna_protein_project/welcome.sh

# Set ownership
chown -R ${USER:-jupyter}:${USER:-jupyter} /home/rna_protein_project

# Create startup completion marker
touch /var/log/startup-complete

echo "✅ VM setup completed successfully!"
echo "Timestamp: $(date)"
echo "🎉 The VM is ready for RNA-Protein Binding Deep Learning!"

# Run welcome script
/home/rna_protein_project/welcome.sh

