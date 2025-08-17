#!/bin/bash
# Script to upload the project to GCP VM
# Run this from your local machine after creating the VM

set -e

# Configuration
VM_NAME="${VM_NAME:-rna-protein-dl-vm}"
ZONE="${ZONE:-us-central1-a}"
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"

echo "📤 Uploading RNA-Protein Binding Project to GCP VM"
echo "=================================================="
echo "VM: $VM_NAME"
echo "Zone: $ZONE"
echo "Project: $PROJECT_ID"
echo ""

# Check if we're in the project directory
if [ ! -f "phase2_fast.py" ]; then
    echo "❌ Error: phase2_fast.py not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if VM exists and is running
echo "🔍 Checking VM status..."
VM_STATUS=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='value(status)' 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" = "NOT_FOUND" ]; then
    echo "❌ VM '$VM_NAME' not found in zone '$ZONE'"
    echo "💡 Create VM first with: ./gcp/deploy_vm.sh"
    exit 1
elif [ "$VM_STATUS" != "RUNNING" ]; then
    echo "⚠️ VM is not running (status: $VM_STATUS)"
    echo "🚀 Starting VM..."
    gcloud compute instances start $VM_NAME --zone=$ZONE
    echo "⏳ Waiting for VM to start..."
    sleep 30
fi

# Upload the project
echo "📁 Uploading project files..."
gcloud compute scp --recurse . $VM_NAME:~/Deep-Learning-in-Computational-Biology --zone=$ZONE --project=$PROJECT_ID

# Make scripts executable on the VM
echo "🔧 Setting up permissions..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="
    cd ~/Deep-Learning-in-Computational-Biology
    chmod +x gcp/*.sh
    chmod +x scripts/*.py 2>/dev/null || true
    echo '✅ Project uploaded successfully!'
"

echo ""
echo "🎉 Upload completed!"
echo ""
echo "📝 Next steps:"
echo "1. SSH into VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "2. Start training:"
echo "   cd ~/Deep-Learning-in-Computational-Biology"
echo "   ./gcp/run_training.sh"
echo ""
echo "3. Monitor progress:"
echo "   ./gcp/monitor_training.sh"
echo ""
echo "💰 Remember to stop the VM when done:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"

