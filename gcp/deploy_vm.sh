#!/bin/bash
# GCP VM Deployment Script for Deep Learning RNA-Protein Binding Project
# This script creates a Compute Engine VM with GPU support optimized for deep learning

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
VM_NAME="${VM_NAME:-rna-protein-dl-vm}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50GB}"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "🚀 Creating GCP VM for RNA-Protein Binding Deep Learning"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "VM Name: $VM_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_TYPE (x$GPU_COUNT)"
echo "Boot Disk: $BOOT_DISK_SIZE"
echo "=================================================="

# Create the VM instance
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=default \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=$VM_NAME,image=projects/$IMAGE_PROJECT/global/images/family/$IMAGE_FAMILY,mode=rw,size=$BOOT_DISK_SIZE,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-standard \
    --metadata-from-file startup-script=startup_script.sh \
    --enable-display-device \
    --tags=deeplearning,rna-protein \
    --preemptible

echo "✅ VM '$VM_NAME' created successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Wait for VM to finish setup (check startup script logs)"
echo "2. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "3. Upload your project code:"
echo "   gcloud compute scp --recurse . $VM_NAME:~/Deep-Learning-in-Computational-Biology --zone=$ZONE --project=$PROJECT_ID"
echo "4. Run the training script on the VM"
echo ""
echo "💰 Cost optimization tips:"
echo "- This VM uses preemptible pricing (up to 80% cheaper)"
echo "- Stop the VM when not in use: gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "- Consider using spot instances for even more savings"
echo ""
echo "🔍 Monitor your VM:"
echo "- Check startup logs: gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE"
echo "- SSH access: gcloud compute ssh $VM_NAME --zone=$ZONE"

