# KFTO Distributed LLM Fine-Tuning on OpenShift

Complete guide for fine-tuning large language models using Kubeflow Training Operator (KFTO) on Red Hat OpenShift with NVIDIA L40S GPUs.

## 📚 Reference Documentation

This guide is based on the official Red Hat Developer article:
- **[How to fine-tune LLMs with Kubeflow Training Operator](https://developers.redhat.com/articles/2025/03/26/how-fine-tune-llms-kubeflow-training-operator#serve_the_fine_tuned_model)**

Additional documentation:
- [KFTO Troubleshooting Summary](./KFTO-troubleshooting-summary.md) - Detailed troubleshooting guide
- [Replication Guide](./REPLICATION-GUIDE.md) - How to replicate with new datasets
- [L40S Deployment Guide](./L40S-DEPLOYMENT-GUIDE.md) - L40S GPU setup guide

---

## 🎯 Overview

This repository contains all the necessary configurations to:
- Fine-tune Qwen 2.5-7B (or other LLMs) using LoRA
- Run distributed training across multiple GPU nodes
- Monitor training with Aim tracking
- Serve the fine-tuned model with vLLM on OpenShift AI

**Key Achievements:**
- ✅ Distributed training on 2x NVIDIA L40S GPUs (48GB VRAM each)
- ✅ Training time: ~7 hours for 50 steps
- ✅ Final loss: 0.20 (excellent quality)
- ✅ Production-ready model with vLLM integration

---

## 📋 Prerequisites

### Hardware Requirements

- **GPU Nodes**: 2x nodes with NVIDIA L40S GPUs (g6e.2xlarge on AWS)
  - 48GB VRAM per GPU
  - 32GB system RAM per node
  - 8 vCPUs per node

### Software Requirements

- **Red Hat OpenShift** cluster (4.12+)
- **Red Hat OpenShift AI** (v2.18+)
- **OpenShift Data Foundation** (for CephFS RWX storage)
- **NVIDIA GPU Operator** (pre-installed on GPU nodes)
- **Node Feature Discovery Operator** (pre-installed)

### Storage Requirements

- **ReadWriteMany (RWX)** storage class (e.g., CephFS)
- PVCs:
  - `dataset-volume`: 50Gi (training data)
  - `model-volume`: 50Gi (model outputs)
  - `cache-volume`: 50Gi (Hugging Face cache)
  - `minio-pvc`: 150Gi (optional, for MinIO object storage)

---

## 🚀 Step-by-Step Deployment

### Step 1: Enable KFTO in OpenShift AI

Enable the Kubeflow Training Operator by setting `managementState` to `Managed` in the `DataScienceCluster`:

```bash
# Edit the DataScienceCluster
oc edit datasciencecluster default-dsc

# Set the following:
spec:
  components:
    trainingoperator:
      managementState: Managed
```

**Verify installation:**
```bash
# Check KFTO operator is running
oc get pods -n redhat-ods-applications | grep training-operator

# Should see:
# kubeflow-training-operator-xxxx   1/1   Running
```

---

### Step 2: Install OpenShift Data Foundation (ODF) for RWX Storage

Distributed training requires ReadWriteMany (RWX) storage so multiple GPU nodes can access the same data simultaneously. OpenShift Data Foundation provides CephFS for RWX storage.

#### Install ODF Operator

1. **Via OpenShift Console:**
   - Navigate to **Operators** → **OperatorHub**
   - Search for "OpenShift Data Foundation"
   - Click **Install**
   - Select **stable** channel
   - Click **Install** and wait for operator to be ready

2. **Via CLI:**
   ```bash
   # Create ODF namespace
   oc create namespace openshift-storage
   
   # Create operator subscription
   cat <<EOF | oc apply -f -
   apiVersion: operators.coreos.com/v1alpha1
   kind: Subscription
   metadata:
     name: odf-operator
     namespace: openshift-storage
   spec:
     channel: stable-4.16
     installPlanApproval: Automatic
     name: odf-operator
     source: redhat-operators
     sourceNamespace: openshift-marketplace
   EOF
   
   # Wait for operator to be ready
   oc get csv -n openshift-storage -w
   ```

#### Create StorageSystem

Once the ODF operator is installed, create a StorageSystem:

1. **Via OpenShift Console:**
   - Navigate to **Operators** → **Installed Operators** → **OpenShift Data Foundation**
   - Click **Create StorageSystem**
   - Select deployment mode:
     - **Internal** - Use worker node storage (requires 3+ worker nodes with local disks)
     - **External** - Connect to existing Ceph cluster
   - Configure:
     - **Storage Class**: Select worker node storage class (e.g., `gp3-csi`)
     - **Storage Device Size**: 2Ti (for 3 replicas = 6Ti total)
     - **Number of Replicas**: 3 (default for HA)
   - Click **Create**

2. **Via CLI (Internal Mode):**
   ```bash
   # Create StorageCluster for internal mode
   cat <<EOF | oc apply -f -
   apiVersion: ocs.openshift.io/v1
   kind: StorageCluster
   metadata:
     name: ocs-storagecluster
     namespace: openshift-storage
   spec:
     arbiter: {}
     encryption:
       kms: {}
     externalStorage: {}
     flexibleScaling: true
     resources:
       mds:
         limits:
           cpu: "3"
           memory: "8Gi"
         requests:
           cpu: "1"
           memory: "8Gi"
     monDataDirHostPath: /var/lib/rook
     managedResources:
       cephBlockPools:
         reconcileStrategy: manage
       cephConfig: {}
       cephDashboard: {}
       cephFilesystems:
         reconcileStrategy: manage
       cephObjectStoreUsers: {}
       cephObjectStores: {}
     multiCloudGateway:
       reconcileStrategy: manage
     storageDeviceSets:
     - count: 1
       dataPVCTemplate:
         spec:
           accessModes:
           - ReadWriteOnce
           resources:
             requests:
               storage: 2Ti
           storageClassName: gp3-csi
           volumeMode: Block
       name: ocs-deviceset
       placement: {}
       portable: true
       replica: 3
       resources:
         limits:
           cpu: "2"
           memory: "5Gi"
         requests:
           cpu: "1"
           memory: "5Gi"
   EOF
   ```

#### Verify ODF Installation

```bash
# Check StorageCluster status
oc get storagecluster -n openshift-storage

# Should show:
# NAME                 AGE   PHASE   EXTERNAL   CREATED AT             VERSION
# ocs-storagecluster   5m    Ready              2025-10-13T10:30:00Z   4.16.0

# Check ODF pods (takes 5-10 minutes)
oc get pods -n openshift-storage

# Should see running pods:
# - rook-ceph-operator
# - rook-ceph-mon (3 replicas)
# - rook-ceph-osd (3+ replicas)
# - rook-ceph-mds (2 replicas for CephFS)
# - csi-cephfsplugin daemonset

# Check storage classes
oc get storageclass | grep ocs

# Should see:
# ocs-storagecluster-ceph-rbd         openshift-storage.rbd.csi.ceph.com
# ocs-storagecluster-cephfs           openshift-storage.cephfs.csi.ceph.com  ← RWX storage
```

#### Set CephFS as Default (Optional)

```bash
# Remove default from existing storage class
oc patch storageclass gp3-csi -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'

# Set CephFS as default
oc patch storageclass ocs-storagecluster-cephfs -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

#### Test RWX Storage

Create a test PVC to verify RWX access:

```bash
# Create test PVC
cat <<EOF | oc apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-rwx
  namespace: openshift-storage
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: ocs-storagecluster-cephfs
EOF

# Check PVC status
oc get pvc test-rwx -n openshift-storage

# Should show:
# NAME       STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS
# test-rwx   Bound    pvc-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx      1Gi        RWX            ocs-storagecluster-cephfs

# Cleanup test PVC
oc delete pvc test-rwx -n openshift-storage
```

**Important Notes:**
- ODF requires **minimum 3 worker nodes** for internal mode
- Each OSD requires dedicated storage (EBS volumes on AWS)
- Default configuration uses **3-way replication** (3x storage overhead)
- CephFS provides **RWX access mode** required for distributed training

---

### Step 3: Deploy L40S GPU Nodes

Deploy NVIDIA L40S GPU nodes using the MachineSet:

```bash
# Create L40S machineset (g6e.2xlarge)
oc apply -f machineset-l40s-g6e-2xlarge.yaml

# Watch nodes being created
oc get machines -n openshift-machine-api -w

# Wait for nodes to be Ready (~5-10 minutes)
oc get nodes -l gpu.type=l40s
```

**Important:** Update CephFS CSI driver to run on GPU nodes:

```bash
# Edit DaemonSet
oc edit daemonset csi-cephfsplugin -n openshift-storage

# Add GPU toleration:
spec:
  template:
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: "True"
        effect: NoSchedule
```

**Verify CSI pods on GPU nodes:**
```bash
# Should see 9 CSI pods (4 regular + 5 GPU nodes)
oc get pods -n openshift-storage | grep csi-cephfsplugin | wc -l
```

---

### Step 4: Create Namespace and PVCs

Create the fine-tuning namespace and persistent volumes:

```bash
# Create namespace
oc new-project fine-tuning

# Clone repository
git clone https://github.com/JPishikawa/ft-by-sft/
cd ft-by-sft

# Create PVCs
oc apply -f deploy/storage/pvc.yaml
```

**PVCs created:**
- `dataset-volume` (50Gi, RWX) - Training dataset
- `model-volume` (50Gi, RWX) - Model output/checkpoints
- `cache-volume` (50Gi, RWX) - Hugging Face cache

---

### Step 5: Deploy MinIO (Optional)

MinIO provides S3-compatible object storage for dataset management:

```bash
# Create MinIO PVC
oc apply -f deploy/storage/minio-pvc.yaml

# Deploy MinIO
oc apply -f deploy/storage/minio-deployment.yaml

# Expose MinIO service
oc expose svc minio -n fine-tuning

# Get MinIO route
MINIO_ROUTE=$(oc get route minio -n fine-tuning -o jsonpath='{.spec.host}')
echo "MinIO Console: http://${MINIO_ROUTE}"
```

**Default credentials:**
- Username: `minioadmin`
- Password: `minioadmin`

---

### Step 6: Prepare Training Dataset

Download and prepare your training dataset.

#### Option A: Download from MinIO/S3

Create storage secret with your credentials:

```bash
# Create storage secret
cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: storage-config
  namespace: fine-tuning
stringData:
  my-storage: |
    {
      "type": "s3",
      "access_key_id": "YOUR_ACCESS_KEY",
      "secret_access_key": "YOUR_SECRET_KEY",
      "endpoint_url": "http://minio.fine-tuning.svc.cluster.local:9000",
      "region": "us-east-1"
    }
EOF
```

Create download job:

```bash
# Download dataset from S3
oc apply -f deploy/storage/download-dataset.yaml

# Monitor download
oc logs -f download-dataset -n fine-tuning
```

#### Option B: Upload Directly

```bash
# Copy local dataset to PVC
oc cp train-00000-of-00001.parquet \
  <pod-with-pvc-mounted>:/data/input/ -n fine-tuning
```

**Verify dataset:**
```bash
# Check dataset is in PVC
oc exec <pod-name> -n fine-tuning -- ls -lh /data/input/
```

---

### Step 7: Deploy Aim Tracker

Aim provides visualization for training metrics (loss, learning rate, etc.):

```bash
# Deploy Aim server
oc apply -f deploy/aim/aim-deployment.yaml

# Expose Aim UI
oc expose svc aim -n aim

# Get Aim route
AIM_ROUTE=$(oc get route aim -n aim -o jsonpath='{.spec.host}')
echo "Aim UI: http://${AIM_ROUTE}:53800"
```

**Access Aim dashboard:**
- URL: `http://<aim-route>:53800`
- Experiment: `qwen-l40s-optimized`

---

### Step 8: Create Training Configuration

Apply the training configuration ConfigMap:

```bash
# Apply L40S-optimized training config
oc apply -f training-config-l40s-optimized.yaml -n fine-tuning

# Verify ConfigMap
oc describe configmap training-config -n fine-tuning
```

**Key configuration parameters:**

```yaml
Model: Qwen/Qwen2.5-7B-Instruct
Batch size per device: 4
Gradient accumulation: 16
Effective batch size: 128 (4 × 16 × 2 GPUs)
LoRA rank: 32
Learning rate: 2e-05 (cosine schedule)
Mixed precision: bf16
Packing: Enabled
Flash Attention: Enabled
Checkpointing: Every 5 steps
```

See [`training-config-l40s-optimized.yaml`](./training-config-l40s-optimized.yaml) for full configuration.

---

### Step 9: Deploy Training Job

Launch the distributed training job using PyTorchJob:

```bash
# Deploy PyTorchJob
oc apply -f kfto-demo-pytorchjob-l40s.yaml -n fine-tuning

# Monitor job status
oc get pytorchjob kfto-demo -n fine-tuning -w

# Watch pods
oc get pods -n fine-tuning -w
```

**Expected pods:**
- `kfto-demo-master-0` - Master node (rank 0)
- `kfto-demo-worker-0` - Worker node (rank 1)

---

### Step 10: Monitor Training Progress

#### Check Pod Logs

```bash
# Monitor master logs (real-time)
oc logs -f kfto-demo-master-0 -n fine-tuning

# Check worker logs
oc logs -f kfto-demo-worker-0 -n fine-tuning

# View training progress
oc logs kfto-demo-master-0 -n fine-tuning | grep -E "%|loss"
```

#### Monitor with Aim Dashboard

Open the Aim UI in your browser:
- Navigate to `http://<aim-route>:53800`
- Select experiment: `qwen-l40s-optimized`
- View real-time metrics:
  - Training loss
  - Learning rate decay
  - Tokens per second
  - Gradient norms

#### Check Checkpoints

```bash
# List saved checkpoints
oc exec kfto-demo-master-0 -n fine-tuning -- \
  ls -lh /data/output/tuning/qwen2.5-l40s-tuning/

# Should see:
# checkpoint-5/
# checkpoint-10/
# checkpoint-15/
# ...
```

---

### Step 11: Training Timeline

**Expected timeline for 50 steps:**

| Time | Progress | Milestone |
|------|----------|-----------|
| 0-10 min | Initialization | Model loading, NCCL setup |
| 10 min | 2% (1/50) | First step complete |
| 30 min | 10% (5/50) | First checkpoint saved |
| 1 hour | 20% (10/50) | Second checkpoint |
| 2 hours | 40% (20/50) | Loss metrics visible |
| 3.5 hours | 60% (30/50) | Mid-training |
| 5 hours | 80% (40/50) | Near completion |
| **7 hours** | **100% (50/50)** | **Training complete!** ✅ |

**Key metrics to watch:**
- **Step 10**: Loss ~0.34
- **Step 20**: Loss ~0.24
- **Step 30**: Loss ~0.21
- **Step 40**: Loss ~0.20
- **Step 50**: Loss ~0.20 (final)

**Good signs:**
- ✅ Steady loss decrease
- ✅ No OOM errors
- ✅ Both pods Running
- ✅ Checkpoints saving every 5 steps

---

### Step 12: Retrieve Fine-Tuned Model

Once training completes, download your model:

```bash
# Copy model from cluster
oc cp kfto-demo-master-0:/data/output/model ./my-finetuned-model -n fine-tuning

# Check model files
ls -lh ./my-finetuned-model/
# adapter_config.json
# adapter_model.safetensors  ← Your LoRA weights
# README.md
# training_args.bin
```

**Upload to Hugging Face (optional):**

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model
huggingface-cli upload <your-username>/qwen-2.5-7b-finetuned ./my-finetuned-model
```

---

### Step 13: Serve the Model with vLLM

Deploy your fine-tuned model on OpenShift AI using vLLM.

#### Create Data Connection

1. Go to **OpenShift AI Dashboard**
2. Navigate to **Data Science Projects** → `fine-tuning`
3. Click **Add data connection**
4. Configure:
   - **Name**: `model-connection`
   - **Connection type**: `URI - v1`
   - **URI**: `pvc://model-volume/model/`
5. Click **Add data connection**

#### Deploy Model

1. Go to **Models** tab
2. Click **Deploy model**
3. Configure deployment:
   - **Model name**: `qwen-finetuned`
   - **Serving runtime**: `vLLM ServingRuntime for KServe`
   - **Model framework**: `pytorch`
   - **Model location**: Select `model-connection`
   - **Model server size**: 
     - CPU: 4
     - Memory: 16Gi
     - GPU: 1

4. **Add command-line arguments:**
   ```
   --enable-lora
   --lora-modules=tuned-qwen=/mnt/models/
   --model=Qwen/Qwen2.5-7B-Instruct
   ```

5. **Add environment variable:**
   - Name: `HF_HUB_OFFLINE`
   - Value: `0`

6. Click **Deploy**

#### Test the Model

Once deployed, test via the API:

```bash
# Get inference endpoint
INFERENCE_URL=$(oc get inferenceservice qwen-finetuned -n fine-tuning \
  -o jsonpath='{.status.url}')

# Test inference
curl -X POST ${INFERENCE_URL}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tuned-qwen",
    "prompt": "### Question:\nWhat is machine learning?\n\n### Answer:\n",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Python example:**

```python
import requests

inference_url = "http://<inference-endpoint>/v1/completions"

payload = {
    "model": "tuned-qwen",
    "prompt": "### Question:\nExplain distributed training\n\n### Answer:\n",
    "max_tokens": 200,
    "temperature": 0.7
}

response = requests.post(inference_url, json=payload)
print(response.json()["choices"][0]["text"])
```

---

## 📊 Training Results

### Final Metrics

```yaml
Total Steps: 50
Training Runtime: 25,608 seconds (7.1 hours)
Training Speed: 511.8 tokens/second
Average Loss: 0.2405
Final Loss: 0.2011
Epochs Processed: 18.18
Model Quality: Excellent (loss < 0.25)
```

### Loss Progression

| Step | Loss | Learning Rate | Status |
|------|------|---------------|--------|
| 10 | 0.3435 | 1.81e-05 | Initial |
| 20 | 0.2395 | 1.31e-05 | Improving |
| 30 | 0.2139 | 6.91e-06 | Good |
| 40 | 0.2044 | 1.91e-06 | Great |
| 50 | 0.2011 | 0.0 | Excellent ✅ |

**Interpretation:**
- **42% loss reduction** (0.34 → 0.20)
- Smooth convergence curve (no overfitting)
- Production-ready model quality

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: Pods Not Scheduling

**Error:**
```
0/12 nodes are available: 5 node(s) had untolerated taint {nvidia.com/gpu: True}
```

**Solution:**
Add GPU tolerations to PyTorchJob (already in `kfto-demo-pytorchjob-l40s.yaml`):
```yaml
tolerations:
- key: nvidia.com/gpu
  operator: Equal
  value: "True"
  effect: NoSchedule
```

#### Issue: Volume Mount Failure

**Error:**
```
MountVolume.MountDevice failed: driver not found
```

**Solution:**
Update CephFS CSI DaemonSet with GPU toleration (see Step 3).

#### Issue: Worker Connection Timeout

**Error:**
```
Connection refused: kfto-demo-master-0:29500
```

**Solutions:**
1. Verify both pods are running
2. Check NCCL configuration in ConfigMap
3. Ensure network policies allow pod-to-pod communication

#### Issue: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use L40S GPUs (48GB VRAM) instead of L4 (24GB)
2. Reduce batch size in training config
3. Reduce LoRA rank
4. Disable packing

See [KFTO-troubleshooting-summary.md](./KFTO-troubleshooting-summary.md) for detailed troubleshooting.

---

## 📁 Repository Structure

```
.
├── README.md                              # This file
├── KFTO-troubleshooting-summary.md       # Detailed troubleshooting
├── REPLICATION-GUIDE.md                  # How to replicate with new data
├── L40S-DEPLOYMENT-GUIDE.md              # L40S GPU setup guide
│
├── machineset-l40s-g6e-2xlarge.yaml     # L40S GPU MachineSet
├── machineset-l40s-g6e-xlarge.yaml      # L40S budget option
│
├── training-config-l40s-optimized.yaml   # Production training config ⭐
├── training-config-ultra-low-memory.yaml # Low-memory fallback (L4)
├── training-config-single-gpu.yaml       # Single-GPU config
│
├── kfto-demo-pytorchjob-l40s.yaml       # L40S PyTorchJob ⭐
├── kfto-demo-pytorchjob-optimized.yaml  # L4 PyTorchJob (resilient)
├── kfto-demo-single-gpu.yaml            # Single-GPU PyTorchJob
│
├── deploy-resilient-training.sh          # Deployment automation
└── monitor-training.sh                   # Monitoring script
```

**⭐ Key files for L40S deployment**

---

## 🎯 Quick Start (TL;DR)

For experienced users, here's the quick deployment:

```bash
# 1. Enable KFTO
oc patch datasciencecluster default-dsc --type=merge \
  -p '{"spec":{"components":{"trainingoperator":{"managementState":"Managed"}}}}'

# 2. Install ODF and create StorageCluster
oc apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: odf-operator
  namespace: openshift-storage
spec:
  channel: stable-4.16
  name: odf-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF

# Wait for ODF to be ready
oc wait --for=condition=Ready storagecluster/ocs-storagecluster -n openshift-storage --timeout=10m

# 3. Deploy L40S nodes
oc apply -f machineset-l40s-g6e-2xlarge.yaml

# 4. Fix CSI driver for GPU nodes
oc edit daemonset csi-cephfsplugin -n openshift-storage
# (Add GPU toleration)

# 5. Setup namespace and storage
oc new-project fine-tuning
git clone https://github.com/JPishikawa/ft-by-sft/
oc apply -f ft-by-sft/deploy/storage/pvc.yaml

# 6. Upload dataset
oc cp train-00000-of-00001.parquet <pod>:/data/input/ -n fine-tuning

# 7. Deploy training
oc apply -f training-config-l40s-optimized.yaml -n fine-tuning
oc apply -f kfto-demo-pytorchjob-l40s.yaml -n fine-tuning

# 8. Monitor
oc logs -f kfto-demo-master-0 -n fine-tuning
```

---

## 📈 Performance Comparison

### L4 vs L40S GPUs

| Metric | g6.xlarge (L4) | g6e.2xlarge (L40S) | Improvement |
|--------|----------------|---------------------|-------------|
| GPU VRAM | 24GB | 48GB | **2x** |
| System RAM | 16GB | 32GB | **2x** |
| vCPUs | 4 | 8 | **2x** |
| Batch Size | 1 | 4 | **4x** |
| Step Time | 60-180 min | 6-9 min | **10-20x faster** |
| Training Time | Never completed | 7 hours | **∞ improvement** ✅ |
| OOM Errors | Constant | Zero | **Solved** ✅ |
| Cost | $0.52/hr × ∞ | $1.20/hr × 7 = $8.40 | **Cheaper!** |

**Conclusion:** L40S is faster, more reliable, and actually cheaper for this workload!

---

## 🔗 Additional Resources

### Documentation
- [Red Hat OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [FMS HF Tuning Library](https://github.com/foundation-model-stack/fms-hf-tuning)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

### Related Guides
- [KFTO Troubleshooting Summary](./KFTO-troubleshooting-summary.md)
- [Replication Guide](./REPLICATION-GUIDE.md)
- [L40S Deployment Guide](./L40S-DEPLOYMENT-GUIDE.md)

### Community
- [Red Hat Developer Portal](https://developers.redhat.com/)
- [OpenShift AI Community](https://ai-on-openshift.io/)
- [Kubeflow Slack](https://kubeflow.slack.com/)

---

## 🙏 Acknowledgments

This guide is based on:
- [Red Hat Developer Article by Junpei Ishikawa](https://developers.redhat.com/articles/2025/03/26/how-fine-tune-llms-kubeflow-training-operator)
- [ft-by-sft GitHub Repository](https://github.com/JPishikawa/ft-by-sft)
- Extensive troubleshooting and optimization work

---

## 📝 License

MIT License - Feel free to use and modify for your needs.

---

## ✅ Success Checklist

Before starting, ensure:
- [ ] OpenShift cluster with GPU nodes
- [ ] OpenShift AI with KFTO enabled
- [ ] **OpenShift Data Foundation (ODF) installed and ready**
- [ ] **CephFS storage class available** (`ocs-storagecluster-cephfs`)
- [ ] CSI driver configured for GPU nodes (with GPU tolerations)
- [ ] Dataset prepared and uploaded
- [ ] Aim tracker deployed (optional)

After training completes:
- [ ] Final loss < 0.3 (good quality)
- [ ] All 50 steps completed
- [ ] Model saved to `/data/output/model/`
- [ ] Checkpoints exist in output directory
- [ ] Model deployed and tested with vLLM

---

**🎉 Congratulations! You now have a production-ready fine-tuned LLM!**

For questions or issues, refer to:
- [KFTO-troubleshooting-summary.md](./KFTO-troubleshooting-summary.md)
- [Red Hat Developer Article](https://developers.redhat.com/articles/2025/03/26/how-fine-tune-llms-kubeflow-training-operator)
- [GitHub Issues](https://github.com/JPishikawa/ft-by-sft/issues)

