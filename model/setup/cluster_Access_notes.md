# BMU Cluster Access — Status, Open Questions & Commands

Target location in the repo: `model/setup/cluster_access_notes.md`. Written for the team and for the conversation with IT/lab admins — everything below is verified from actual command output on `bmu-headnode`, not assumed.

---

## 1. What We Have Confirmed

- **This is not a single DGX box you SSH into and train on directly.** `bmu-headnode` is a login/head node only — it has no GPU (`nvidia-smi` isn't even installed there, and can't be installed without root). Actual compute happens on a separate worker node, scheduled through Kubernetes.
- **Cluster type:** NVIDIA Base Command Manager (Bright Cluster Manager) running Kubernetes v1.32.11, on Ubuntu 24.04.
- **Nodes:** `bmu-master` (control-plane) and `bmu-worker` (the actual GPU-carrying node).
- **Access:** SSH with username + password (no SSH key), one account per team member, format `dgx-s-bmu-cse-<id>@10.1.0.176`.
- **Each user gets a personal, restricted Kubernetes namespace** — ours is `dgx-s-bmu-cse-240574-restricted`. We do not have cluster-wide visibility (can't list all namespaces or all pods) — this is expected/normal for a restricted account, not a bug.
- **GPU is exposed as MIG (Multi-Instance GPU) slices, not whole GPUs:**

  | Resource | Available on `bmu-worker` |
  |---|---|
  | `nvidia.com/gpu` (whole GPU) | 0 |
  | `nvidia.com/mig-1g.18gb` | 21 |
  | `nvidia.com/mig-2g.35gb` | 9 |
  | `nvidia.com/mig-3g.71gb` | 2 |
  | `nvidia.com/mig-4g.71gb` | 0 |

  The 18/35/71 GB slice sizes match NVIDIA's standard MIG profiles for a 141GB-class card — consistent with H200. We request a MIG slice as a resource in a pod spec instead of a whole GPU.
- **`bmu-worker` also has 224 CPU cores and ~2TB RAM** available at the node level (shared across everyone's scheduled pods, not ours alone).
- **We have permission, in our own namespace, to create:** Jobs, Pods, and PersistentVolumeClaims (confirmed via `kubectl auth can-i`).
- **Training happens as a Kubernetes Job**, running inside a container pulling a public image (e.g. a PyTorch+CUDA image from Docker Hub) — no local Docker install needed, since Kubernetes pulls the image itself.

---

## 2. What We Still Need From IT

These are genuinely blocking or quality-affecting, and aren't answerable from our side alone:

1. **Storage.** `kubectl get storageclass` returns nothing — there is no StorageClass to provision a PersistentVolumeClaim against, even though we have permission to create PVCs. **Ask:** is there a StorageClass we should use, or a shared NFS/mount path for our namespace, for dataset storage (~60GB now, up to ~260GB total) and model checkpoints? Without this, anything we write inside a pod disappears when the pod ends.
2. **Outbound internet / image registry access.** We haven't yet confirmed whether `bmu-worker` can pull public images from Docker Hub. **Ask:** does the cluster allow pulling public container images, or is there an internal/mirrored registry we're expected to use instead?
3. **Namespace resource quota.** We don't know the actual limits on our namespace (max concurrent MIG slices, max Jobs, storage quota, max job wall-clock time). **Ask:** what's our namespace's ResourceQuota/LimitRange, and is there a maximum job runtime we need to design around?
4. **Which MIG profile we should actually use for training**, not just verification. We can see 21× the smallest slice (18GB) but only 2× the largest (71GB). **Ask:** is there guidance on which profile to request for typical training workloads, and whether requesting the larger profiles is expected/allowed for a team like ours.
5. **Whether Kubernetes Jobs are the only supported path**, or whether there's also a Slurm-style / bare-metal execution option we're missing — the head node has HPC-style environment modules available (`module avail` shows a CUDA 12.9 toolkit module, MPI modules, etc.), which is a bit inconsistent with a container-only workflow. Worth asking directly rather than assuming.
6. **Onboarding documentation.** Is there an existing wiki/doc for this cluster? Everything in this file was reverse-engineered from command output — a lot of this would've been answered by a one-page onboarding doc, if one exists.

---

## 3. Diagnostic Commands Run (and what each told us)

Run these from `bmu-headnode` after `ssh dgx-s-bmu-cse-<your-id>@10.1.0.176`.

```bash
# Confirm you're on the head node, not a GPU node
whoami
hostname

# GPU is NOT on the head node — this fails here by design
nvidia-smi

# See what HPC-style modules exist (informational — see open question #5 above)
module avail

# Confirms this is a k8s cluster, not Slurm
which sinfo squeue sbatch srun salloc kubectl
sinfo

# Cluster/node topology
kubectl get nodes

# GPU resources actually schedulable on the worker node
kubectl describe node bmu-worker | sed -n '/Capacity/,/System Info/p'

# Your assigned namespace (cluster-wide namespace listing is forbidden — expected)
kubectl config view --minify -o jsonpath='{..namespace}'
kubectl get namespace

# Confirm what you're actually allowed to create in your own namespace
kubectl auth can-i create jobs
kubectl auth can-i create pods
kubectl auth can-i create persistentvolumeclaims

# Storage — currently empty/missing, see open question #1
kubectl get storageclass
kubectl get pvc

# Confirm no local Docker is needed/available (expected — not a problem by itself)
docker version
which docker podman buildah

# Device plugin check (inconclusive on its own — pods list is forbidden cluster-wide,
# but the MIG resource counts under kubectl describe node already confirm it's running)
kubectl get pods -A | grep -i nvidia
```

**GPU verification job** (submits an actual test workload requesting a MIG slice):

```bash
# Create the file directly on bmu-headnode (don't rely on file transfer from a laptop —
# see §4 below), then:
kubectl apply -f verify_gpu_job.yaml
kubectl get pods -n dgx-s-bmu-cse-240574-restricted -w
kubectl logs job/milan-verify-gpu -n dgx-s-bmu-cse-240574-restricted

# If stuck Pending:
kubectl describe pod -n dgx-s-bmu-cse-240574-restricted -l job-name=milan-verify-gpu

# Clean up after:
kubectl delete -f verify_gpu_job.yaml
```

The manifest itself is at `model/setup/k8s/verify_gpu_job.yaml`.

**Status as of writing:** job submitted, result not yet confirmed — update this line once `kubectl logs` shows `PASS` or a real error.

---

## 4. Gaps We Noticed in How Access Was Provisioned

Raised in a spirit of "this cost us debugging time," not a complaint — worth mentioning to IT so the next team doesn't hit the same wall:

- **No indication up front that this is a Kubernetes cluster, not direct-SSH GPU access.** We initially assumed `ssh` + `conda activate` + run a training script would work, since that's the standard pattern for a "GPU box." Nothing in the credentials handoff mentioned Kubernetes, MIG, or that the head node itself has no GPU — we found this out by trial and error (`nvidia-smi: command not found`, which reads like a driver problem, not a "wrong node type" problem).
- **PVC-create permission granted with no StorageClass to bind against.** We can create PersistentVolumeClaims, but there's nothing for them to provision from — this is a half-configured setup: either the permission or the StorageClass is missing something on the provisioning side.
- **No documented resource quota for our namespace.** We can't responsibly plan job sizes (which MIG profile, how many concurrent jobs) without knowing our actual ceiling — this should be provided alongside account creation, not discovered by hitting a quota error mid-run.
- **No onboarding document accompanying the credentials.** A short doc covering "this is a k8s cluster, here's your namespace, here's how to request a GPU, here's your storage path" would have replaced most of this file.
