# DGX-H200 Access — SSH Workflow

Target location in the repo: `model/setup/dgx_access.md`. This is the reference for T-A2 and for anyone new to the team who needs to get onto the box.

---

## 1. First login

```bash
ssh <your-username>@<dgx-host>
```

Get `<dgx-host>` and your credentials/SSH key from the BMU AI Lab admins — this isn't something to guess at or share in the repo. If they issue an SSH key rather than a password, add it to your local `~/.ssh/config` so you don't retype the path every time:

```
Host milan-dgx
    HostName <dgx-host>
    User <your-username>
    IdentityFile ~/.ssh/<your-key>
```

Then just `ssh milan-dgx`.

**Ask the admins these two things explicitly, don't assume:**
1. Is GPU access open (any user, any GPU, any time) or scheduled (Slurm `srun`/`sbatch`)? This changes step 4 below.
2. Is there a shared, fast-storage directory for datasets, or does each user get their own quota? This changes where INCLUDE/iSign get downloaded to (§4).

## 2. Persistent sessions — always use tmux

**Never run a training job in a raw SSH session.** A dropped connection kills the process. Every long-running command goes inside `tmux`:

```bash
tmux new -s milan-train        # start a named session
# ... run your training command here ...

# detach (leaves it running): Ctrl+b, then d
# reattach later, from a fresh SSH login:
tmux attach -t milan-train

# list all your sessions:
tmux ls
```

Name sessions after what they're doing (`milan-train`, `milan-encoder-pretrain`, `milan-eval`) — on a shared box, a session just called `tmux` from three different people is how work gets accidentally killed.

## 3. Environment setup (one-time per user)

```bash
git clone <repo-url>
cd milan
conda env create -f environment.yml
conda activate milan
bash model/setup/install_torch.sh
python model/setup/verify_gpu.py
```

If `verify_gpu.py` prints `PASS`, you're done — see `environment.yml` and `model/setup/install_torch.sh` for why torch is installed as a separate step.

## 4. Data storage

Do **not** let every team member download INCLUDE (56GB) and iSign (~200GB) into their own home directory — confirm a shared dataset path with the admins (e.g. `/data/milan/` or similar) and point `model/data/download_include.sh` / `download_isign.sh` at it. One shared, read-only-mounted copy for everyone's training jobs.

## 5. Multi-user coordination

A shared DGX with no coordination is how two people collide on the same GPU and both lose a run. Before starting anything that will run for a while:

1. Check `RUNS.md` in the repo root — log what you're about to run, which GPU, and roughly how long, *before* you start it.
2. If GPU access is Slurm-scheduled (§1), this is partly handled for you — but still log it, since Slurm won't tell your teammates what's queued.
3. Update `RUNS.md` again when the run finishes (or fails) — include the result, not just the start.

## 6. Multi-GPU — only when you actually need it

Every task in the current plan (`docs/technical-spec.md` §1) is scoped to run on a **single GPU with mixed precision** (`torch.cuda.amp` / bf16). H200s have plenty of headroom for the encoder pretrain and the iSign fine-tune at the sizes we're targeting. Don't reach for `torchrun`/DDP or `accelerate`'s multi-GPU path unless a specific run is measurably too slow single-GPU — it adds real debugging complexity for no benefit at this scale, and nothing in the task list requires it as written.

## 7. Data transfer (if you ever need to move something off/onto the DGX yourself)

```bash
# from your local machine, copying up:
rsync -avz --progress ./some_dir/ milan-dgx:~/milan/some_dir/

# copying a result down:
rsync -avz --progress milan-dgx:~/milan/model/export/output.onnx ./
```

Prefer `rsync -avz` over `scp` for anything more than a single small file — it resumes on interruption, `scp` doesn't.
