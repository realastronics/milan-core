# DGX-H200 Access — SSH Workflow

Target location in the repo: `model/setup/dgx_access.md`. This is the reference for T-A2 and for anyone new to the team who needs to get onto the box.

---

## 1. First login

```bash
ssh <your-username>@<your-ip>
```

Each teammate gets their own IP, username, and password from the BMU AI Lab admins (see `local_setup.md` for the exact format and a walkthrough if this is your first time). This is **password auth, not a key** — you'll be prompted for your password on every connection, which is normal.

Save yourself retyping the full command by adding this to `~/.ssh/config`:

```
Host milan-dgx
    HostName <your-ip>
    User <your-username>
```

Then just `ssh milan-dgx` (still password-prompted each time).

**Worth confirming with the admins, since it affects the rest of this doc:**
1. Is GPU access shared/open across the team's accounts, or does each person's login get their own allocation? (Individual usernames per teammate suggest the latter, but confirm rather than assume — it changes how much §5 coordination actually matters.)
2. Is there a shared, fast-storage directory for datasets, or does each user get their own quota? This changes where INCLUDE gets downloaded to (§4).

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

Do **not** let every team member download INCLUDE (~60GB) into their own home directory — confirm a shared dataset path with the admins, or agree on one as a team (e.g. `~/milan-data/` per user if there's no shared volume, or a single `/data/milan/` if there is). Point `model/data/download_include.sh` at whichever it is. One copy per storage location, not one per person. (iSign is not in scope yet — this phase is INCLUDE-only, see `docs/technical-spec.md`.)

## 5. Multi-user coordination

A shared DGX with no coordination is how two people collide on the same GPU and both lose a run. Before starting anything that will run for a while:

1. Check `RUNS.md` in the repo root — log what you're about to run, which GPU, and roughly how long, *before* you start it.
2. If GPU access turned out to be shared/open (confirmed per §1), logging here is what actually prevents a collision — nothing else will.
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

Both will prompt for your password (no key involved, same as `ssh`). Prefer `rsync -avz` over `scp` for anything more than a single small file — it resumes on interruption, `scp` doesn't.
