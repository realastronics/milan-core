# Local Setup & Connecting to the DGX-H200

Target location in the repo: `model/setup/local_setup.md`. Written for a first-time remote-GPU user — if you've never SSH'd into a shared machine before, start here. `dgx_access.md` (same folder) is the reference once you're comfortable; this is the walkthrough.

---

## 0. What you need installed locally

- **Git** — to clone the repo.
- **An SSH client** — already built into macOS and Linux terminals. On Windows, use the OpenSSH client bundled with recent Windows (available from PowerShell/Terminal), or WSL if you already use it.
- **Python 3.11** and **conda** (Miniconda is enough — you don't need the full Anaconda distribution) — [miniconda install](https://docs.conda.io/projects/miniconda/en/latest/).
- **(Optional, recommended) VS Code** with the **Remote - SSH** extension — lets you browse/edit files on the DGX as if they were local, instead of editing over a bare terminal.

You do **not** need a GPU on your own laptop for anything in this project — all training happens on the DGX. Your laptop's job is: write code, push it to GitHub, and drive the DGX over SSH.

## 1. Clone the repo

```bash
git clone <repo-url>
cd milan
```

## 2. Local Python environment (for local scripts/tools only — not training)

This is a **separate, lightweight environment from the DGX's** — use it for things you run on your own machine, like testing the extension's build tooling or a small local script. It is not where model training happens.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt   # if/when this file exists — otherwise install per-task as needed
```

If you're not sure whether something needs the DGX environment (`environment.yml`, §3 of `dgx_access.md`) or this local one: **if it touches the GPU, trains a model, or processes the full datasets, it's DGX. If it's a quick script, a linter, or extension tooling, it's local.**

## 3. Generate an SSH key (if you don't already have one)

Check first — you might already have one:

```bash
ls ~/.ssh
```

If you see `id_ed25519` / `id_ed25519.pub` (or `id_rsa` / `id_rsa.pub`), you're set — skip to step 4. Otherwise:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

Press Enter through the prompts (default file location is fine; a passphrase is optional but recommended). This creates two files: `~/.ssh/id_ed25519` (**private** — never share this, never commit it) and `~/.ssh/id_ed25519.pub` (**public** — this is the one you send to the lab admins).

Send the **public** key to the BMU AI Lab admins for account access:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output and send it to them however they've asked for it (email, form, etc.).

## 4. Set up an SSH shortcut

Once the admins confirm your account is set up and give you the host address and username, add this to `~/.ssh/config` (create the file if it doesn't exist):

```
Host milan-dgx
    HostName <dgx-host-they-gave-you>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

Now, instead of typing the full `ssh username@host` every time, you can just run:

```bash
ssh milan-dgx
```

## 5. First connection — what to expect

```bash
ssh milan-dgx
```

- First time connecting to any new host, you'll see a prompt like `The authenticity of host '...' can't be established. Are you sure you want to continue connecting?` — type `yes`. This is normal and only happens once per machine.
- If it asks for a passphrase, that's the one you set on your key in step 3 (not your account password).
- If you land at a shell prompt on the DGX, you're in.

**Common first-time errors:**

| Error | Likely cause |
|---|---|
| `Permission denied (publickey)` | Your public key hasn't been added on the lab's side yet, or you pointed `IdentityFile` at the wrong file — double check step 3/4 |
| `Connection refused` / `Connection timed out` | Wrong host address, or you need to be on a specific network/VPN — check with the admins |
| `Host key verification failed` | Rare, usually means the machine's identity changed — don't just bypass this, ask the admins first |

## 6. Confirm you can see the GPU

Once connected:

```bash
nvidia-smi
```

This should print a table listing the H200(s) and their current usage. If this fails or shows nothing, stop here and check with the admins before doing anything else — don't proceed to environment setup on a session that can't see a GPU.

## 7. Set up the shared environment and verify

Follow `dgx_access.md` §3 from here — clone the repo *on the DGX* (yes, separately from your local clone in step 1), create the `milan` conda environment, and run `verify_gpu.py`. That script's `PASS` output is the real confirmation everything works end to end.

## 8. Working without losing your session — tmux, quickly

You'll run real training jobs inside `tmux` so a dropped connection doesn't kill them. The four commands you actually need day to day:

```bash
tmux new -s <name>       # start a new named session
# Ctrl+b, then d         # detach — leaves it running in the background
tmux attach -t <name>    # reattach from a fresh SSH login
tmux ls                  # list your running sessions
```

Full session-naming and coordination conventions (so you don't collide with a teammate) are in `dgx_access.md` §2 and §5.

## 9. Where things live

- **`RUNS.md`** — repo root (`milan/RUNS.md`), not inside `model/`. It's shared across the whole team and logs every DGX training run, so it lives somewhere everyone sees it immediately, not buried in a subfolder. Log your run there *before* you start it.
- **This file and `dgx_access.md`** — `model/setup/`, since they're specifically about the model-training track's infrastructure.

## 10. Checklist — you're ready when

- [ ] `ssh milan-dgx` connects without errors
- [ ] `nvidia-smi` on the DGX shows the GPU(s)
- [ ] `conda activate milan` works on the DGX
- [ ] `python model/setup/verify_gpu.py` prints `PASS`
- [ ] You know where `RUNS.md` is and what to write in it before starting a run
