# Local Setup & Connecting to the DGX-H200

Target location in the repo: `model/setup/local_setup.md`.

Everyone on the team gets their own IP, username, and password from the BMU AI Lab admins — same format, different values:

```
IP Address: 10.1.0.176
Username:   dgx-s-bmu-cse-enrollment
Password:   your-own-password
```

Fill in your own values wherever `<ip>` / `<username>` appear below.

---

## 1. What you need installed locally

- **Git**
- **An SSH client** — built into macOS/Linux terminals already. On Windows, use the OpenSSH client (PowerShell/Terminal) or WSL.
- **(Optional) VS Code + the Remote - SSH extension** — lets you edit files on the DGX directly instead of a bare terminal. Worth it if you'll be writing/debugging code on the box.

Nothing else is required locally — no CUDA, no conda, no GPU. All training happens on the DGX; your machine just drives it over SSH.

## 2. Connect

```bash
ssh <username>@<ip>
```

You'll be prompted for your password each time (this is normal for password auth — there's no key involved). Type it and hit enter; nothing will appear on screen as you type, that's expected.

**Save yourself retyping the full command** — add this to `~/.ssh/config` (create the file if it doesn't exist):

```
Host milan-dgx
    HostName <ip>
    User <username>
```

Now `ssh milan-dgx` works, and it'll still prompt for your password.

**First connection** will show `The authenticity of host '...' can't be established. Are you sure you want to continue connecting?` — type `yes`. Only happens once per machine.

**If it doesn't connect:**

| Error | Likely cause |
|---|---|
| `Connection refused` / timed out | Wrong IP, or you need to be on a specific network/VPN — check with admins |
| `Permission denied` | Wrong username/password — re-check what you were given |
| Connects but `nvidia-smi` fails (next step) | Ask the admins before doing anything else |

## 3. Confirm the GPU is visible

Once connected:

```bash
nvidia-smi
```

Should print a table with the H200(s). If not, stop here and check with the admins.

## 4. Check for required software on the DGX — install if missing

The box is shared, so tools may or may not already be set up. Check each, and install what's missing:

```bash
which git tmux conda unzip jq curl
```

- **`conda` missing** → install Miniconda:
  ```bash
  wget https://repo.anaconda.com/miniconda3/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  # follow the prompts, then restart your shell (or `source ~/.bashrc`)
  ```
- **`git`, `tmux`, `unzip`, `jq`, `curl` missing** → if you have `sudo`:
  ```bash
  sudo apt-get update && sudo apt-get install -y git tmux unzip jq curl
  ```
  If you don't have `sudo` on a shared lab machine, ask the admins to install them once for everyone rather than everyone hitting the same wall separately.

`jq` and `unzip` specifically are only needed for `model/data/download_include.sh` — skip installing them if you're not the one downloading the dataset.

## 5. Clone the repo and set up the environment — on the DGX

```bash
git clone <repo-url> ~/milan
cd ~/milan
conda env create -f environment.yml
conda activate milan
bash model/setup/install_torch.sh
python model/setup/verify_gpu.py
```

`verify_gpu.py` printing `PASS` is the real confirmation everything works end to end.

## 6. tmux — don't run anything long outside of it

A dropped SSH connection kills whatever's running in that session. Always wrap training/long jobs:

```bash
tmux new -s <name>       # start, named after what you're doing
# Ctrl+b, then d         # detach — keeps running in the background
tmux attach -t <name>    # reattach later
tmux ls                  # see your running sessions
```

Full coordination conventions (so you don't collide with a teammate on the same GPU) are in `dgx_access.md` §2 and §5.

## 7. Where things live

- **`RUNS.md`** — repo root, not inside `model/`. Log your run there *before* you start it, shared across the whole team.
- **The dataset itself** — not in the repo. Downloaded separately on the DGX, in a directory outside your git clone (e.g. `~/milan-data/`) — see `model/data/download_include.sh` and the note in that script.
- **This file + `dgx_access.md`** — `model/setup/`.

## 8. Checklist — you're ready when

- [ ] `ssh milan-dgx` connects (password prompt, then a shell)
- [ ] `nvidia-smi` shows the GPU(s)
- [ ] `git`, `tmux`, `conda`, `unzip`, `jq`, `curl` all present (or installed)
- [ ] `conda activate milan` works
- [ ] `python model/setup/verify_gpu.py` prints `PASS`
- [ ] You know where `RUNS.md` and the dataset directory are
