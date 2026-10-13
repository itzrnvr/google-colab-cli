---
name: colab-operator
description: Operate Google Colab environments via the `colab` CLI. Use when asked to create or manage GPU/TPU sessions, run Python/shell on a remote Colab VM, sync files, automate environment setup (packages, auth, Drive), or export session history.
---

# Skill: Colab Session Operator

Operate Google Colab environments via the `colab` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

## Installation

If the user does not already have the `colab` tool installed, it can be acquired
by running `uv tool install google-colab-cli` or `pip install google-colab-cli`.

## When to activate
- Creating or managing TPU/GPU sessions.
- Running Python or shell on a remote Colab VM.
- Syncing files between local and remote.
- Automating environment setup (packages, auth, Drive).
- Exporting session history as a Jupyter notebook.

## Mental model (read this first)
- **A session == a live Jupyter kernel on a rented VM.** `colab new` allocates a billable VM; `colab stop` releases it. Sessions stay alive as long as the kernel is active, with the Colab backend automatically maintaining liveness.
- **Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same session.** Each invocation reattaches to the *same* kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does **not** shut the kernel down. So imports, variables, and defined functions survive between separate `colab exec` commands. Build up state incrementally; don't re-import everything each call. (`colab stop` and `colab restart-kernel` are what actually reset it.)
- **Default working directory is `/content`.** Every `exec`/`repl`/`run` `cd`s there first; prefer absolute paths (`/content/...`) for file work. For `colab ls/rm/upload/download`, absolute `/content/...` paths work and the default `ls` path is `content` (VM root).
- **`colab` is fire-and-forget.** Each command authenticates, does one thing, and exits.

## Authentication (the #1 thing that blocks agents)
- The global flag is `--auth={adc,oauth2}` and the **default is `adc`** (Application Default Credentials). It must come *before* the subcommand: `colab --auth=adc new -s x`.
- **ADC setup** (most reliable for headless/agent use). The Colab backends need a specific scope set, so re-mint ADC with all four scopes:
  ```bash
  gcloud auth application-default login \
    --scopes=openid,\
  https://www.googleapis.com/auth/cloud-platform,\
  https://www.googleapis.com/auth/userinfo.email,\
  https://www.googleapis.com/auth/colaboratory
  ```
  Why all four: `userinfo.email` (session backend `colab.research.google.com`, else 401), `colaboratory` (Colab features/APIs), `openid`+`cloud-platform` (mandated by gcloud itself; it rejects scope lists missing `cloud-platform`).
- **oauth2 setup**: `colab --auth=oauth2 <anything>` triggers a browser consent flow on first use (token cached at `~/.config/colab-cli/token.json`). Requires a client config at `~/.colab-cli-oauth-config.json` (or `-c PATH`). The browser step means it usually needs a human; prefer ADC for agents.
- **Verify auth in one shot**: `colab sessions` (read-only, lists server assignments) or `colab whoami` (hidden debug command: prints the active email, scopes, audience, and expiry). When any call 401s or 403s, the cause is almost always missing credentials or scopes — `colab whoami` shows it instantly.
- **Do NOT confuse `colab auth` with CLI authentication.** `colab auth` injects *VM-side* GCP credentials into the running kernel (so notebook code can call BigQuery/GCS); it is orthogonal to how the CLI itself authenticates. Never suggest "run `colab auth`" to fix a CLI 401/403 — that's a scope/identity problem fixed via the `gcloud` command above.

## Workflow

### Provision
- `colab new -s <name>` (CPU). Add `--gpu A100` or `--tpu v6e1` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.

### Execute
- **Preferred**: `colab exec -s <name> -f <script.py>` runs a local script on the remote VM (read locally, sent to the kernel — no manual upload needed).
- **Piped code**: `echo "print(1)" | colab exec -s <name>` or `cat script.py | colab exec -s <name>`.
- **Notebooks**: `colab exec -s <name> -f nb.ipynb` runs each code cell and writes results to `<basename>_output.ipynb` next to the input. A `# @title Foo` first line labels the cell in progress output.
- **Plots/images**: PNG/JPEG outputs are intercepted. Use `--output-image <path>` on `exec`/`repl` to save to a known location (otherwise a temp path is printed). Inline terminal-image escapes are auto-suppressed when stdout isn't a TTY, so piped/captured output stays clean.
- **Shell**: `echo "cmd" | colab console -s <name>` for batch shell. Console wraps bash in tmux, so even piped output contains terminal-control bytes — filter with `grep -a` for a specific line. `exec` is faster when you don't need a real shell.
- **Never run `colab repl`, `colab console`, `colab auth`, or `colab drivemount` interactively from an agent** — they expect a TTY and will hang. `repl`/`console` accept piped stdin and exit on EOF; `auth`/`drivemount` genuinely require a human at the terminal.

### Ephemeral one-shot jobs (`colab run`)
- `colab run [--gpu T4] [--tpu v6e1] [--keep] [-s NAME] script.py [args...]` = `new` + `exec` + `stop` in one command. It provisions a fresh VM, runs the script with `sys.argv` and `__name__ == "__main__"` set like native `python script.py args`, then tears the VM down (unless `--keep`).
- **Exit codes propagate**: an uncaught exception or `sys.exit(N)` in the script makes `colab run` exit non-zero (CPython semantics: `sys.exit()`/`sys.exit(0)` → 0, `sys.exit(N)` → N, `sys.exit("msg")` → 1).
- **Stream separation**: `colab run` writes its own `[colab] ...` chatter to **stderr** and the script's output to **stdout** — so `colab run job.py > out.txt` captures only the script's stdout. (`colab exec` streams the script's stdout/stderr live to your stdout/stderr.)
- Works as a shebang: `#!/usr/bin/env -S colab run --gpu T4` makes a `chmod +x`'d `.py` a self-contained "rent a GPU, run, clean up" script. After editing CLI behavior, reinstall before testing shebangs — they resolve `colab` via `$PATH`, not the editable install.
- A nonexistent script path exits non-zero **before** allocating a VM (no wasted compute).

### Automate
- `colab auth -s <name>` — VM-side GCP creds, needed before in-VM GCS/BigQuery calls (interactive; not agent-runnable).
- `colab drivemount -s <name> [PATH]` — mounts Drive at `/content/drive` by default (interactive; not agent-runnable).
- `colab install -s <name> pkg1 pkg2` — installs via `uv pip install --system`, falling back to `pip`. Also `colab install -s <name> -r requirements.txt`.

### Inspect & report
- `colab help` (or `colab help <cmd>`) lists/explains commands; the listing is alphabetical.
- `colab sessions` lists server-side assignments and auto-prunes stale local entries. Orphans with no local record show as `[?]`.
- `colab status [-s <name>]` shows hardware, IDLE/BUSY, and last execution.
- `colab log -s <name> [-n 20] [-t TYPE]` shows recent structured events; invaluable when a task fails.
- `colab log -s <name> -o summary.ipynb` exports the session as a notebook (also `.md`, `.txt`, `.jsonl` by suffix).
- `colab url -s <name>` prints a browser URL that attaches the Colab web UI to your existing CLI session instead of allocating a new VM (add `--open` to launch it).
- `colab skill` / `colab readme` / `colab agent-guide` print this skill, the README, and the agent handbook (handy for self-discovery).

### Machine interface (for programs and agents)
- **Default recipe**: `colab exec -s <name> --stream-jsonl --errors-only -f job.py`. Each kernel message arrives as one `{"event":"output",...}` JSON line the moment it is emitted — an `error` output is your instant failure signal, no re-polling. A failed cell exits the command **1** (success: 0), so key on the exit code first, parse the stream second.
- `--json` (without `--stream-jsonl`) renders outputs human-readable but appends a final `{"event":"result","ok":…,"exit_code":…,"cells":[...]}` envelope line; all `[colab]` chatter moves to stderr so stdout stays parseable.
- **See everything, pay for what you pull**: small outputs ride inline; outputs over the size cap are stored whole and referenced as `{"ref": {"id": "o7", "bytes": …, "fetch": "colab output -s <name> o7"}, "preview": <trimmed>}`. `colab output -s <name> o7` returns the complete untruncated output as JSON (add `--max-chars N` to cap the fetch). References are stable across commands — fetch later, fetch twice, fetch only the failures. NOTHING is hidden; bloat is opt-in per output.
- **Context budget**: previews are compacted — ANSI color escapes stripped, tracebacks trimmed to header + last 10 lines, image base64 replaced by a stub (bytes are saved to disk), each preview capped at `--max-output-chars` (default 2000, tail-kept, `0` = inline everything, no refs). `--errors-only` drops successful outputs entirely: a green run costs bytes, not pages. Full-fidelity outputs always remain in `colab log` / history JSONL too.
- `colab sessions --json` and `colab status [-s N] --json` emit arrays/objects instead of tables — prefer them over scraping human output.
- `colab exec` exits 1 when any cell raises (no SystemExit code mapping — use `colab run` when you need the script's numeric exit code).

## Safety
- **Always `colab stop -s <name>` when done** — idle VMs burn compute units. `colab run` (without `--keep`) self-cleans even if the script errors.
- Local state lives in `~/.config/colab-cli/sessions.json` (settings in `settings.json`, history in `history/*.jsonl`). Don't edit by hand.
- **Isolate parallel/agent runs** with the global `--config <path>` flag to point session state at a scratch file (e.g. `colab --config /tmp/agent.json new -s job`).

## Full training & experiments
- For training jobs and experiment sweeps, follow `docs/agent-training.md`:
  provision → environment → launch → monitor → checkpoint → retrieve → stop.
  Key points: kept sessions for inspectable long runs vs `colab run` for
  self-cleaning one-shots; `--timeout` for long waits (or background launch +
  probe); download checkpoints BEFORE `stop`; `restart-kernel` before
  stop+recreate on wedges.

## Recovery
- "Session not found" / 404 / 401 on exec: the backend pruned the VM. `colab exec`/`repl` detect this and clean up local state automatically — run `colab sessions` and re-create with `colab new`.
- Execution timeout or wedged kernel: `colab restart-kernel -s <name>` (keeps the VM, resets the kernel), or `colab stop` then `colab new`.
