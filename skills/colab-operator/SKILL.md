---
name: colab-operator
description: Operate Google Colab environments via the `colab` CLI. Use when asked to create or manage GPU/TPU sessions, train or fine-tune a model on Colab, run Python/shell on a remote Colab VM, sync files, automate environment setup (packages, auth, Drive), or export session history.
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
- **Multiple accounts (per-account quotas).** Colab enforces limits **per Google account**: a daily compute quota and a concurrency limit on GPU assignments. When one account is exhausted, `colab new --gpu T4` on it fails (HTTP 412 "precondition failed"), rejects the accelerator (HTTP 400), or the allocation request simply times out — while another signed-in account still has headroom.
  - `colab account list` shows every signed-in account, its email, and which one is active. `colab account add <label>` signs in another one (interactive browser step; the token is stored under `~/.config/colab-cli/accounts/<label>.json`). `colab account use <label>` switches; `colab account rm <label>` forgets it. `colab account add <label> --from-file <token.json>` imports an already-saved token without a browser.
  - `colab --account <label> <cmd>` uses one account for a single command.
  - **Automatic rotation**: when an allocation is refused (412/400/429) or times out, `colab new` switches to the next account that has a token and retries, announcing the switch on stderr. So an agent does not have to manage this by hand.
  - Each session records the account that owns it, so `colab stop` and token refreshes go to the right account even after a rotation.
- **Signing in needs a real TTY.** `colab account add` runs the copy-paste OAuth flow and calls `input()` for the authorization code; in a piped shell stdin hits EOF and the sign-in dies right after printing the URL. Use the psmux helper (see below) or `psmux attach` for it.

## When a command must outlive the shell (psmux)

Two CLI operations need more than a piped shell can give:

- **The OAuth sign-in** above — it needs a PTY for the code prompt.
- **A long job** (`colab run`, `colab exec` on a multi-hour script). In a plain shell the caller blocks and dies with it; in a psmux session the job keeps running and its output can be collected in a later step instead of re-running it (re-running is what burns quota and context).

`scripts\colab-psmux.ps1` wraps this:

```bash
# sign in an account in a real PTY (prints the URL, then paste the code)
scripts/colab-psmux.ps1 auth work
scripts/colab-psmux.ps1 auth work -Code 4/0Abc...     # or paste it yourself

# run a job that outlives the current tool call
scripts/colab-psmux.ps1 job train exec -s train -f job.py --json
scripts/colab-psmux.ps1 log train      # tail the output later
scripts/colab-psmux.ps1 list           # sessions + logs
scripts/colab-psmux.ps1 kill train     # tear it down
```

The job's stdout+stderr and its exit code land in `~/.config/colab-cli/psmux-logs/<session>.log`, so a later step reads the result instead of re-running the command.

## Workflow

### Provision
- `colab new -s <name>` (CPU). Add `--gpu A100` or `--tpu v6e1` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.
- **A matching runtime is reused, not duplicated.** If the account already has a runtime with the requested hardware, `colab new` adopts it and says so (`Adopted the running T4 runtime …`) instead of asking for a second one — which the account's concurrency limit would refuse (HTTP 412) and which would waste time. Pass `--force-new` to insist on a fresh runtime.
- **Quota exhaustion rotates accounts.** If the allocation is refused (412/400/429) or the request times out, the CLI switches to the next signed-in account and retries. See the authentication section above.

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
- **`--timeout` is a real wall-clock bound.** When it expires the CLI interrupts the kernel and exits non-zero, instead of letting a silent cell spin forever while the VM keeps billing compute quota. If the interrupt is not acknowledged the error says so explicitly — stop the session (`colab stop -s <name>`) to release the VM.

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

## Training a model, end to end

The loop, in order. Each step says what the next one is.

0. **`colab doctor`** — one verdict, and if something is broken the exact
   commands to fix it. Run this first; it costs nothing and touches no quota.
   `colab doctor --json` is the machine-readable form.
1. **Rent** — `colab new -s proj --gpu T4`. Always pass `-s`. If the quota is
   spent the CLI rotates to another account by itself; if a matching runtime
   is already running it adopts that one.
2. **Train** — `colab exec -s proj -f train.py --timeout 3600 --json`.
   `-f` reads the script **locally**; `--timeout` is an *idle* limit (seconds
   between outputs) so a run that logs every minute may take hours; `--json`
   puts exactly one JSON envelope on stdout and the raw log on stderr.
3. **Collect** — `colab status -s proj --json --files` lists what is on the VM
   (with the paths that work), then `colab download -s proj <path> <local>`.
4. **Iterate** — edit the script locally and run step 2 on the *same* session:
   the runtime, the dataset and the installed packages all stay.
5. **Stop** — `colab stop -s proj`. A T4 left running keeps burning the
   account's daily quota; this is the fastest way to exhaust it.

Remote paths: the contents API is rooted at the notebook root but the VM's
working directory is `/content`, so `model.json` and `content/model.json` name
the same file and both work.

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
