---
name: colab-operator
description: Operate Google Colab environments via the `colab-tool` CLI, a CUSTOM Colab CLI build (itzrnvr/google-colab-cli) with multi-account support, automatic quota failover, a `doctor` self-check and agent-friendly output. Use when asked to create or manage GPU/TPU sessions, train or fine-tune a model on Colab, run Python/shell on a remote Colab VM, sync files, automate environment setup (packages, auth, Drive), or export session history.
---

# Skill: colab-tool (custom Colab CLI)

`colab-tool` is a **custom** Colab CLI (itzrnvr/google-colab-cli), not the stock
upstream package. On top of the upstream behaviour it adds: multiple signed-in
Google accounts with automatic quota failover, `colab-tool doctor` (one command
that says what is broken and how to fix it), stdout that stays machine-parseable
under `--json`, and `colab-tool agent-guide`, which can install itself as skill
data.

Operate Google Colab environments via the `colab-tool` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

## Installation

If the user does not already have the `colab-tool` tool installed, it can be acquired
by running `uv tool install google-colab-cli` or `pip install google-colab-cli`.

## When to activate
- Creating or managing TPU/GPU sessions.
- Running Python or shell on a remote Colab VM.
- Syncing files between local and remote.
- Automating environment setup (packages, auth, Drive).
- Exporting session history as a Jupyter notebook.

## Mental model (read this first)
- **A session == a live Jupyter kernel on a rented VM.** `colab-tool new` allocates a billable VM; `colab-tool stop` releases it. Sessions stay alive as long as the kernel is active, with the Colab backend automatically maintaining liveness.
- **Kernel state PERSISTS across `colab-tool exec` / `colab-tool repl` calls in the same session.** Each invocation reattaches to the *same* kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does **not** shut the kernel down. So imports, variables, and defined functions survive between separate `colab-tool exec` commands. Build up state incrementally; don't re-import everything each call. (`colab-tool stop` and `colab restart-kernel` are what actually reset it.)
- **Default working directory is `/content`.** Every `exec`/`repl`/`run` `cd`s there first; prefer absolute paths (`/content/...`) for file work. For `colab-tool ls/rm/upload/download`, absolute `/content/...` paths work and the default `ls` path is `content` (VM root).
- **`colab-tool` is fire-and-forget.** Each command authenticates, does one thing, and exits.

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
- **Verify auth in one shot**: `colab-tool sessions` (read-only, lists server assignments) or `colab-tool whoami` (hidden debug command: prints the active email, scopes, audience, and expiry). When any call 401s or 403s, the cause is almost always missing credentials or scopes — `colab-tool whoami` shows it instantly.
- **Do NOT confuse `colab auth` with CLI authentication.** `colab auth` injects *VM-side* GCP credentials into the running kernel (so notebook code can call BigQuery/GCS); it is orthogonal to how the CLI itself authenticates. Never suggest "run `colab auth`" to fix a CLI 401/403 — that's a scope/identity problem fixed via the `gcloud` command above.
- **Multiple accounts (per-account quotas).** Colab enforces limits **per Google account**: a daily compute quota and a concurrency limit on GPU assignments. When one account is exhausted, `colab-tool new --gpu T4` on it fails (HTTP 412 "precondition failed"), rejects the accelerator (HTTP 400), or the allocation request simply times out — while another signed-in account still has headroom.
  - `colab-tool account list` shows every signed-in account, its email, and which one is active. `colab-tool account add <label>` signs in another one (interactive browser step; the token is stored under `~/.config/colab-cli/accounts/<label>.json`). `colab-tool account use <label>` switches; `colab-tool account rm <label>` forgets it. `colab-tool account add <label> --from-file <token.json>` imports an already-saved token without a browser.
  - `colab --account <label> <cmd>` uses one account for a single command.
  - **Automatic rotation**: when an allocation is refused (412/400/429) or times out, `colab-tool new` switches to the next account that has a token and retries, announcing the switch on stderr. So an agent does not have to manage this by hand.
  - Each session records the account that owns it, so `colab-tool stop` and token refreshes go to the right account even after a rotation.
- **Signing in needs a real TTY.** `colab-tool account add` runs the copy-paste OAuth flow and calls `input()` for the authorization code; in a piped shell stdin hits EOF and the sign-in dies right after printing the URL. Use the psmux helper (see below) or `psmux attach` for it.

## When a command must outlive the shell (psmux)

Two CLI operations need more than a piped shell can give:

- **The OAuth sign-in** above — it needs a PTY for the code prompt.
- **A long job** (`colab-tool run`, `colab-tool exec` on a multi-hour script). In a plain shell the caller blocks and dies with it; in a psmux session the job keeps running and its output can be collected in a later step instead of re-running it (re-running is what burns quota and context).

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
- `colab-tool new -s <name>` (CPU). Add `--gpu A100` or `--tpu v6e1` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab-tool new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.
- **A matching runtime is reused, not duplicated.** If the account already has a runtime with the requested hardware, `colab-tool new` adopts it and says so (`Adopted the running T4 runtime …`) instead of asking for a second one — which the account's concurrency limit would refuse (HTTP 412) and which would waste time. Pass `--force-new` to insist on a fresh runtime.
- **Quota exhaustion rotates accounts.** If the allocation is refused (412/400/429) or the request times out, the CLI switches to the next signed-in account and retries. See the authentication section above.

### Execute
- **Preferred**: `colab-tool exec -s <name> -f <script.py>` runs a local script on the remote VM (read locally, sent to the kernel — no manual upload needed).
- **Piped code**: `echo "print(1)" | colab-tool exec -s <name>` or `cat script.py | colab-tool exec -s <name>`.
- **Notebooks**: `colab-tool exec -s <name> -f nb.ipynb` runs each code cell and writes results to `<basename>_output.ipynb` next to the input. A `# @title Foo` first line labels the cell in progress output.
- **Plots/images**: PNG/JPEG outputs are intercepted. Use `--output-image <path>` on `exec`/`repl` to save to a known location (otherwise a temp path is printed). Inline terminal-image escapes are auto-suppressed when stdout isn't a TTY, so piped/captured output stays clean.
- **Shell**: `echo "cmd" | colab-tool console -s <name>` for batch shell. Console wraps bash in tmux, so even piped output contains terminal-control bytes — filter with `grep -a` for a specific line. `exec` is faster when you don't need a real shell.
- **Never run `colab-tool repl`, `colab-tool console`, `colab auth`, or `colab-tool drivemount` interactively from an agent** — they expect a TTY and will hang. `repl`/`console` accept piped stdin and exit on EOF; `auth`/`drivemount` genuinely require a human at the terminal.

### Ephemeral one-shot jobs (`colab-tool run`)
- `colab-tool run [--gpu T4] [--tpu v6e1] [--keep] [-s NAME] script.py [args...]` = `new` + `exec` + `stop` in one command. It provisions a fresh VM, runs the script with `sys.argv` and `__name__ == "__main__"` set like native `python script.py args`, then tears the VM down (unless `--keep`).
- **Exit codes propagate**: an uncaught exception or `sys.exit(N)` in the script makes `colab-tool run` exit non-zero (CPython semantics: `sys.exit()`/`sys.exit(0)` → 0, `sys.exit(N)` → N, `sys.exit("msg")` → 1).
- **Stream separation**: `colab-tool run` writes its own `[colab] ...` chatter to **stderr** and the script's output to **stdout** — so `colab-tool run job.py > out.txt` captures only the script's stdout. (`colab-tool exec` streams the script's stdout/stderr live to your stdout/stderr.)
- Works as a shebang: `#!/usr/bin/env -S colab-tool run --gpu T4` makes a `chmod +x`'d `.py` a self-contained "rent a GPU, run, clean up" script. After editing CLI behavior, reinstall before testing shebangs — they resolve `colab-tool` via `$PATH`, not the editable install.
- A nonexistent script path exits non-zero **before** allocating a VM (no wasted compute).
- **`--timeout` is a real wall-clock bound.** When it expires the CLI interrupts the kernel and exits non-zero, instead of letting a silent cell spin forever while the VM keeps billing compute quota. If the interrupt is not acknowledged the error says so explicitly — stop the session (`colab-tool stop -s <name>`) to release the VM.

### Automate
- `colab auth -s <name>` — VM-side GCP creds, needed before in-VM GCS/BigQuery calls (interactive; not agent-runnable).
- `colab-tool drivemount -s <name> [PATH]` — mounts Drive at `/content/drive` by default (interactive; not agent-runnable).
- `colab-tool install -s <name> pkg1 pkg2` — installs via `uv pip install --system`, falling back to `pip`. Also `colab-tool install -s <name> -r requirements.txt`.

### Inspect & report
- `colab-tool help` (or `colab-tool help <cmd>`) lists/explains commands; the listing is alphabetical.
- `colab-tool sessions` lists server-side assignments and auto-prunes stale local entries. Orphans with no local record show as `[?]`.
- `colab-tool status [-s <name>]` shows hardware, IDLE/BUSY, and last execution.
- `colab-tool log -s <name> [-n 20] [-t TYPE]` shows recent structured events; invaluable when a task fails.
- `colab-tool log -s <name> -o summary.ipynb` exports the session as a notebook (also `.md`, `.txt`, `.jsonl` by suffix).
- `colab-tool url -s <name>` prints a browser URL that attaches the Colab web UI to your existing CLI session instead of allocating a new VM (add `--open` to launch it).
- `colab-tool skill` / `colab-tool readme` / `colab-tool agent-guide` print this skill, the README, and the agent handbook (handy for self-discovery).

### Machine interface (for programs and agents)
- **Default recipe**: `colab-tool exec -s <name> --stream-jsonl --errors-only -f job.py`. Each kernel message arrives as one `{"event":"output",...}` JSON line the moment it is emitted — an `error` output is your instant failure signal, no re-polling. A failed cell exits the command **1** (success: 0), so key on the exit code first, parse the stream second.
- `--json` (without `--stream-jsonl`) renders outputs human-readable but appends a final `{"event":"result","ok":…,"exit_code":…,"cells":[...]}` envelope line; all `[colab]` chatter moves to stderr so stdout stays parseable.
- **See everything, pay for what you pull**: small outputs ride inline; outputs over the size cap are stored whole and referenced as `{"ref": {"id": "o7", "bytes": …, "fetch": "colab-tool output -s <name> o7"}, "preview": <trimmed>}`. `colab-tool output -s <name> o7` returns the complete untruncated output as JSON (add `--max-chars N` to cap the fetch). References are stable across commands — fetch later, fetch twice, fetch only the failures. NOTHING is hidden; bloat is opt-in per output.
- **Context budget**: previews are compacted — ANSI color escapes stripped, tracebacks trimmed to header + last 10 lines, image base64 replaced by a stub (bytes are saved to disk), each preview capped at `--max-output-chars` (default 2000, tail-kept, `0` = inline everything, no refs). `--errors-only` drops successful outputs entirely: a green run costs bytes, not pages. Full-fidelity outputs always remain in `colab-tool log` / history JSONL too.
- `colab-tool sessions --json` and `colab-tool status [-s N] --json` emit arrays/objects instead of tables — prefer them over scraping human output.
- `colab-tool exec` exits 1 when any cell raises (no SystemExit code mapping — use `colab-tool run` when you need the script's numeric exit code).

## Training a model, end to end

The loop, in order. Each step says what the next one is.

0. **`colab-tool doctor`** — one verdict, and if something is broken the exact
   commands to fix it. Run this first; it costs nothing and touches no quota.
   `colab-tool doctor --json` is the machine-readable form.
1. **Rent** — `colab-tool new -s proj --gpu T4`. Always pass `-s`. If the quota is
   spent the CLI rotates to another account by itself; if a matching runtime
   is already running it adopts that one.
2. **Train** — `colab-tool exec -s proj -f train.py --timeout 3600 --json`.
   `-f` reads the script **locally**; `--timeout` is an *idle* limit (seconds
   between outputs) so a run that logs every minute may take hours; `--json`
   puts exactly one JSON envelope on stdout and the raw log on stderr.
3. **Collect** — `colab-tool status -s proj --json --files` lists what is on the VM
   (with the paths that work), then `colab-tool download -s proj <path> <local>`.
4. **Iterate** — edit the script locally and run step 2 on the *same* session:
   the runtime, the dataset and the installed packages all stay.
5. **Stop** — `colab-tool stop -s proj`. A T4 left running keeps burning the
   account's daily quota; this is the fastest way to exhaust it.

Remote paths: the contents API is rooted at the notebook root but the VM's
working directory is `/content`, so `model.json` and `content/model.json` name
the same file and both work.

## A successful tool call is NOT a successful run

**This is the most expensive mistake an agent makes here.** The usual failure:
launch the run, the tool returns successfully, the agent assumes it worked and
moves on — while the run actually died with a traceback in the output nobody
read. Then an hour of wall-clock and Colab quota is spent on a job that failed
in the first thirty seconds.

**The failure is almost always IN THE OUTPUT, not in the tool's return
value.** Read it, every time:

| You ran | What the return value tells you | Where the real failure hides |
|---|---|---|
| `colab-tool exec --json` | one envelope on stdout | `cells[].ok` and `cells[].outputs[]` — an `output_type: "error"` entry carries the traceback |
| MCP `colab_notebook_run_cell` | `[cell 0 id=… status=ok]` — **`status=ok` means the CALL worked, not that the cell did** | `colab_notebook_read --include_outputs` → look for `output_type: error`, a traceback |
| MCP `colab_notebook_run_all` | `"Ran 2 cell(s); 0 failed"` | check the count AND read the outputs |
| MCP `colab_job_start` | "Started job <id>" — **that is not a result at all** | `colab_job_logs` → scan for `Traceback`, `Error`, `CUDA out of memory`, `Killed`; and check `exit_code` |
| `colab-tool run` | the exit code | non-zero on an uncaught exception, but read the output too |

Concrete habit — after **every** run: (1) read the outputs, (2) scan for
`Traceback`, `Error`, `OutOfMemory`, `Killed`, and (3) look for the script's
**own success marker**. A training script that prints `TRAINING DONE` and one
that dies with a CUDA OOM both return a successful tool call; only the output
tells them apart. If the script has no explicit success marker, add one — it
is the cheapest insurance in the whole loop.

Corollary: if a run is supposed to take minutes, do not conclude it is still
running because the tool call returned. Poll it, and read the log each time.

## Moving big files: Google Drive

The contents API takes a whole file as one base64 request. Measured: **50 MB
works, 100 MB is aborted** by the host (`ConnectionAbortedError 10053`) — the
same limit upstream has on Linux. Uploads above 16 MB are now chunked and
the size is verified afterwards, so the CLI copes; for genuinely large
artefacts (multi-GB checkpoints, datasets) Drive is the better channel.

**Three Google accounts are mounted on this machine** (Google Drive for
Desktop, mirror mode):

| Drive | Account | Same as CLI account |
|---|---|---|
| **G:** | cinnybun02@gmail.com | `account2` |
| **H:** | sanskar9797@gmail.com | — |
| **I:** | stupidity97001@gmail.com | `default` |

Drive for Desktop is **not running by default** — start it if a drive letter
is missing:

```
& "C:\Program Files\Google\Drive File Stream\133.0.2.0\GoogleDriveFS.exe"
```

**Upload (local → VM):** put the file in the local Drive folder (e.g.
`I:\My Drive\Colab-MCP\data.bin`), then on the VM copy it out of the mounted
Drive into `/content`. The copy happens on the VM — no local transfer, no
base64, no size limit. Mounting Drive needs one human consent click
(`colab-tool drivemount`, or the MCP's `colab_drive_mount_start` /
`colab_drive_mount_finish` pair).

**Download (VM → local):** on the VM, copy the artefact into the mounted
Drive; Drive syncs it down to the local `G:`/`H:`/`I:` folder by itself.

**Never transfer packages or wheels** — `colab-tool install -s proj <pkg>`
installs on the VM directly.

| Size | Route |
|---|---|
| < 50 MB | `colab-tool upload` / `download` — direct |
| 50 MB – a few GB | chunked upload (built in) |
| a few GB and up | Google Drive |

## Safety
- **Always `colab-tool stop -s <name>` when done** — idle VMs burn compute units. `colab-tool run` (without `--keep`) self-cleans even if the script errors.
- Local state lives in `~/.config/colab-cli/sessions.json` (settings in `settings.json`, history in `history/*.jsonl`). Don't edit by hand.
- **Isolate parallel/agent runs** with the global `--config <path>` flag to point session state at a scratch file (e.g. `colab --config /tmp/agent.json new -s job`).

## Full training & experiments
- For training jobs and experiment sweeps, follow `docs/agent-training.md`:
  provision → environment → launch → monitor → checkpoint → retrieve → stop.
  Key points: kept sessions for inspectable long runs vs `colab-tool run` for
  self-cleaning one-shots; `--timeout` for long waits (or background launch +
  probe); download checkpoints BEFORE `stop`; `restart-kernel` before
  stop+recreate on wedges.

## Recovery
- "Session not found" / 404 / 401 on exec: the backend pruned the VM. `colab-tool exec`/`repl` detect this and clean up local state automatically — run `colab-tool sessions` and re-create with `colab-tool new`.
- Execution timeout or wedged kernel: `colab restart-kernel -s <name>` (keeps the VM, resets the kernel), or `colab-tool stop` then `colab-tool new`.
