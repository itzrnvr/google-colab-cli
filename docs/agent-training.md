# Agent Training & Experiment Playbook

Run full training jobs and experiment sweeps on Colab VMs from a program or AI
agent. This is the companion to `colab agent-guide` (operator handbook): that
page covers auth, sessions, and the machine interface; this page covers the
training lifecycle end to end.

## The lifecycle

```
provision → environment → launch → monitor → checkpoint → retrieve → stop
```

Every step below is a CLI command. Nothing requires a TTY except `drivemount`
and VM-side `auth` (both need a human); everything else is agent-runnable.

## 1. Provision

```bash
colab new -s exp1 --gpu A100            # or T4 / L4 / --tpu v6e1, or CPU
colab status -s exp1 --json             # confirm hardware + IDLE before spending
```

- Always pass `-s <name>`: omitted names are random hex, unmanageable later.
- A `400` on `new` with an accelerator = no quota/entitlement on this account.
  Fall back to `--gpu T4` or CPU. Most accounts can only get CPU — never assume.
- `--high-mem` needs Pro/Pro+ (ignored for L4/TPU single-shape runtimes).
- Parallel experiments: isolate state per run with
  `colab --config /tmp/exp1.json new -s exp1` so concurrent agents never race
  on `sessions.json`.

## 2. Environment

```bash
colab install -s exp1 torch transformers datasets    # uv, falls back to pip
colab install -s exp1 -r requirements.txt            # or a requirements file
colab exec -s exp1 -f env_check.py --json            # verify: torch, cuda, versions
```

- Kernel state persists across `exec` calls in one session: imports, variables,
  and functions survive. Build incrementally; don't reinstall every step.
- Set run config via `--env KEY=VALUE` (repeatable) on `exec`/`run`, or export
  inside the script. Secrets go through `--env`, never baked into files.
- Need Drive data? `colab drivemount -s exp1` (human step). Need GCP from inside
  the VM? `colab auth -s exp1` (human step). Schedule these before handing off
  to the agent.

## 3. Launch

Two patterns. Pick one deliberately.

**Long training on a kept session** (full control, inspectable mid-run):

```bash
colab exec -s exp1 -f train.py --stream-jsonl --errors-only
```

- The command stays attached and streams one JSON object per kernel message.
  An `error` output is the instant failure signal; the process exits 1.
- Large logs arrive as preview + `ref`; pull full text only for what matters:
  `colab output -s exp1 o7`.
- The kernel keeps running between `exec` calls, so you can probe mid-training
  from a second command (`colab exec -s exp1` with a checkpoint-inspection
  snippet) without disturbing the run.

**Ephemeral one-shot** (fire-and-forget, self-cleaning):

```bash
colab run --gpu A100 train.py -- --epochs 10 --lr 3e-4
```

- `new` + `exec` + `stop` in one command. Exit code follows CPython
  (`sys.exit(N)` → N, uncaught exception → 1). `--keep` preserves the VM for
  follow-up inspection; without it the VM dies even when the script errors.
- `script_args` after the path become `sys.argv[1:]`, `__name__ == "__main__"`.

Timeouts: `--timeout` (default 30s) bounds each `exec`/`run` call's wait for
kernel messages. For multi-hour training, either raise it
(`--timeout 3600`) or — better — launch training as a background process on
the VM (nohup-style via `console`/shell) and poll checkpoints with short
`exec` probes. A wedged kernel: `colab restart-kernel -s exp1` (VM kept).

## 4. Monitor

```bash
colab status -s exp1 --json        # BUSY(task) vs IDLE; last execution stamp
colab exec -s exp1 --stream-jsonl --errors-only -f probe_loss.py
colab log -s exp1 -n 5            # recent structured events, human-readable
```

- Prefer small probe scripts over re-reading full logs: query the checkpoint
  dir, tail the log file on the VM, emit one JSON line of metrics.
- `colab log -s exp1 -o report.md` exports the session narrative for the final
  write-up; `-o run.ipynb` captures code + outputs as a notebook.

## 5. Checkpoint & retrieve

```bash
colab exec -s exp1 -f save_ckpt.py --json            # write /content/ckpts/*
colab ls -s exp1 /content/ckpts                      # confirm artifacts exist
colab download -s exp1 /content/ckpts/model.bin ./model.bin
```

- Write checkpoints to `/content/...` (absolute paths; every command starts
  in `/content` but absolute is unambiguous).
- Download artifacts **before** `stop`: stopping releases the VM and its disk.
- Large artifacts: download incrementally between epochs rather than once at
  the end — a pruned VM loses everything not yet retrieved.

## 6. Stop

```bash
colab stop -s exp1
```

Idle VMs burn compute units. `colab run` without `--keep` self-cleans; kept
sessions and `new` sessions must be stopped explicitly. Verify with
`colab sessions --json`.

## Recovery

- 404/401 or "Session not found" on exec: the backend pruned the VM. Local
  state auto-cleans; re-create with `colab new` and resume from the last
  downloaded checkpoint.
- Wedged kernel, runaway process: `colab restart-kernel -s exp1` first (cheap),
  `stop` + `new` second (loses VM disk — checkpoints must already be local).
- OOM / CUDA errors: they arrive as `error` outputs with full tracebacks
  fetchable by ref. Reduce batch, enable grad checkpointing, retry the cell —
  the session survives cell failures.
- Lost local state (`sessions.json` deleted): `colab sessions` re-syncs from
  the server; orphans show as `[?]`.

## Minimal full-training example

```bash
colab new -s ft --gpu T4
colab install -s ft torch datasets
echo 'import torch; print(torch.cuda.get_device_name(0))' | colab exec -s ft --json
colab exec -s ft -f train.py --stream-jsonl --errors-only
colab download -s ft /content/ckpts/best.pt ./best.pt
colab log -s ft -o training-report.md
colab stop -s ft
```
