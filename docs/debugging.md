# Debugging on a compute node

Step through a script in the VS Code debugger while it runs on a Maguire compute node. Don't debug on
the login node: the scripts load large checkpoints, use the GPU and read real data.

The script runs on the compute node under `debugpy`, listening on a port. VS Code, connected to the
login node, attaches to that node and port.

## 1. Get a compute node

Request an interactive allocation (1 GPU, 32 CPUs, 48 GB, 3 h):

```bash
salloc -N 1 --gpus=1 --cpus-per-task=32 --mem=48G --time=03:00:00
```

Adjust the resources to the task; for pure logic without the GPU, drop `--gpus=1`. A personal wrapper
script around this `salloc` works too.

Once the allocation is granted, move your shell onto the node:

```bash
srun --jobid=$SLURM_JOBID --pty bash
hostname          # must print a compute node, e.g. gpu02, not the login node; note it for step 4
```

## 2. Activate the environment

```bash
source ~/.bashrc && conda activate moco
```

## 3. Start the script under debugpy

The script starts, then waits for the debugger before running any code.

**Generic form**, from the project root:

```bash
python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    scripts/<script>.py <hydra overrides>
```

**Example: denoising**

```bash
python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    scripts/denoise.py checkpoint=runs/stv6_convlstm_cycleGANS/best_model.pt \
    output_root=runs/_checks/debug_denoise overwrite=true
```

- Point the output at `runs/_checks/debug_<script>`, never at the shared derivatives folder.
  `overwrite=true` lets you rerun into the same folder.
- Hydra overrides go after the script name exactly as in a normal run; debugpy passes them through.
- `-Xfrozen_modules=off` stops debugpy warning that breakpoints in the standard library may be missed.
- If port 5678 is already in use on that node, for example by another session, pick another such as
  5679 and use the same number in step 4.

## 4. Attach VS Code

Put this in `.vscode/launch.json` (local only; `.vscode/` is git-ignored). Set `host` to the node
from `hostname` and `port` to the one in step 3:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Slurm GPU",
            "type": "debugpy",
            "request": "attach",
            "connect": {"host": "gpu02", "port": 5678},
            "pathMappings": [
                {"localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}"}
            ],
            "justMyCode": false
        }
    ]
}
```

Set breakpoints, open **Run and Debug** in the left sidebar, choose **Attach to Slurm GPU** and press
Start. The waiting script continues and stops at your first breakpoint.

### `launch.json` reference

**Where it lives:** `<project root>/.vscode/launch.json`. If it doesn't exist, open **Run and Debug**
and click "create a launch.json file", or create the folder and file by hand. It is git-ignored, so
each machine or user keeps their own copy.

| Setting | Value | What it does |
|---|---|---|
| `version` | `"0.2.0"` | Schema version of `launch.json`; always this value |
| `configurations` | list | One entry per debug target; each appears by `name` in the Run and Debug dropdown |
| `name` | `"Attach to Slurm GPU"` | Label shown in the dropdown; any text |
| `type` | `"debugpy"` | Use the Python debugger (needs the Python / Python Debugger extensions) |
| `request` | `"attach"` | Connect to a program that is already running and waiting (step 3), instead of `"launch"`, which would start it on the login node |
| `connect.host` | `"gpu02"` | Node running the script; **change it each session** to the `hostname` from step 1 |
| `connect.port` | `5678` | Must equal the port in `--listen 0.0.0.0:<port>` |
| `pathMappings` | `${workspaceFolder}` → `${workspaceFolder}` | Translates file paths between VS Code and the node, so breakpoints land in the right file. The same on both sides because both see the same Lustre paths |
| `justMyCode` | `false` | Let breakpoints and stepping go into library code (torch, nibabel, Hydra); `true` stays in project files only |

**Updating it each session:** only `connect.host` usually changes, and occasionally `connect.port`.
To debug several jobs at once, add one configuration per node or port, each with its own `name`:

```json
{"name": "Attach gpu03:5679", "type": "debugpy", "request": "attach",
 "connect": {"host": "gpu03", "port": 5679},
 "pathMappings": [{"localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}"}],
 "justMyCode": false}
```

## 5. Finish

Stop the debugger once you have seen what you need. Most scripts loop over every subject or run, and
there is no need to let them finish. Then give the node back:

```bash
exit    # leave the node shell (srun)
exit    # end the allocation (salloc); otherwise it holds the GPU until the 3 h limit
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| VS Code: "connection refused" or timeout | The `host` or `port` in `launch.json` doesn't match `hostname` / `--listen`, or the script has exited |
| `hostname` shows the login node | You skipped `srun --jobid=$SLURM_JOBID --pty bash` |
| `Address already in use` | Port taken on that node; use another and update `launch.json` |
| Breakpoints are hollow / not hit | Attach before the code runs (`--wait-for-client`), and set breakpoints in the files under the project root |
| `CUDA out of memory` or no GPU | Check `nvidia-smi` on the node; for pure logic, add `device=cpu` (slow but fine for stepping) |
