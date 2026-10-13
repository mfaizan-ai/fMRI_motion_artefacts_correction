# Debugging on a compute node

Use the VS Code debugger to step through a Python script running on a Maguire compute node. Keep GPU and data-heavy work inside a Slurm allocation.

This guide assumes VS Code is connected to the login node through Remote SSH, with the repository open. The Python process runs on the compute node; VS Code attaches from the login node.

Direct attachment requires the login node to reach the compute node's debug port. Follow the cluster's access policy. Binding debugpy to `0.0.0.0` exposes it on all network interfaces; anyone who can reach the port can execute code in the debugged process. If direct attachment is not permitted, use the cluster's supported SSH-forwarding setup.

## One-time setup: configure VS Code

Install or enable the Python and Python Debugger extensions in the Remote SSH environment.

Create `.vscode/launch.json` in the repository root. If the file already exists, add the following configuration to its `configurations` list.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Slurm GPU",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "gpu02",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "${workspaceFolder}"
                }
            ],
            "justMyCode": false
        }
    ]
}
```

`gpu02` is an example. Replace it with the allocated node's hostname each session.

The identical path mappings assume the project is visible at the same absolute path on the login and compute nodes. If the paths differ, set `remoteRoot` to the compute-node project path. Here, `localRoot` means the workspace opened by VS Code on the login node.

`justMyCode: false` allows stepping into libraries such as PyTorch and Hydra. Use `true` to focus on your own code.

Keep machine-specific settings out of commits. Check the repository's `.gitignore` before assuming `.vscode/launch.json` is ignored.

## 1. Request resources — login-node terminal

In the VS Code terminal, request an interactive allocation:

```bash
salloc -N 1 --ntasks=1 --gpus=1 --cpus-per-task=32 --mem=48G --time=03:00:00
```

This example requests one node, one task, one GPU, 32 CPUs and 48 GB of memory for up to three hours. Adjust these resources to the script. Omit `--gpus=1` for a CPU-only task.

Wait until Slurm grants the allocation. Note its job ID for cleanup.

## 2. Enter the compute node — same terminal

Start a shell inside the allocation:

```bash
srun --jobid="$SLURM_JOB_ID" --ntasks=1 --pty bash
hostname
```

Check that `hostname` prints a compute node, such as `gpu02`, rather than the login node. Use this hostname in `launch.json`.

## 3. Activate the environment — compute-node terminal

```bash
source ~/.bashrc
conda activate moco
```

Change to the repository root if needed. Check the environment and, for GPU tasks, the allocated GPU:

```bash
python -c "import sys, debugpy; print(sys.executable); print(debugpy.__version__)"
nvidia-smi
```

The Python path should belong to the `moco` environment. If importing debugpy fails, add it through the project's environment setup before continuing.

## 4. Start the script — compute-node terminal

From the repository root, start denoising under debugpy:

```bash
python -Xfrozen_modules=off -m debugpy \
    --listen 0.0.0.0:5678 --wait-for-client \
    scripts/denoise.py \
    checkpoint=runs/stv6_convlstm_cycleGANS/best_model.pt \
    output_name=debug_denoise \
    output_root=runs/_checks/debug_denoise \
    runs=first_2mo \
    overwrite=true
```

Python starts debugpy and waits for VS Code before executing the target script. The terminal may remain quiet while it waits.

- Keep debug output in a dedicated folder such as `runs/_checks/debug_denoise`.
- Use `overwrite=true` only for a folder whose existing output you intend to reuse.
- Put Hydra overrides after the script name, as in a normal run.
- If port `5678` is occupied, choose another port and update `launch.json` to match.

To debug another script, replace `scripts/denoise.py` and its overrides. Use arguments supported by that script.

## 5. Attach — VS Code

1. Update `connect.host` in `.vscode/launch.json` to the hostname from step 2.
2. Check that `connect.port` matches the port from step 4.
3. Set a breakpoint on an executable line in the script or a function it calls.
4. Open **Run and Debug**, select **Attach to Slurm GPU**, and start debugging.

The target script begins executing and pauses when it reaches a breakpoint. Inspect variables and step through the section you need to investigate.

For later sessions, reuse this configuration and update the hostname and port as needed.

## 6. Stop the script and release the allocation

Disconnect the debugger. If Python is still running, press **Ctrl+C** in the compute-node terminal and wait for the shell prompt.

Then exit both shells:

```bash
exit  # leave the compute-node shell started by srun
exit  # leave the allocation shell started by salloc
```

To cancel the allocation explicitly, run `scancel` with the job ID recorded in step 1:

```bash
scancel <job-id>
```

Replace `<job-id>` with your allocation's numeric ID. Disconnecting VS Code alone does not release the Slurm allocation.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `hostname` shows the login node | Enter the allocation using the `srun` command in step 2. |
| `No module named debugpy` | Check the active Python environment and whether debugpy is installed there. |
| Connection refused | Check the hostname and port, and confirm the debugpy process is still running. |
| Connection timeout | Check whether the login node can reach the compute node's port and whether direct attachment is permitted. |
| `Address already in use` | Choose another port and update both the debugpy command and `launch.json`. |
| Breakpoints are hollow or never hit | Check path mappings, confirm VS Code shows the same source files as the target, and set breakpoints on executable lines. Use `--wait-for-client` for startup code. |
| No GPU is available | Check the allocation request and `nvidia-smi` on the compute node. |
| `CUDA out of memory` | Reduce the script's supported memory-demanding settings or check GPU memory use. Use CPU execution only if the script supports it. |
| Allocation remains active after debugging | Stop the Python process, exit the allocation shell, or cancel the recorded job ID with `scancel`. |

## References

- [VS Code: Python debugging](https://code.visualstudio.com/docs/python/debugging)
- [debugpy: command-line reference](https://github.com/microsoft/debugpy/wiki/Command-Line-Reference)
- [debugpy: remote listening and script arguments](https://github.com/microsoft/debugpy/blob/main/README.md)
- [Slurm: salloc](https://slurm.schedmd.com/salloc.html)
- [Slurm: srun](https://slurm.schedmd.com/srun.html)
- [Slurm: scancel](https://slurm.schedmd.com/scancel.html)