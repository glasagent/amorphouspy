# Install API as a system service
As `root` (or via `sudo`)
```
# Copy the unit files
cp amorphouspy-api.service /etc/systemd/system/
cp amorphouspy-api.path /etc/systemd/system/

# Enable (start on boot) and start
systemctl daemon-reload
systemctl enable --now amorphouspy-api.service
systemctl enable --now amorphouspy-api.path
systemctl restart amorphouspy-api

```

Check status
```
systemctl status amorphouspy-api
```

View logs
```
journalctl -u amorphouspy-api -f
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `EXECUTOR_TYPE` | `local` | Executor backend (`local` or `slurm`) |
| `SLURM_PARTITION` | — | Slurm partition for job submission |
| `LAMMPS_CORES` | `1` | Number of cores for LAMMPS simulations |
| `AMORPHOUSPY_PROJECTS` | `<package>/projects` | Root directory for project data and the SQLite database |
| `AMORPHOUSPY_VERSION_PROJECTS` | `1` | Set to `0`/`false`/`no` to disable version-based project subdirectories. When enabled (default), cache is stored under `amorphouspy_<version>/meltquench/`; when disabled, directly under `meltquench/`. Disable this to keep cached results accessible across version upgrades. |