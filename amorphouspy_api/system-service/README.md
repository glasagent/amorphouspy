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