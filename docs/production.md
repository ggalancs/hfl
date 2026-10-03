# Running HFL in production

How to run `hfl serve` as a service other people or programs depend on.
Each recipe here was run as written (2026-10-02), on the setup named with
it; what was not run says so.

## Before you start

- **One process holds the models.** HFL keeps the loaded models in its own
  memory and its registry on one disk: run one instance per machine (or per
  volume), never several replicas behind a load balancer.
- **Several requests at once** on GGUF models come from llama.cpp's
  `llama-server`. The Docker image and the installers carry it; with pip,
  run `hfl install llama-server` once. `hfl serve` then serves each GGUF
  model with 4 slots (`HFL_NUM_PARALLEL` changes it; `1` keeps llama.cpp in
  process, one request at a time).
- **Models are provisioned by the machine's owner**: `hfl pull` on the host
  (or `docker exec … hfl pull`, `kubectl exec … -- hfl pull`). API clients
  cannot pull, push or delete models unless `HFL_ALLOW_REMOTE_PULL=true`.

## As a service

### Linux: systemd

Tested in Debian 13 (trixie) with systemd as PID 1. Install HFL in its own
venv, as its own user:

```bash
sudo useradd --system --home /var/lib/hfl --create-home hfl
sudo python3 -m venv /opt/hfl
sudo /opt/hfl/bin/pip install hfl
sudo -u hfl HFL_HOME=/var/lib/hfl /opt/hfl/bin/hfl install llama-server --yes
sudo -u hfl HFL_HOME=/var/lib/hfl /opt/hfl/bin/hfl pull Qwen/Qwen2.5-7B-Instruct-GGUF
```

`/etc/hfl/hfl.env` (`chmod 600`, owned by root — it holds the key):

```bash
HFL_HOME=/var/lib/hfl
HFL_HOST=127.0.0.1
HFL_PORT=11434
HFL_API_KEY=change-me-to-a-long-random-key
# With a reverse proxy on this machine forwarding the public name (below):
# HFL_ORIGINS=https://hfl.example.com
```

`/etc/systemd/system/hfl.service`:

```ini
[Unit]
Description=HFL — HuggingFace models, served locally
After=network-online.target
Wants=network-online.target

[Service]
User=hfl
Group=hfl
EnvironmentFile=/etc/hfl/hfl.env
ExecStart=/opt/hfl/bin/hfl serve
Restart=on-failure
RestartSec=5
# HFL stops its llama-server children on SIGTERM; give a reply in flight time.
TimeoutStopSec=60
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/hfl

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now hfl
journalctl -u hfl -f
```

Checked: it starts and answers (401 without the key, 200 with it), serves
through `llama-server` with 4 slots, logs to the journal; killed with
`kill -9`, systemd restarts it in 5 s and the old `llama-server` does not
survive; `systemctl stop` leaves no process behind.

To listen on the network instead of `127.0.0.1`, put a reverse proxy in
front (below). Binding a public address directly needs
`HFL_ACCEPT_NETWORK_EXPOSURE=true` (a service has no terminal to confirm
in) and always a key.

### macOS: launchd

`~/Library/LaunchAgents/com.github.ggalancs.hfl.plist` (replace `YOU`, and
the path with `which hfl`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.github.ggalancs.hfl</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.local/bin/hfl</string>
    <string>serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HFL_HOST</key><string>127.0.0.1</string>
    <key>HFL_PORT</key><string>11434</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>/Users/YOU/Library/Logs/hfl.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/Library/Logs/hfl.log</string>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.github.ggalancs.hfl.plist
```

A user agent (not a system daemon): Metal is available to the logged-in
user's processes. `ProcessType Interactive` keeps macOS from throttling it.
Checked: the file passes `plutil -lint`; loading it was not tested here.
On a laptop, generation is several times slower on battery than on power.

### Windows

The MSI puts `hfl` on the PATH and registers no service. No recipe for
running it as a service has been tested yet.

## Docker and Compose

The image runs `hfl serve` on port 11434 as user 1000, keeps everything in
`/var/lib/hfl`, and carries `llama-server`. Tested with Compose, with Caddy
in front for TLS:

`compose.yaml`:

```yaml
services:
  hfl:
    image: ghcr.io/ggalancs/hfl:latest
    restart: unless-stopped
    environment:
      HFL_API_KEY: ${HFL_API_KEY:?set HFL_API_KEY}
      HFL_KEEP_ALIVE: 30m
    volumes:
      - hfl-data:/var/lib/hfl
    # No ports: only the proxy reaches it.
  proxy:
    image: caddy:2
    restart: unless-stopped
    ports:
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
volumes:
  hfl-data:
  caddy-data:
```

`Caddyfile` (Caddy gets the certificate for the domain by itself):

```
hfl.example.com {
	reverse_proxy hfl:11434 {
		flush_interval -1
	}
}
```

```bash
HFL_API_KEY=… docker compose up -d
docker compose exec hfl hfl pull Qwen/Qwen2.5-7B-Instruct-GGUF
```

Checked (with `tls internal` for a local test): 401 without the key, 200
with it, tokens streamed as they come, `/api/pull` from a client refused
(403), the container healthy, models kept across `down` and `up`.

## Kubernetes

Tested on k3s 1.31. One replica, `Recreate` (the volume is
`ReadWriteOnce`: two pods at once would share it).

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hfl
stringData:
  api-key: change-me-to-a-long-random-key
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hfl-data
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hfl
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels: {app: hfl}
  template:
    metadata:
      labels: {app: hfl}
    spec:
      # Kubernetes otherwise sets HFL_PORT=tcp://… for a Service named hfl,
      # which HFL before 0.26 refused at start.
      enableServiceLinks: false
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: hfl
          image: ghcr.io/ggalancs/hfl:latest
          ports:
            - containerPort: 11434
          env:
            - name: HFL_API_KEY
              valueFrom:
                secretKeyRef: {name: hfl, key: api-key}
            - name: HFL_KEEP_ALIVE
              value: 30m
          resources:
            requests: {memory: 8Gi, cpu: "4"}
            limits: {memory: 16Gi}
          volumeMounts:
            - name: data
              mountPath: /var/lib/hfl
          startupProbe:
            httpGet: {path: /healthz, port: 11434}
            failureThreshold: 30
            periodSeconds: 2
          livenessProbe:
            httpGet: {path: /healthz, port: 11434}
            periodSeconds: 30
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: hfl-data
---
apiVersion: v1
kind: Service
metadata:
  name: hfl
spec:
  selector: {app: hfl}
  ports:
    - port: 11434
      targetPort: 11434
```

```bash
kubectl exec deploy/hfl -- hfl pull Qwen/Qwen2.5-7B-Instruct-GGUF
```

Checked (with smaller requests and limits): the pod becomes ready, the
Service answers by name (401 without the key, 200 with it), a model pulled
in the pod answers and is still there after the pod is replaced.
`/healthz` needs no key, so the probes work with one set.

## Security

- **A key, always, beyond this machine.** `HFL_API_KEY` (in the
  environment or a secret, not `--api-key` on the command line, which other
  local users can read in the process list). Clients send
  `Authorization: Bearer <key>` or `X-API-Key`.
- **TLS from a reverse proxy.** HFL speaks plain HTTP. With the proxy on the
  same machine, every request reaches HFL from loopback, which is the
  machine's owner: the proxy must say whom it relays, with
  `X-Forwarded-For` (Caddy does by default). HFL also recognises
  `Forwarded` and `X-Real-IP` (since 0.26); a proxy that sends none of them
  passes its clients through as the owner. nginx, checked in front of the
  image (403 for a client's `/api/pull`, tokens streamed):

  ```nginx
  location / {
      proxy_pass http://127.0.0.1:11434;
      proxy_http_version 1.1;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header Host $host;
      proxy_buffering off;          # tokens as they come
      proxy_read_timeout 600s;      # long generations
      proxy_set_header Upgrade $http_upgrade;   # /ws/chat
      proxy_set_header Connection "upgrade";
      client_max_body_size 0;       # HFL caps bodies itself
  }
  ```

- **Host names**: a server bound to loopback answers only to `localhost`,
  `127.0.0.1`, `::1` and the names in `HFL_ORIGINS` (a web page cannot reach
  it through DNS rebinding). A reverse proxy on the same machine that
  forwards the public `Host` — nginx with `proxy_set_header Host $host`
  as above, Caddy by default — needs that name in `HFL_ORIGINS`
  (`HFL_ORIGINS=https://hfl.example.com`), or every request gets 403.
- **Browsers**: a web UI on another origin needs it in `HFL_ORIGINS`.
- **Rate limit**: 60 requests a minute per client by default
  (`HFL_RATE_LIMIT_*`); requests from this machine are exempt unless
  `HFL_RATE_LIMIT_LOCAL=true`.
- **Model code**: `HFL_ALLOW_REMOTE_CODE` stays off unless you trust every
  model you serve.

## Memory

- `HFL_MEMORY_BUDGET` (default 85 % of RAM) decides how many models stay
  loaded: before each load HFL measures the machine and unloads idle models,
  least recently used first, to stay under it. In a container it plans
  within the container's limit (cgroup), not the host's memory.
  (Before 0.26 a pip install or the image could lack `psutil`, and then
  nothing was checked; the log line "Loading … budget …" shows the check
  runs.)
- `HFL_MAX_LOADED_MODELS` adds a ceiling on the count.
- `HFL_KEEP_ALIVE` (default `5m`) unloads a model idle that long; longer
  keeps it ready, at the cost of memory.
- Several requests at once on one model share its context: each of the 4
  slots gets a quarter of `num_ctx` unless set.

## Watching it

- `GET /healthz` — liveness, no key needed. `/health/ready` and
  `/health/deep` say more.
- `GET /metrics` — Prometheus, needs the key:

  ```yaml
  scrape_configs:
    - job_name: hfl
      authorization:
        credentials: change-me-to-a-long-random-key
      static_configs:
        - targets: ["hfl:11434"]
  ```

  What it exports is in [metrics.md](metrics.md).
- Logs: `hfl serve --json-logs` for a log collector; `HFL_DEBUG=1` for more.
  `HFL_AUDIT_LOG_PATH` keeps an audit trail of administrative actions.
- Traces: `HFL_OTEL_ENABLED=1` and `HFL_OTEL_EXPORTER_ENDPOINT` (with
  `pip install "hfl[otel]"`).
- The `X-Queue-Depth` response header and `/healthz` show the queue.

## Upgrades and backups

- Upgrade with `pip install -U hfl` (then `hfl install llama-server` if a
  release moves its pinned build) or a newer image; restart the service.
  The registry and models stay; what is stable across versions is in
  [stability.md](stability.md).
- Back up `HFL_HOME/models.json` (HFL keeps `models.json.bak` beside it). The
  model files can be downloaded again; back them up only if the download
  matters.

## When something goes wrong

| What you see | What it means |
|---|---|
| `401` | No key, or the wrong one. |
| `403` with `remote_admin_forbidden` | A client asked for an owner operation (pull, push, delete): run it on the host, or set `HFL_ALLOW_REMOTE_PULL=true`. |
| `403` with `cross_origin_admin_forbidden` | A web page asked for an owner operation. |
| `429` with `Retry-After` | The queue is full, or the client is over the rate limit: retry after the seconds given. |
| `503` | No slot freed in time for the request; the server is saturated. |
| "Not enough memory to load … on this server" (`MemoryBudgetExceededError`) | The model, with its context, does not fit the memory budget even with the idle models unloaded: a smaller quantization, a shorter `num_ctx`, or a higher `HFL_MEMORY_BUDGET`. The server's log (and a request from the host itself) gives the figures. |
| "Cannot load … yet: the memory it needs is in use" (`ModelsBusyError`) | Room needs a model other requests are using: retry shortly. |
| "Parallel requests need llama.cpp's llama-server, which is not installed" | `hfl install llama-server`, or `HFL_NUM_PARALLEL=1` for one request at a time in process. |
| Kubernetes: "Invalid value for 'HFL_PORT': tcp://…" | HFL before 0.26 and a Service named `hfl`: `enableServiceLinks: false`. |
| Generation much slower than expected on a Mac | On battery: macOS slows the GPU several times; `hfl doctor` shows the power source. |
