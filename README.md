# nemo-gym-tenki

A [Tenki](https://tenki.cloud) sandbox provider for [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym).

Runs each NeMo Gym sandbox in a **Tenki Firecracker microVM**: it boots in seconds, is kernel-isolated, and is thrown away when the rollout ends. It sits alongside NeMo Gym 0.7.0's built-in providers (`apptainer`, `daytona`, `docker`, `e2b`, `ecs_fargate`, `enroot`, `local`, `opensandbox`, `openshell`) as a hosted-cloud option, and is a strong fit for **RL rollout collection**, where you fan out many short-lived sandboxes to run an agent's generated code for verifiable rewards (coding, math, tool-use).

## Install

Not on PyPI yet. Install it from this repo at a pinned commit, into the same environment as NeMo Gym (0.7.0 or later, which needs Python 3.13.14+):

```bash
pip install "nemo-gym>=0.7.0" "nemo-gym-tenki @ git+https://github.com/opencolin/nemo-gym-tenki@<commit>"
```

Installing registers a `tenki` provider via the `nemo_gym.sandbox_providers` entry point — no code changes to NeMo Gym.

## Use it

Select the provider in any NeMo Gym agent/environment config that takes a `sandbox_provider`:

```yaml
sandbox_provider:
  tenki:
    connection: {}                 # auth from TENKI_API_KEY; workspace inferred from the key
    create:
      cpu_cores: 2
      memory_mb: 4096
      disk_size_gb: 10
      # image: "your-registry/ref"   # optional Tenki registry image; default is the platform base
```

Programmatically:

```python
from nemo_gym.sandbox.providers.registry import create_provider
from nemo_gym.sandbox.providers.base import SandboxSpec

provider = create_provider({"tenki": {"create": {"cpu_cores": 2, "memory_mb": 4096}}})
handle = await provider.create(SandboxSpec())
result = await provider.exec(handle, "python3 -c 'print(6*7)'")   # -> return_code 0, stdout "42"
await provider.close(handle)
await provider.aclose()
```

## Authentication

The provider constructs a Tenki SDK `Client` (the `tenki` package, imported as `tenki_sandbox`). Auth resolves in order:

1. `connection.auth_token` in the config, else
2. `TENKI_API_KEY` / `TENKI_AUTH_TOKEN` in the environment. The SDK accepts only **`tk_` API keys**.

The SDK doesn't read the file `tenki login` writes. (`verify.py` reads `auth_token` from `~/.config/tenki/config.yaml` itself and passes it in, as a convenience.)

**Set a lifetime cap.** `create.max_duration_s`, or a spec's `ttl_s`, becomes the VM's hard `max_duration`. With neither set there is no cap, so a rollout whose `close()` never runs leaves its VM running.

`connection.workspace_id` is **optional**: omit it and the SDK uses the workspace your key belongs to, which is why this works in CI with only a key set.

## What it supports

Implements the full `SandboxProvider` protocol — `create`, `exec` (native `cwd` / `env` / `timeout`), `upload_file`, `download_file`, `status`, `close`, `aclose` — plus the optional **`ConnectableProvider`** capability: Tenki sandboxes are reachable by id and support pause/resume, so a handle can be `serialize_handle`'d and rebuilt with `connect` in another process. That's what lets a Tenki sandbox be shared across a distributed rollout without fronting it with a sandbox server (the local `docker`/`enroot`/`apptainer` providers can't do this).

Config maps to the Tenki SDK: `SandboxSpec.resources.{cpu,memory_mib,disk_gib}` → `cpu_cores`/`memory_mb`/`disk_size_gb`; `spec.image` → Tenki image; `spec.env`/`spec.metadata` pass through; `spec.ttl_s` → the VM's hard `max_duration`; `spec.files` are written after create.

## Scope: CPU only

Tenki runs **CPU** microVMs. A `SandboxSpec` that requests `resources.gpu` is rejected with a clear error — use a GPU-capable provider for GPU environments. Tenki targets the CPU code/command-execution case (coding, math, tool-use terminal/SWE agents), which is the bulk of NeMo Gym's sandbox workload. (Outgrowing that into GPU compute is where [Luxor](https://luxor.ai) comes in.)

Note: outbound network is on by default, so agents that `pip install` or `git clone` work out of the box. Set `create.allow_outbound: false` to lock a rollout down.

## Verify

```bash
export TENKI_API_KEY=tk_...
python verify.py      # create -> exec 42 -> files -> connect roundtrip -> terminate, against live Tenki
```

Licensed Apache-2.0.
