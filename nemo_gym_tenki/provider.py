# Copyright (c) 2026 Luxor AI. Apache-2.0.
"""Tenki sandbox provider for NVIDIA NeMo Gym.

Runs each NeMo Gym sandbox in a Tenki Firecracker microVM: boots in seconds, per-second
billing, disposable, kernel-isolated. It implements the ``SandboxProvider``
protocol (``nemo_gym.sandbox.providers.base``) plus the optional
``ConnectableProvider`` capability -- Tenki sandboxes are reachable by id and
support pause/resume, so a handle can be serialized and rebuilt in another
process (useful for distributed rollout collection), which the purely-local
providers (docker/enroot/apptainer) cannot do.

Select it from a NeMo Gym agent config::

    sandbox_provider:
      tenki:
        connection: {}            # auth from TENKI_API_KEY; workspace inferred from the key
        create: {cpu_cores: 2, memory_mb: 4096}

The Tenki SDK (``tenki``, imported as ``tenki_sandbox``) is synchronous, so every SDK call runs in a
worker thread via ``asyncio.to_thread`` to satisfy the async provider protocol.

Scope: Tenki runs **CPU** microVMs. It does not attach GPUs -- a spec that
requests ``resources.gpu`` is rejected with a clear error (use a GPU-capable
provider for those environments). This covers the CPU code/command-execution
case: coding, math, and tool-use terminal/SWE agents.
"""
from __future__ import annotations

import asyncio
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)

try:  # coerce_config normalizes a plain YAML mapping into a config dataclass.
    from nemo_gym.sandbox.providers.utils import coerce_config as _coerce_config
except Exception:  # pragma: no cover - fallback if the helper moves upstream

    def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
        if value is None:
            return config_cls()
        if isinstance(value, config_cls):
            return value
        if isinstance(value, Mapping):
            return config_cls(**value)
        raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


@dataclass(frozen=True)
class TenkiConnectionConfig:
    """How to reach Tenki.

    All fields are optional. When ``auth_token`` is unset the SDK reads
    ``TENKI_API_KEY`` / ``TENKI_AUTH_TOKEN`` from the environment (a ``tk_`` API
    key is preferred). When ``workspace_id`` is unset the SDK uses the workspace
    the key belongs to -- so this runs in CI with only a key set.
    """

    auth_token: str | None = None
    base_url: str | None = None
    workspace_id: str | None = None
    request_timeout_s: float | None = None


@dataclass(frozen=True)
class TenkiCreateConfig:
    """Defaults for sandbox creation; a ``SandboxSpec`` overrides these per call."""

    cpu_cores: int = 1
    memory_mb: int = 2048
    disk_size_gb: int = 10
    image: str | None = None  # a Tenki registry ref; None -> platform default base image
    ready_timeout_s: float = 180.0
    allow_outbound: bool = True  # agents that pip install / git clone need egress; set False to lock a rollout down
    idle_timeout_minutes: int | None = None  # ignored by the tenki 1.x SDK (sandboxes don't pause on idle)
    max_duration_s: float | None = None  # hard VM lifetime; None -> no cap, so set it or spec.ttl_s
    sticky: bool = False

    def __post_init__(self) -> None:
        if self.cpu_cores < 1:
            raise ValueError("create.cpu_cores must be >= 1")
        if self.memory_mb < 256:
            raise ValueError("create.memory_mb must be >= 256")
        if self.disk_size_gb < 1:
            raise ValueError("create.disk_size_gb must be >= 1")
        if self.ready_timeout_s <= 0:
            raise ValueError("create.ready_timeout_s must be > 0")


@dataclass(frozen=True)
class TenkiExecConfig:
    shell: str = "bash"  # commands run as ``<shell> -lc <command>``
    default_timeout_s: float | None = 180.0

    def __post_init__(self) -> None:
        if not self.shell:
            raise ValueError("exec.shell must be a non-empty shell name/path")
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")


_ALIVE = {"running"}
_STARTING = {"starting", "creating", "pending", "resuming"}
_STOPPED = {"paused", "pausing", "stopped", "terminating", "terminated", "user_shutdown"}


def _to_status(state: str | None) -> SandboxStatus:
    s = str(state or "").lower()
    if s in _ALIVE:
        return SandboxStatus.RUNNING
    if s in _STARTING:
        return SandboxStatus.STARTING
    if s in _STOPPED:
        return SandboxStatus.STOPPED
    if s == "error":
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


class TenkiProvider:
    """SandboxProvider backed by the Tenki SDK (``tenki``)."""

    name = "tenki"

    def __init__(
        self,
        *,
        connection: TenkiConnectionConfig | Mapping[str, Any] | None = None,
        create: TenkiCreateConfig | Mapping[str, Any] | None = None,
        exec: TenkiExecConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, TenkiConnectionConfig)
        self._create = _coerce_config(create, TenkiCreateConfig)
        self._exec = _coerce_config(exec, TenkiExecConfig)
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()

    # -- SDK client (built lazily in a thread; the SDK is synchronous) -------
    async def _client_handle(self) -> Any:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = await asyncio.to_thread(self._build_client)
        return self._client

    def _build_client(self) -> Any:
        from tenki_sandbox import Client

        kwargs: dict[str, Any] = {}
        if self._connection.auth_token is not None:
            kwargs["auth_token"] = self._connection.auth_token
        if self._connection.base_url is not None:
            kwargs["base_url"] = self._connection.base_url
        if self._connection.request_timeout_s is not None:
            kwargs["timeout"] = self._connection.request_timeout_s
        return Client(**kwargs)

    # -- create -------------------------------------------------------------
    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        res = spec.resources
        if getattr(res, "gpu", None):
            raise SandboxCreateError(
                "The Tenki provider runs CPU Firecracker microVMs and does not attach GPUs "
                "(SandboxSpec.resources.gpu was set). Use a GPU-capable provider for GPU "
                "environments; Tenki covers CPU code/command execution (coding, math, tool-use)."
            )
        client = await self._client_handle()
        cpu = int(math.ceil(res.cpu)) if res.cpu else self._create.cpu_cores
        mem = int(res.memory_mib) if res.memory_mib else self._create.memory_mb
        disk = int(res.disk_gib) if res.disk_gib else self._create.disk_size_gb
        timeout = float(spec.ready_timeout_s) if spec.ready_timeout_s else self._create.ready_timeout_s
        sandbox = await asyncio.to_thread(self._create_sandbox, client, spec, cpu, mem, disk, timeout)
        if spec.workdir or spec.files:
            await asyncio.to_thread(self._provision, sandbox, spec.workdir, spec.files)
        return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox)

    def _create_sandbox(self, client: Any, spec: SandboxSpec, cpu: int, mem: int, disk: int, timeout: float) -> Any:
        kwargs: dict[str, Any] = dict(
            cpu_cores=cpu,
            memory_mb=mem,
            disk_size_gb=disk,
            allow_outbound=self._create.allow_outbound,
            sticky=self._create.sticky,
            wait=True,
            timeout=timeout,
        )
        if self._connection.workspace_id:
            kwargs["workspace_id"] = self._connection.workspace_id
        image = spec.image or self._create.image
        if image:
            kwargs["image"] = image
        if spec.env:
            kwargs["env"] = dict(spec.env)
        if spec.metadata:
            kwargs["metadata"] = dict(spec.metadata)
        ttl = spec.ttl_s if spec.ttl_s is not None else self._create.max_duration_s
        if ttl is not None:
            kwargs["max_duration"] = float(ttl)
        if self._create.idle_timeout_minutes is not None:
            kwargs["idle_timeout_minutes"] = self._create.idle_timeout_minutes
        try:
            return client.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - normalize to the provider's create error
            raise SandboxCreateError(f"Tenki sandbox create failed: {type(e).__name__}: {e}") from e

    # NOTE ON FILE I/O: the Tenki `fs` API is workdir-scoped and rejects absolute
    # paths ("path outside workdir"). Provider callers pass arbitrary (often
    # absolute) targets, so we transfer files through the shell via exec + stdin /
    # stdout instead -- binary-safe (CommandResult.stdout is raw bytes) and path-
    # agnostic, the same way a terminal agent writes files.
    def _provision(self, sandbox: Any, workdir: str | None, files: Mapping[str, str]) -> None:
        if workdir:
            r = sandbox.exec(self._exec.shell, "-lc", f"mkdir -p {shlex.quote(workdir)}", check=False)
            if r.exit_code != 0:
                raise SandboxCreateError(f"could not create workdir {workdir!r}: {r.stderr_text}")
        prefix = f"cd {shlex.quote(workdir)} && " if workdir else ""
        for path, content in (files or {}).items():
            data = content.encode() if isinstance(content, str) else bytes(content)
            self._write_bytes(sandbox, path, data, prefix=prefix, on_error=SandboxCreateError)

    def _write_bytes(self, sandbox: Any, path: str, data: bytes, *, prefix: str = "", on_error: type = RuntimeError) -> None:
        parent = str(Path(path).parent)
        mk = f"mkdir -p {shlex.quote(parent)} && " if parent and parent not in (".", "", "/") else ""
        r = sandbox.exec(self._exec.shell, "-lc", f"{prefix}{mk}cat > {shlex.quote(path)}", input=data, check=False)
        if r.exit_code != 0:
            raise on_error(f"writing {path!r} failed (exit {r.exit_code}): {r.stderr_text}")

    # -- exec ---------------------------------------------------------------
    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        sandbox = handle.raw
        effective = command
        # Tenki exec has no uid parameter. Best-effort: drop to a named non-root
        # user via `su`. Numeric uids and root run as the sandbox default user.
        if isinstance(user, str) and user and user != "root":
            effective = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(user)}"
        argv = [self._exec.shell, "-lc", effective]
        timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        try:
            result = await asyncio.to_thread(
                lambda: sandbox.exec(
                    *argv,
                    cwd=cwd,
                    env=(dict(env) if env else None),
                    timeout=timeout,
                    check=False,
                )
            )
        except Exception as e:  # noqa: BLE001 - a runtime failure without a process exit code
            name = type(e).__name__
            if "Timeout" in name:
                return SandboxExecResult(stdout=None, stderr=f"{name}: {e}", return_code=124, error_type="timeout")
            return SandboxExecResult(stdout=None, stderr=f"{name}: {e}", return_code=125, error_type="sandbox")
        return SandboxExecResult(
            stdout=result.stdout_text or None,
            stderr=result.stderr_text or None,
            return_code=int(result.exit_code),
        )

    # -- files --------------------------------------------------------------
    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        sandbox = handle.raw
        data = source_path.read_bytes()
        await asyncio.to_thread(self._write_bytes, sandbox, target_path, data)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        sandbox = handle.raw

        def _download() -> None:
            r = sandbox.exec(self._exec.shell, "-lc", f"cat {shlex.quote(source_path)}", check=False)
            if r.exit_code != 0:
                raise FileNotFoundError(f"download of {source_path!r} failed (exit {r.exit_code}): {r.stderr_text}")
            raw = r.stdout if isinstance(r.stdout, (bytes, bytearray)) else str(r.stdout).encode()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(bytes(raw))

        await asyncio.to_thread(_download)

    # -- status / lifecycle -------------------------------------------------
    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        sandbox = handle.raw

        def _status() -> SandboxStatus:
            try:
                sandbox.refresh()
            except Exception:  # noqa: BLE001
                return SandboxStatus.UNKNOWN
            return _to_status(getattr(sandbox, "state", None))

        return await asyncio.to_thread(_status)

    async def close(self, handle: SandboxHandle) -> None:
        sandbox = handle.raw

        def _close() -> None:
            try:
                sandbox.close()  # terminates the microVM
            except Exception:  # noqa: BLE001 - already gone; a failed close leaves the VM until max_duration
                pass

        await asyncio.to_thread(_close)

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    # -- ConnectableProvider (optional capability) --------------------------
    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        # Tenki sandboxes are reachable by id from any process holding the key,
        # so the id alone rebuilds the handle. `scope` is unused (no lease concept).
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        client = await self._client_handle()
        sandbox_id = str(descriptor["sandbox_id"])

        def _connect() -> Any:
            sandbox = client.get(sandbox_id)
            try:
                sandbox.refresh()
                state = str(getattr(sandbox, "state", "")).lower()
                if state in {"paused", "pausing"}:
                    sandbox.resume()
                if str(getattr(sandbox, "state", "")).lower() != "running":
                    sandbox.wait_ready(self._create.ready_timeout_s)
            except Exception:  # noqa: BLE001 - surface the handle; callers re-check status
                pass
            return sandbox

        sandbox = await asyncio.to_thread(_connect)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox)
