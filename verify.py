"""Live smoke test for the Tenki NeMo Gym provider. No mocks. Exits non-zero on failure.

Proves the provider against a real Tenki account through NeMo Gym's own protocol:
create -> exec (assert 42) -> status -> upload/download roundtrip -> the
ConnectableProvider serialize/connect path -> close. Every sandbox is terminated.

Token/project: env (CI) or ~/.config/tenki/config.yaml (local `tenki login`).
Requires NeMo Gym (`nemo_gym`) installed in the environment.

    python verify.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from nemo_gym.sandbox.providers.base import SandboxSpec

# Prefer going through the registry (proves the entry point resolves); fall back
# to importing the class directly if the package isn't pip-installed yet.
try:
    from nemo_gym.sandbox.providers.registry import create_provider

    def make_provider(cfg):
        return create_provider({"tenki": cfg})

    ENTRYPOINT = True
except Exception:  # pragma: no cover
    from nemo_gym_tenki import TenkiProvider

    def make_provider(cfg):
        return TenkiProvider(**cfg)

    ENTRYPOINT = False


def cfg(key: str) -> str:
    try:
        with open(os.path.expanduser("~/.config/tenki/config.yaml")) as f:
            for line in f:
                if line.startswith(key + ":"):
                    return line.split(":", 1)[1].strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def connection_config() -> dict:
    token = os.environ.get("TENKI_API_KEY") or os.environ.get("TENKI_AUTH_TOKEN") or cfg("auth_token")
    if not token:
        print("FAIL: no token. Set TENKI_API_KEY (tk_...), or run `tenki login`.")
        sys.exit(1)
    conn = {"auth_token": token}
    # Optional: only pass a project if one is provided; otherwise the SDK resolves
    # the single project on the key (the CI path).
    project = os.environ.get("TENKI_PROJECT_ID") or cfg("current_project_id")
    workspace = os.environ.get("TENKI_WORKSPACE_ID") or cfg("current_workspace_id")
    if project:
        conn["project_id"] = project
    if workspace:
        conn["workspace_id"] = workspace
    return conn


async def main() -> None:
    provider = make_provider({"connection": connection_config(), "create": {"cpu_cores": 1, "memory_mb": 1024}})
    print(f"provider: {provider.name}  (via {'entry point' if ENTRYPOINT else 'direct import'})")

    handle = await provider.create(SandboxSpec(env={"NEMO_GYM_CHECK": "wired"}))
    try:
        print(f"created sandbox: {handle.sandbox_id}")

        r = await provider.exec(handle, "python3 -c 'print(6 * 7)'")
        assert r.return_code == 0, f"exit {r.return_code}, stderr={r.stderr!r}"
        assert (r.stdout or "").strip() == "42", f"stdout={r.stdout!r}"
        print("ok  exec -> 42")

        r = await provider.exec(handle, 'printf "%s" "$NEMO_GYM_CHECK"')
        assert (r.stdout or "").strip() == "wired", f"env not wired: {r.stdout!r}"
        print("ok  spec.env reached the guest")

        r = await provider.exec(handle, "exit 7")
        assert r.return_code == 7, f"expected 7, got {r.return_code}"
        print("ok  non-zero exit preserved")

        st = await provider.status(handle)
        assert str(st.value) == "running", f"status={st}"
        print(f"ok  status -> {st.value}")

        # file upload/download roundtrip
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "up.txt"
            src.write_text("tenki-nemo-gym\n")
            await provider.upload_file(handle, src, "/home/tenki/up.txt")
            back = await provider.exec(handle, "cat /home/tenki/up.txt")
            assert (back.stdout or "").strip() == "tenki-nemo-gym", f"upload roundtrip: {back.stdout!r}"
            dst = Path(d) / "down.txt"
            await provider.download_file(handle, "/home/tenki/up.txt", dst)
            assert dst.read_text().strip() == "tenki-nemo-gym", "download roundtrip"
        print("ok  upload/download roundtrip")

        # ConnectableProvider: serialize -> rebuild a handle -> still usable
        descriptor = await provider.serialize_handle(handle)
        assert descriptor.get("sandbox_id") == handle.sandbox_id
        reconnected = await provider.connect(descriptor)
        r = await provider.exec(reconnected, "cat /home/tenki/up.txt")
        assert (r.stdout or "").strip() == "tenki-nemo-gym", "reconnected handle lost the fs"
        print(f"ok  ConnectableProvider serialize/connect -> same sandbox {reconnected.sandbox_id}")

        print("\nPASS: nemo-gym-tenki provider verified against live Tenki.")
    finally:
        await provider.close(handle)
        await provider.aclose()
        print("cleanup: sandbox terminated.")


if __name__ == "__main__":
    asyncio.run(main())
