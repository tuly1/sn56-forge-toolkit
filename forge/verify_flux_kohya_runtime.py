"""Build-time interface proof for the pinned Kohya FLUX runtime."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

from forge import flux_kohya_config


_PINNED_SOURCE_SHA256 = {
    "flux_train_network.py": "a7614dec11ad967200fdc56005140cc2cf1892b666bcc8da206b7cefaa8d5d6a",
    "train_network.py": "d326c29596c2b39b2dfe6ca17af540b1a649aa25261725b6c9658ea1c6cfe44f",
    "networks/lora_flux.py": "671736e1e4bd4d6e0c925b5e26be20c48dfd63ba9042e118fbe63b7db223076b",
    "library/train_util.py": "2c424c9f0df9b75258926d7c3693b6efd5f9dcb3629f374fd666c547b85b0543",
}
# The pinned revision consumes this directly from the TOML-populated Namespace;
# it is intentionally not an argparse action.
_CONFIG_ONLY_KEYS = frozenset({"mem_eff_save"})


def verify(sd_scripts_dir: str = "/app/sd-scripts") -> dict[str, object]:
    script = os.path.join(sd_scripts_dir, "flux_train_network.py")
    if not os.path.isfile(script):
        raise RuntimeError(f"missing pinned FLUX trainer: {script}")
    for relative, expected in _PINNED_SOURCE_SHA256.items():
        path = os.path.join(sd_scripts_dir, relative)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"pinned Kohya source mismatch for {relative}: {actual}"
            )
    sys.path.insert(0, sd_scripts_dir)
    try:
        import flux_train_network
        import lion_pytorch  # noqa: F401
        import networks.lora_flux  # noqa: F401
        from library import train_util
    finally:
        try:
            sys.path.remove(sd_scripts_dir)
        except ValueError:
            pass

    parser = flux_train_network.setup_parser()
    recognized = {action.dest for action in parser._actions}
    with tempfile.TemporaryDirectory(prefix="forge-kohya-verify-") as temp:
        path = os.path.join(temp, "config.toml")
        config = flux_kohya_config.build_config(
            base_model="/cache/models/example/base.safetensors",
            train_data_dir="/dataset/images",
            output_dir="/app/checkpoints/task/repo",
            output_name="repo",
            config_file=path,
        )
        unknown = sorted(set(config) - recognized - _CONFIG_ONLY_KEYS)
        if unknown:
            raise RuntimeError(
                f"Kohya parser does not recognize config keys: {unknown}"
            )
        flux_kohya_config.write_config(config, path)
        initial = parser.parse_args(["--config_file", path])
        parsed = train_util.read_config_from_file(initial, parser)
        for key, expected in config.items():
            actual = getattr(parsed, key, None)
            if key == "config_file":
                expected = os.path.splitext(expected)[0]
            if actual != expected:
                raise RuntimeError(
                    f"Kohya config round-trip mismatch for {key}: "
                    f"expected {expected!r}, got {actual!r}"
                )

    step_name = train_util.get_step_ckpt_name(parsed, ".safetensors", 25)
    final_name = train_util.get_last_ckpt_name(parsed, ".safetensors")
    if step_name != "repo-step00000025.safetensors":
        raise RuntimeError(f"unexpected Kohya step checkpoint name: {step_name}")
    if final_name != "repo.safetensors":
        raise RuntimeError(f"unexpected Kohya final checkpoint name: {final_name}")

    return {
        "result": "PASS",
        "recognized_config_keys": len(config),
        "step_checkpoint": step_name,
        "final_checkpoint": final_name,
        "source_files_verified": len(_PINNED_SOURCE_SHA256),
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    print(verify())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
