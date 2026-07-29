from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).parents[2]


def _yaml(path: str) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _workflow_on(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML follows YAML 1.1 and may decode the unquoted key `on` as True.
    raw_workflow = cast("dict[Any, Any]", workflow)
    value = raw_workflow.get("on", raw_workflow.get(True))
    assert isinstance(value, dict)
    return value


def test_only_standalone_official_runtime_variants_are_packaged() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert set(project["project"]["optional-dependencies"]) == {
        "official-cpu",
        "official-gpu",
    }
    assert project["tool"]["uv"]["conflicts"] == [
        [{"extra": "official-cpu"}, {"extra": "official-gpu"}]
    ]
    assert "onnxruntime==1.20.1" in project["project"]["optional-dependencies"]["official-cpu"]
    assert "onnxruntime-gpu==1.20.2" in project["project"]["optional-dependencies"]["official-gpu"]


def test_dockerfile_requires_immutable_apt_inputs_and_bounds_shutdown() -> None:
    dockerfile = (ROOT / "deploy/Dockerfile").read_text(encoding="utf-8")
    required_args = {
        "APT_SNAPSHOT_URL",
        "APT_CA_CERTIFICATES_VERSION",
        "APT_LIBGOMP1_VERSION",
        "APT_LIBSNDFILE1_VERSION",
        "APT_TINI_VERSION",
    }
    arg_lines = {
        line.removeprefix("ARG ") for line in dockerfile.splitlines() if line.startswith("ARG APT_")
    }

    assert arg_lines == required_args
    assert r"snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z" in dockerfile
    for package, variable in (
        ("ca-certificates", "APT_CA_CERTIFICATES_VERSION"),
        ("libgomp1", "APT_LIBGOMP1_VERSION"),
        ("libsndfile1", "APT_LIBSNDFILE1_VERSION"),
        ("tini", "APT_TINI_VERSION"),
    ):
        assert f'"{package}=${{{variable}}}"' in dockerfile
    assert "ASR_ENGINE=official" in dockerfile
    assert '"--timeout-graceful-shutdown", "150"' in dockerfile


def test_compose_supports_official_cpu_and_gpu_with_isolated_serving() -> None:
    compose = _yaml("deploy/compose.yaml")
    services = compose["services"]

    assert services["asr-cpu"]["build"]["args"]["UV_EXTRA"] == "official-cpu"
    assert services["asr-cpu"]["environment"]["ASR_ENGINE"] == "official"
    assert services["asr-cpu"]["environment"]["ASR_REQUIRE_CUDA"] == "false"
    assert services["asr-gpu"]["build"]["args"]["UV_EXTRA"] == "official-gpu"
    assert services["asr-gpu"]["environment"]["ASR_ENGINE"] == "official"
    assert services["asr-gpu"]["environment"]["ASR_REQUIRE_CUDA"] == "true"
    assert services["asr-gpu"]["networks"] == ["serving"]
    assert services["asr-cpu"]["networks"] == ["serving"]
    assert services["model-init"]["networks"] == ["provisioning"]
    assert compose["networks"]["serving"]["internal"] is True
    assert services["asr-gpu"]["stop_grace_period"] == "180s"
    assert services["asr-cpu"]["stop_grace_period"] == "180s"


def test_compose_prepares_identity_readable_secrets_without_exposing_both() -> None:
    services = _yaml("deploy/compose.yaml")["services"]
    initializer = services["secret-init"]
    command = "\n".join(initializer["command"])

    assert initializer["network_mode"] == "none"
    assert initializer["user"] == "0:0"
    assert "-o 10001 -g 10001 -m 0400" in command
    assert services["asr-gpu"]["volumes"] == [
        "model-data:/models:ro",
        "api-key-data:/run/secrets/api:ro",
    ]
    assert services["model-init"]["volumes"] == [
        "model-data:/models",
        "hf-token-data:/run/secrets/hf:ro",
    ]


def test_image_workflow_stages_sha_before_gated_semantic_promotion() -> None:
    workflow = _yaml(".github/workflows/image.yml")
    jobs = workflow["jobs"]
    build_steps = jobs["build"]["steps"]
    metadata = next(step for step in build_steps if step.get("id") == "meta")
    staging_tags = metadata["with"]["tags"]

    assert "type=sha,format=long,prefix=sha-" in staging_tags
    assert "type=semver" not in staging_tags
    promotion = jobs["promote-release-tags"]
    assert set(promotion["needs"]) == {"build", "gpu-release-gate"}
    assert "needs.gpu-release-gate.result == 'success'" in promotion["if"]
    promotion_steps = promotion["steps"]
    release_metadata = next(step for step in promotion_steps if step.get("id") == "release-meta")
    assert "type=semver" in release_metadata["with"]["tags"]


def test_image_pull_request_filter_covers_every_docker_copy_input() -> None:
    workflow = _yaml(".github/workflows/image.yml")
    paths = set(_workflow_on(workflow)["pull_request"]["paths"])

    assert {
        "deploy/**",
        "app/**",
        "scripts/**",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".dockerignore",
    } <= paths


def test_gpu_smoke_prefers_reusable_workflow_inputs_over_scheduled_defaults() -> None:
    workflow = _yaml(".github/workflows/gpu-smoke.yml")
    environment = workflow["env"]

    assert environment["SELECTED_IMAGE_DIGEST"] == (
        "${{ inputs.image_digest || vars.GPU_SMOKE_IMAGE_DIGEST }}"
    )
    assert environment["SELECTED_MODEL_REVISION"] == (
        "${{ inputs.model_revision || vars.GPU_SMOKE_MODEL_REVISION }}"
    )


def test_gpu_smoke_uses_runtime_owned_key_and_proves_offline_egress() -> None:
    workflow = _yaml(".github/workflows/gpu-smoke.yml")
    steps = workflow["jobs"]["smoke"]["steps"]
    commands = "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))

    assert "docker network create --internal" in commands
    assert '--network "${ASR_OFFLINE_NETWORK}"' in commands
    assert "ASR_ENGINE=official" in commands
    assert "-o 10001 -g 10001 -m 0400" in commands
    assert '--api-key-file "${ASR_API_KEY_TOKEN_FILE}"' in commands
    assert "socket.create_connection" in commands
