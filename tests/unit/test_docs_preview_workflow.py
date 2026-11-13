"""Exercise the request gate without contacting GitHub or deploying a site."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/docs-preview.yml"
SHA = "a" * 40

pytestmark = pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash") or not shutil.which("jq"),
    reason="The deployment workflow runs in Bash with jq on Ubuntu.",
)


@pytest.fixture
def request_environment(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "GH_REPO": "confusius-tools/confusius",
        "PR_NUMBER": "581",
        "EVENT_NAME": "pull_request_target",
        "BUILD_SHA": SHA,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "COMMENT_LOG": str(tmp_path / "comments"),
        "MOCK_PR": json.dumps(
            {
                "state": "open",
                "head": {"sha": SHA, "ref": "feature", "repo": {"id": 123}},
                "labels": [{"name": "docs-preview"}],
            }
        ),
        "MOCK_RUNS": json.dumps(
            {
                "workflow_runs": [
                    {
                        "id": 101,
                        "head_sha": SHA,
                        "head_branch": "feature",
                        "head_repository": {"id": 123},
                        "status": "completed",
                        "conclusion": "success",
                        "pull_requests": [{"number": 581}],
                    }
                ]
            }
        ),
        "MOCK_ARTIFACTS": json.dumps(
            {"artifacts": [{"name": f"site-{SHA}", "expired": False}]}
        ),
    }


def run_request(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the actual workflow selection step with a fake GitHub CLI."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = workflow["jobs"]["request-preview"]["steps"][0]["run"]
    fake_cli = r"""
    gh() {
      case "$*" in
        *pulls/*) printf '%s' "$MOCK_PR" ;;
        *commits/*) printf '%s' "$MOCK_PRS" ;;
        *workflows/docs.yml/runs*) printf '%s' "$MOCK_RUNS" ;;
        *artifacts*) printf '%s' "$MOCK_ARTIFACTS" ;;
        'pr comment '*) printf '%s\n' "$*" >> "$COMMENT_LOG" ;;
        *) echo "Unexpected gh call: $*" >&2; return 99 ;;
      esac
    }
    """
    return subprocess.run(
        ["bash", "-c", fake_cli + script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("event", ["pull_request_target", "workflow_run"])
def test_preview_selects_labeled_pr_build(
    request_environment: dict[str, str], event: str
) -> None:
    request_environment["EVENT_NAME"] = event

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert Path(request_environment["GITHUB_OUTPUT"]).read_text() == (
        f"ready=true\npr_number=581\nhead_sha={SHA}\nrun_id=101\nartifact=site-{SHA}\n"
    )


@pytest.mark.parametrize("event", ["pull_request_target", "workflow_run"])
def test_preview_requires_label(
    request_environment: dict[str, str], event: str
) -> None:
    request_environment["EVENT_NAME"] = event
    pr = json.loads(request_environment["MOCK_PR"])
    pr["labels"] = []
    request_environment["MOCK_PR"] = json.dumps(pr)

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert not Path(request_environment["GITHUB_OUTPUT"]).exists()
    assert not Path(request_environment["COMMENT_LOG"]).exists()


@pytest.mark.parametrize(
    "change",
    ["wrong_sha", "wrong_pr", "wrong_repo", "wrong_branch", "failed", "running"],
)
def test_preview_rejects_wrong_or_unfinished_builds(
    request_environment: dict[str, str], change: str
) -> None:
    runs = json.loads(request_environment["MOCK_RUNS"])
    run = runs["workflow_runs"][0]
    if change == "wrong_sha":
        run["head_sha"] = "b" * 40
    elif change == "wrong_pr":
        run["pull_requests"] = [{"number": 999}]
    elif change == "wrong_repo":
        run["head_repository"] = {"id": 999}
    elif change == "wrong_branch":
        run["head_branch"] = "unrelated"
    elif change == "failed":
        run["conclusion"] = "failure"
    else:
        run["status"] = "in_progress"
    request_environment["MOCK_RUNS"] = json.dumps(runs)

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert not Path(request_environment["GITHUB_OUTPUT"]).exists()
    assert not Path(request_environment["COMMENT_LOG"]).exists()


@pytest.mark.parametrize("artifacts", [[], [{"name": f"site-{SHA}", "expired": True}]])
def test_preview_silently_waits_for_missing_or_expired_artifacts(
    request_environment: dict[str, str], artifacts: list[dict[str, object]]
) -> None:
    request_environment["MOCK_ARTIFACTS"] = json.dumps({"artifacts": artifacts})

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert not Path(request_environment["GITHUB_OUTPUT"]).exists()
    assert not Path(request_environment["COMMENT_LOG"]).exists()


def test_preview_finds_fork_pr_without_run_metadata(
    request_environment: dict[str, str],
) -> None:
    request_environment["EVENT_NAME"] = "workflow_run"
    request_environment["PR_NUMBER"] = ""
    pr = json.loads(request_environment["MOCK_PR"])
    pr["number"] = 581
    request_environment["MOCK_PRS"] = json.dumps([pr])
    runs = json.loads(request_environment["MOCK_RUNS"])
    runs["workflow_runs"][0]["pull_requests"] = []
    request_environment["MOCK_RUNS"] = json.dumps(runs)

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert Path(request_environment["GITHUB_OUTPUT"]).read_text() == (
        f"ready=true\npr_number=581\nhead_sha={SHA}\nrun_id=101\nartifact=site-{SHA}\n"
    )


def test_preview_does_not_spam_comments_on_build_completion(
    request_environment: dict[str, str],
) -> None:
    request_environment["EVENT_NAME"] = "workflow_run"
    request_environment["MOCK_ARTIFACTS"] = json.dumps({"artifacts": []})

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert not Path(request_environment["GITHUB_OUTPUT"]).exists()
    assert not Path(request_environment["COMMENT_LOG"]).exists()


@pytest.mark.parametrize("change", ["new_commit", "label_removed", "closed"])
def test_deployment_rechecks_pr_before_using_deploy_credentials(
    request_environment: dict[str, str], change: str
) -> None:
    pr = json.loads(request_environment["MOCK_PR"])
    if change == "new_commit":
        pr["head"]["sha"] = "b" * 40
    elif change == "label_removed":
        pr["labels"] = []
    else:
        pr["state"] = "closed"
    request_environment.update(
        MOCK_PR=json.dumps(pr), EXPECTED_SHA=SHA, DOCS_DEPLOY_TOKEN="test-token"
    )
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    script = workflow["jobs"]["deploy-preview"]["steps"][2]["run"]
    fake_commands = r"""
    gh() { printf '%s' "$MOCK_PR"; }
    git() { echo 'Must not reach git' >> "$COMMENT_LOG"; return 99; }
    """

    result = subprocess.run(
        ["bash", "-c", fake_commands + script],
        env=request_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not Path(request_environment["COMMENT_LOG"]).exists()


def test_cleanup_uses_trusted_code_for_all_closed_prs() -> None:
    path = WORKFLOW.with_name("docs-cleanup-preview.yml")
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert workflow["on"] == {"pull_request_target": {"types": ["closed", "unlabeled"]}}
    cleanup = workflow["jobs"]["cleanup"]
    assert cleanup["if"] == (
        "github.event.action == 'closed' || github.event.label.name == 'docs-preview'"
    )
    checkout = cleanup["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"] == {
        "ref": "${{ github.event.repository.default_branch }}",
        "persist-credentials": "false",
    }
    assert "python ../tools/deduplicate_docs_assets.py ." in cleanup["steps"][1]["run"]


@pytest.mark.parametrize(
    "filename, job",
    [
        ("docs.yml", "deploy-docs"),
        ("docs-preview.yml", "deploy-preview"),
        ("docs-cleanup-preview.yml", "cleanup"),
    ],
)
def test_all_docs_publishers_queue_without_replacing_pending_jobs(
    filename: str, job: str
) -> None:
    workflow = yaml.safe_load(WORKFLOW.with_name(filename).read_text(encoding="utf-8"))
    assert workflow["jobs"][job]["concurrency"] == {
        "group": "docs-publish",
        "queue": "max",
        "cancel-in-progress": False,
    }


@pytest.mark.parametrize(
    "state, labels, remove",
    [
        ("closed", ["docs-preview"], True),
        ("open", [], True),
        ("open", ["docs-preview"], False),
    ],
)
def test_cleanup_removes_only_unwanted_previews_and_unused_assets(
    tmp_path: Path, state: str, labels: list[str], remove: bool
) -> None:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    preview = seed / "pr-preview" / "pr-581"
    preview.mkdir(parents=True)
    (preview / "index.html").write_text('<img src="../../shared-assets/unused.png">')
    versions = [{"version": "pr-581", "title": "PR #581 Preview", "aliases": []}]
    (seed / "pr-preview" / "versions.json").write_text(json.dumps(versions))
    shared = seed / "shared-assets"
    shared.mkdir()
    (shared / "unused.png").write_bytes(b"preview image")
    (shared / "live.png").write_bytes(b"released image")
    (seed / "index.html").write_text('<img src="shared-assets/live.png">')
    for arguments in [
        ["init", "--bare", str(remote)],
        ["init", "-b", "main", str(seed)],
        ["-C", str(seed), "config", "user.name", "Test"],
        ["-C", str(seed), "config", "user.email", "test@example.com"],
        ["-C", str(seed), "add", "."],
        ["-C", str(seed), "commit", "-m", "Initial site"],
        ["-C", str(seed), "push", str(remote), "main"],
    ]:
        subprocess.run(["git", *arguments], check=True, capture_output=True)
    sandbox = tmp_path / "sandbox"
    (sandbox / "tools").mkdir(parents=True)
    shutil.copyfile(
        WORKFLOW.parents[2] / "tools" / "deduplicate_docs_assets.py",
        sandbox / "tools" / "deduplicate_docs_assets.py",
    )
    workflow = yaml.safe_load(
        WORKFLOW.with_name("docs-cleanup-preview.yml").read_text(encoding="utf-8")
    )
    script = workflow["jobs"]["cleanup"]["steps"][1]["run"]
    script = script.replace("${{ secrets.DOCS_DEPLOY_TOKEN }}", "test-token")
    script = script.replace("${{ github.event.number }}", "581")
    fake_commands = r"""
    gh() { printf '%s' "$MOCK_PR"; }
    git() {
      if [ "$1" = clone ]; then
        command git clone "$MOCK_REMOTE" docs-repo
      else
        command git "$@"
      fi
    }
    """
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-c", fake_commands + script],
        cwd=sandbox,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "GH_REPO": "confusius-tools/confusius",
            "PR_NUMBER": "581",
            "GITHUB_OUTPUT": str(output),
            "MOCK_REMOTE": str(remote),
            "MOCK_PR": json.dumps(
                {"state": state, "labels": [{"name": label} for label in labels]}
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"removed={str(remove).lower()}\n"
    paths = subprocess.run(
        ["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert ("pr-preview/pr-581/index.html" not in paths) == remove
    assert ("shared-assets/unused.png" not in paths) == remove
    assert "shared-assets/live.png" in paths
    published_versions = subprocess.run(
        ["git", "--git-dir", str(remote), "show", "main:pr-preview/versions.json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert json.loads(published_versions) == ([] if remove else versions)


def test_preview_rejects_closed_pr(request_environment: dict[str, str]) -> None:
    request_environment["MOCK_PR"] = json.dumps(
        {"state": "closed", "head": {"sha": SHA}}
    )

    result = run_request(request_environment)

    assert result.returncode == 0, result.stderr
    assert not Path(request_environment["GITHUB_OUTPUT"]).exists()
    assert not Path(request_environment["COMMENT_LOG"]).exists()
