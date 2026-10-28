# confusius-load-data

A portable [Agent Skill](https://agentskills.io/specification) for importing custom
arrays into ConfUSIus. The canonical instructions are in [SKILL.md](SKILL.md);
[references/geometry.md](references/geometry.md) contains the detailed coordinate
rules and [scripts/examples.py](scripts/examples.py) contains runnable examples.
Copy the **whole directory**, not just `SKILL.md`, to retain these relative links.
No host-specific tool names, commands, or frontmatter extensions are required.

## Install for an assistant

With ConfUSIus installed, choose a scope explicitly. By default, installation uses
the shared `.agents/skills/` location supported by Pi, Codex, Cursor, and other
compatible assistants. Use `--claude` for Claude Code's separate discovery directory:

```sh
confusius skills list
confusius skills install --scope global
confusius skills install --scope project
confusius skills install --claude --scope global
confusius skills install --claude --scope project
```

`list` prints each skill's name and description. `install` defaults to all bundled
skills; pass `confusius-load-data` before the options to install only this skill.
All destinations are checked for conflicts before any skill is copied.

Project scope installs relative to the current working directory; global scope uses
the user's home. Existing skills are refused unless `--force` is supplied; forced
replacement removes the old skill directory, including local changes and stale files.
No assistant configuration is modified and no network access is needed to install.

For a manual installation, copy this entire directory to one of these locations:

| Assistant | Project directory | User directory |
|---|---|---|
| Pi | `.agents/skills/` | `~/.agents/skills/` |
| Codex | `.agents/skills/` | `~/.agents/skills/` |
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| Cursor | `.agents/skills/` | `~/.agents/skills/` |

Restart/reload your assistant's skill discovery if necessary, then ask it to
"use confusius-load-data to import my lab's MAT recording." The canonical source is
`src/confusius/_skills/`, which is bundled in the wheel, not automatically a host
discovery location. Repository maintenance workflows such as the release skill are
not distributed.
Shared-directory discovery depends on the assistant, not the skill format alone.
Avoid installing duplicate copies in multiple directories scanned by one assistant.
The same files work in all four hosts; interactive host discovery is not part of
the Python example checks.

Documented locations:
[Pi](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/skills.md),
[Codex](https://developers.openai.com/codex/skills),
[Claude Code](https://code.claude.com/docs/en/skills),
[Cursor](https://cursor.com/docs/context/skills).

## Run the examples

In this repository's development environment, from the repository root:

```sh
uv run python src/confusius/_skills/confusius-load-data/scripts/examples.py
```

Or use `python /path/to/confusius-load-data/scripts/examples.py` from any working
directory with ConfUSIus, NumPy, SciPy, h5py, Dask, and Zarr installed in that Python
environment. The script creates only temporary synthetic data, needs no network,
and uses assertions (do not run Python with `-O`). It checks exact sample mappings,
physical coordinates, jittered timestamps, IQ values, laziness, multi-pose timing,
and geometry-preserving serialization. All examples use public ConfUSIus APIs.
These checks prove the illustrated conventions, not a real scanner's calibration.

Optionally validate the skill format with the Agent Skills reference validator:

```sh
uvx --from skills-ref agentskills validate src/confusius/_skills/confusius-load-data
```

ConfUSIus is evolving; verify the installed `create_voxeldata` signature before
adapting examples to older releases. This skill targets the native `k/j/i` plus
`VoxelToWorldIndex` data model, not legacy plain `z/y/x` dimension APIs.

## Repository sources used

These paths identify the implementation and regression tests consulted when writing
the skill; they are not runtime dependencies of the copied skill directory:

- `src/confusius/xarray/create.py`: constructor, padding, ordering, and metadata.
- `src/confusius/_utils/geometry.py`: voxel-label transforms and per-pose geometry.
- `src/confusius/_utils/coordinates.py`: default probe-surface origin.
- `src/confusius/validation/voxeldata.py`: universal and optional validation rules.
- `tests/unit/test_xarray/test_create.py`: constructor examples and failure cases.
- `tests/unit/test_validation/test_voxeldata.py`: model validation regressions.
- `docs/user-guide/voxeldata.md` and `docs/user-guide/io.md`: data model and I/O.
- `docs/examples/01_io/02_create_voxeldata_from_mat.py`: real v7.3 MAT import.
