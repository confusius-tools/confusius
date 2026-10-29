# confusius-load-data

A portable [Agent Skill](https://agentskills.io/specification) for loading supported
files or wrapping custom arrays as ConfUSIus VoxelData.

- [SKILL.md](SKILL.md): loader selection, metadata questions, and a minimal example.
- [references/geometry.md](references/geometry.md): affine conventions and edge cases.
- [scripts/examples.py](scripts/examples.py): offline, assert-based checks.

## Install

With ConfUSIus installed:

```sh
confusius skills list
confusius skills install
confusius skills install --global
confusius skills install --claude
confusius skills install --claude --global
```

Installation defaults to `.agents/skills/` in the current working directory, shared
by Pi, Codex, Cursor, and other compatible assistants. `--global` uses the home
directory; `--claude` selects `.claude/skills/` instead. Pass `confusius-load-data`
after `install` to select only this skill. Existing destinations require `--force`,
which replaces the whole installed skill, including local changes.

For manual installation, copy this **whole directory** to the appropriate skills
location. Reload your assistant's skill discovery if needed, then ask it to use
`confusius-load-data`. Shared-directory discovery depends on the assistant; avoid
duplicate installations in directories scanned by the same assistant.

The canonical source is `src/confusius/_skills/confusius-load-data/`; all supporting
files ship in the wheel. Repository maintenance skills such as the release workflow
are not distributed. No host-specific tools or frontmatter extensions are needed.

## Verify

From the repository root:

```sh
uv run python src/confusius/_skills/confusius-load-data/scripts/examples.py
uvx --from skills-ref agentskills validate src/confusius/_skills/confusius-load-data
```

The script needs ConfUSIus, NumPy, SciPy, h5py, Dask, and Zarr. It uses temporary
synthetic files, needs no downloads, and checks sample values, world coordinates,
timing, IQ, multiple poses, and serialization. Do not run Python with `-O` because
it disables assertions. These checks do not establish a real scanner's calibration.

This skill targets the native `k/j/i` plus `VoxelToWorldIndex` data model. Consult
the installed `create_voxeldata` signature and the ConfUSIus I/O and VoxelData guides
when adapting it to another version.
