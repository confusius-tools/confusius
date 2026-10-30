---
icon: lucide/bot
---

# Agents

ConfUSIus bundles [Agent Skills](https://agentskills.io/): instructions and supporting
references that coding assistants load when relevant to your task. They work with
Pi, Codex, Claude Code, Cursor, and other compatible assistants.

## Available Skills

Run this command to see the bundled skills and their descriptions:

```sh
confusius skills list
```

The first skill, **`confusius-load-data`**, helps assistants choose the right
ConfUSIus loader or wrap custom arrays as VoxelData. It guides them through finding
axis order, spatial calibration, acquisition timing, and affine conventions.
It includes geometry references and runnable, offline examples.

## Install Skills

Install all bundled skills into the current project:

```sh
confusius skills install
```

The default destination is `.agents/skills/` in the **current working directory**,
not an automatically detected repository root. Pi, Codex, Cursor, and other
assistants support this shared location; discovery depends on your assistant.

Use `--global` to install in your home directory. Use `--target` to choose another
skills directory, such as Claude Code's `.claude/skills/`:

| Command | Destination |
|---|---|
| `confusius skills install` | `.agents/skills/` |
| `confusius skills install --global` | `~/.agents/skills/` |
| `confusius skills install --target .claude/skills` | `.claude/skills/` |
| `confusius skills install --global --target .claude/skills` | `~/.claude/skills/` |

Relative targets resolve against the current directory, or your home with
`--global`. Absolute targets are used directly; a leading `~` is expanded.

To install only one named skill:

```sh
confusius skills install confusius-load-data
```

Existing destinations are refused unless you pass `--force`. Successful replacement
discards local changes and stale files. The installer stages all resources and keeps
originals as backups until every replacement succeeds. Filesystem errors trigger
rollback; if restoration also fails, backups are retained and their location is
reported for manual recovery.

Installation includes all references and scripts and needs no separate download.
Repository maintenance skills, such as the release workflow, are not distributed.

## Use a Skill

Reload your assistant's skill discovery if necessary, then ask it to use the skill,
for example: “Use confusius-load-data to help me load my MATLAB recording.” Avoid
installing duplicate copies in locations scanned by the same assistant.

The installed `confusius-load-data/scripts/examples.py` runs with the Python
environment containing ConfUSIus. It checks synthetic data and geometry conventions,
not your scanner's physical calibration. For the underlying APIs and data model,
see [Input/Output](io.md) and [The VoxelData Model](voxeldata.md).

## Documentation for Assistants

You can give an assistant [llms.txt](https://confusius.tools/latest/llms.txt), an
index linking to Markdown versions of the documentation. For the combined reference,
use [llms-full.txt](https://confusius.tools/latest/llms-full.txt), which is much larger.
These are documentation resources, not installable skills, and are useful even when
your assistant does not support skills. The links above target the latest docs;
choose the documentation version matching your installed ConfUSIus release.
