---
icon: lucide/bot
---

# Agents

ConfUSIus includes [Agent Skills](https://agentskills.io/): task-specific instructions
that help coding agents use ConfUSIus. Skills work with Pi, Codex, Claude Code, Cursor,
and other compatible agents.

## Available Skills

List the bundled skills:

```sh
confusius skills list
```

| Skill | Description |
|---|---|
| `confusius-load-data` | Load custom or supported ultrasound data into ConfUSIus. Use for file imports, VoxelData dimensions, acquisition timing, and voxel-to-world geometry. |

## Install Skills

Install all bundled skills in your project:

```sh
confusius skills install
```

By default, skills are installed in `.agents/skills/` relative to your current working
directory (not necessarily the repository root). This shared location is supported by
Pi, Codex, Cursor, and other agents.

You can also install skills globally or choose a different destination:

| Command | Destination |
|---|---|
| `confusius skills install` | `.agents/skills/` |
| `confusius skills install --global` | `~/.agents/skills/` |
| `confusius skills install --target .claude/skills` | `.claude/skills/` |
| `confusius skills install --global --target .claude/skills` | `~/.claude/skills/` |

Relative paths resolve from the current directory, or your home directory with
`--global`. Absolute paths and `~` are also supported. Existing skills are preserved
unless `--force` is specified, which replaces them and discards local changes.

To install a specific skill:

```sh
confusius skills install confusius-load-data
```

## Use a Skill

After installation, reload your agent's skill discovery if needed. You can then ask it
to use a skill directly, for example:

> Use confusius-load-data to help me load my MATLAB recording.

Skill discovery varies between agents. Avoid installing duplicate copies in directories
scanned by the same agent.

## Documentation for Assistants

ConfUSIus also provides agent-friendly documentation:

- [llms.txt](https://confusius.tools/latest/llms.txt): an index linking to Markdown
  versions of the documentation.
- [llms-full.txt](https://confusius.tools/latest/llms-full.txt): the complete
  documentation in a single, much larger file.

These resources can be used with any agent, even without skills support. The links above
point to the latest documentation; use the version matching your installed ConfUSIus
release.
