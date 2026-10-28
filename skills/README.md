# ConfUSIus Agent Skills

Distributable skills have a single canonical source in
[`src/confusius/_skills/`](../src/confusius/_skills/). The complete directories ship
in the Python wheel, including references and scripts.

```sh
confusius skills list
# Shared installation for Pi, Codex, Cursor, and other compatible agents.
confusius skills install
confusius skills install --global

# Claude Code uses a separate discovery directory.
confusius skills install --claude --global

# Install just one skill instead.
confusius skills install confusius-load-data
```

`list` displays names and descriptions from each skill's YAML frontmatter.
`install` defaults to all bundled skills, checking all destinations for conflicts
before copying anything. Installation defaults to the current working directory;
`--global` uses the user's home instead. Installation uses `.agents/skills/` by
default; `--claude` selects `.claude/skills/` instead. Shared-directory discovery
depends on the assistant; it does not guarantee support by every agent.
Existing skills require `--force` to replace. See the
[load-data skill README](../src/confusius/_skills/confusius-load-data/README.md)
for directory locations and executable examples.

To add a distributable skill, create a directory named for the skill under
`src/confusius/_skills/` with an Agent Skills-compliant `SKILL.md`. Keep supporting
files inside that directory and reference them with relative links. No explicit
packaging manifest or CLI registry change is needed. Run the packaging tests when
adding resources.

Repository-only workflows, including `.claude/skills/release/`, stay in place and
are not listed, installed, or packaged by this feature.
