# ConfUSIus Agent Skills

Distributable skills have a single canonical source in
[`src/confusius/_skills/`](../src/confusius/_skills/). The complete directories ship
in the Python wheel, including references and scripts.

```sh
confusius skills list
confusius skills install --agent pi --scope global
confusius skills install --agent codex --scope project
confusius skills install --agent claude-code --scope project
confusius skills install --agent cursor --scope global

# Install just one skill instead.
confusius skills install confusius-load-data --agent pi --scope global
```

`list` displays names and descriptions from each skill's YAML frontmatter.
`install` defaults to all bundled skills, checking all destinations for conflicts
before copying anything. `--scope project` uses the current working directory; `--scope global` uses the
user's home. Existing skills require `--force` to replace. See the
[load-data skill README](../src/confusius/_skills/confusius-load-data/README.md)
for directory locations and executable examples.

To add a distributable skill, create a directory named for the skill under
`src/confusius/_skills/` with an Agent Skills-compliant `SKILL.md`. Keep supporting
files inside that directory and reference them with relative links. No explicit
packaging manifest or CLI registry change is needed. Run the packaging tests when
adding resources.

Repository-only workflows, including `.claude/skills/release/`, stay in place and
are not listed, installed, or packaged by this feature.
