---
name: release
description: Create a new ConfUSIus release (version bump, commit, tag, push, release notes)
argument-hint: <new-version>
disable-model-invocation: true
---

Perform a full ConfUSIus release. The new version string is: `$ARGUMENTS`

Follow these steps in order. **Do not push or take irreversible action until explicitly
confirmed by the user.**

---

## Step 1 — Validate input

If no version was provided, ask the user for the version string (e.g. `0.0.1-a16`).

Read the current version from `pyproject.toml`. Find the most recent tag:

```bash
git describe --tags --abbrev=0
```

Store both for use in later steps.

---

## Step 2 — Update version references

Edit the following files. Use the current year (from today's date) for any year fields.

### `pyproject.toml`
Replace `version = "OLD"` with `version = "NEW"`.

### `CITATION.cff`
- Replace `version: OLD` with `version: NEW`.
- Replace `date-released: 'OLD_DATE'` with `date-released: 'TODAY'` (ISO format: YYYY-MM-DD).

### `README.md`
In the citation section only (do **not** touch badge URLs or Zenodo DOI links):
- Replace the prose citation version: `ConfUSIus (vOLD)` → `ConfUSIus (vNEW)`.
- Replace the BibTeX version field: `version   = {vOLD}` → `version   = {vNEW}`.
- Replace the BibTeX year field if the current year differs: `year      = {OLD_YEAR}` → `year      = {CURRENT_YEAR}`.

### `docs/citing.md`
Same replacements as `README.md`.

---

## Step 3 — Sync lock file

```bash
uv sync
```

---

## Step 4 — Run pre-commit checks

```bash
just pre-commit
```

Fix any failures before continuing.

---

## Step 5 — Create version bump commit

Stage only these files: `pyproject.toml`, `CITATION.cff`, `README.md`,
`docs/citing.md`, `uv.lock`.

Commit message:

```
chore: bump version to vNEW
```

---

## Step 6 — Create annotated tag

Collect the commit list since the previous tag (excluding the version bump commit itself):

```bash
git log vPREV..HEAD~1 --oneline
```

Group by prefix into sections (omit empty sections):

| Commit prefix        | Section heading     |
|----------------------|---------------------|
| `feat`               | **New features**    |
| `fix`                | **Bug fixes**       |
| `docs`               | **Documentation**   |
| `refactor`, `perf`   | **Improvements**    |
| `test`, `chore`, `style` | **Other**       |

Use the grouped list as SUMMARY in the tag message below:

```
ConfUSIus vNEW

SUMMARY (bullet list, one line per commit, strip the conventional commit prefix)
```

Create the tag:

```bash
git tag -a vNEW -m "$(cat <<'EOF'
<the message above>
EOF
)"
```

---

## Step 7 — Review and confirm

Show the user:

1. The full commit diff: `git show HEAD`
2. The tag message: `git tag -n99 vNEW`

Then ask: **"Ready to push commit and tag to origin? (yes / no)"**

Do **not** push until the user explicitly says yes.

---

## Step 8 — Push

```bash
git push origin main
git push origin vNEW
```

---

## Step 9 — GitHub release message

Generate and display the following for the user to paste into the GitHub release UI.
Use the same grouped commit list from Step 7 (omit **Other** unless notable).

```markdown
## What's new in vNEW

### New features
- ...

### Bug fixes
- ...

### Documentation
- ...

### Improvements
- ...

```

---

## Step 10 — Discord announcement

Generate and display the following for the user to post in Discord.
Write ONE_OR_TWO_SENTENCE_HIGHLIGHT as a plain-English summary of the most
notable changes (no jargon, no commit hashes).

```
🎉 **ConfUSIus vNEW** is out!

ONE_OR_TWO_SENTENCE_HIGHLIGHT

📋 **Changelog:** https://github.com/sdiebolt/confusius/releases/tag/vNEW
```
