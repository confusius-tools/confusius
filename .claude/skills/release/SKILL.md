---
name: release
description: Prepare, publish, or post-bump a ConfUSIus release
argument-hint: <prepare|publish|post-bump> <version>
disable-model-invocation: true
---

Perform one step of the ConfUSIus release workflow. The command arguments are:
`$ARGUMENTS`

This skill supports three modes:

- `prepare VERSION`: create a release-preparation branch and PR.
- `publish VERSION`: tag the merged release commit on `main` and prepare release notes.
- `post-bump VERSION`: create a post-release development-version branch and PR.

Always ask for explicit confirmation before pushing, creating a PR, tagging, or taking any
other irreversible action.

---

## Shared validation

1. Parse the first argument as `MODE` and the second as `VERSION`.
2. If either is missing, ask the user.
3. Validate that `MODE` is one of `prepare`, `publish`, or `post-bump`.
4. Read the current version from `pyproject.toml`.
5. Find the most recent tag:

```bash
git describe --tags --abbrev=0
```

6. Use today's date in ISO format (`YYYY-MM-DD`) whenever a release date is needed.

Version conventions:

- `prepare` should normally target a final release version such as `0.2.0`.
- `publish` should target the same final release version.
- `post-bump` should normally target a development version such as `0.3.0.dev0`.

---

## Mode: `prepare VERSION`

Use this mode to prepare the release PR. This happens before tagging.

### Step 1 — Create a release branch

Create a branch named:

```text
release/VERSION
```

Do not switch branches or push until the user confirms.

### Step 2 — Update release files

Edit the following files.

### `pyproject.toml`
- Replace `version = "OLD"` with `version = "VERSION"`.

### `docs/changelog.md`
- Replace the development heading for this release with the final release heading.
  Example: `## 0.2.0.dev0` -> `## 0.2.0`.
- Add a release line directly under the version heading:

```md
Released YYYY-MM-DD.
```

- Keep the rest of the entry intact unless the user asked for content edits.

### `CITATION.cff`
- Replace `version: OLD` with `version: VERSION`.
- Replace `date-released: 'OLD_DATE'` with `date-released: 'TODAY'`.

### `README.md`
In the citation section only:

- Replace `ConfUSIus (vOLD)` with `ConfUSIus (vVERSION)`.
- Replace `version   = {vOLD}` with `version   = {vVERSION}`.
- Replace the citation year if needed.

### `docs/citing.md`
Apply the same citation replacements as in `README.md`.

Update any other release/version/date files only if they actually exist and clearly need to
track the release version.

### Step 3 — Sync and verify

Run:

```bash
uv sync
just pre-commit
```

Fix failures before continuing.

### Step 4 — Commit release prep

Stage only the files relevant to the release preparation.

Use commit message:

```text
chore: prepare vVERSION release
```

### Step 5 — Open the PR

Push the branch and create a PR targeting `main`.

Recommended PR title:

```text
chore: prepare vVERSION release
```

Recommended PR body:

```markdown
## Summary
- finalize version metadata for `vVERSION`
- convert the changelog entry from development to released form
- update release citation metadata and docs references
```

When done, return the PR URL and remind the user that tagging must happen only after this
PR is merged.

---

## Mode: `publish VERSION`

Use this mode only after the release PR has been merged.

### Step 1 — Verify merged `main`

1. Check out the latest `main` state locally.
2. Confirm that `pyproject.toml` now reports `VERSION` exactly, not a `.dev0` version.
3. Confirm that `docs/changelog.md` contains `## VERSION` and a release date.

Do not tag a PR branch or any commit that is not the merged `main` release commit.

### Step 2 — Build release summary

Collect commits since the previous tag:

```bash
git log vPREV..HEAD --oneline
```

Group them by conventional commit prefix, omitting empty sections:

| Commit prefix              | Section heading   |
|----------------------------|-------------------|
| `feat`                     | New features      |
| `fix`                      | Bug fixes         |
| `docs`                     | Documentation     |
| `refactor`, `perf`         | Improvements      |
| `test`, `chore`, `style`   | Other             |

Strip the prefix from each bullet in the generated summary.

### Step 3 — Draft annotated tag message

Use this tag message template:

```text
ConfUSIus vVERSION

SUMMARY
```

Where `SUMMARY` is the grouped bullet list from the previous step.

Prepare the annotated tag command using:

```bash
git tag -a vVERSION -m "$(cat <<'EOF'
ConfUSIus vVERSION

SUMMARY
EOF
)"
```

Do not run the command yet.

### Step 4 — Review and confirm

Show the user:

1. `git show HEAD`
2. The tag message draft

Then ask whether to create and push the tag.

### Step 5 — Create and push tag

After the user confirms, create the annotated tag and push only the tag:

```bash
git tag -a vVERSION -m "$(cat <<'EOF'
ConfUSIus vVERSION

SUMMARY
EOF
)"
git push origin vVERSION
```

Do not create a new commit in this mode unless the user explicitly asks for one.

### Step 6 — Release notes and announcement

Generate for the user:

1. A GitHub release body in Markdown.
2. A short Discord announcement.

Use the grouped commit summary. Omit `Other` unless it contains notable user-facing work.

---

## Mode: `post-bump VERSION`

Use this mode immediately after the tagged release to move the repository back to a
development version.

### Step 1 — Create a post-release branch

Create a branch named:

```text
release/VERSION
```

Examples:

- `release/0.3.0.dev0`
- `release/0.2.1.dev0`

### Step 2 — Update development version files

Edit the following files.

### `pyproject.toml`
- Replace `version = "OLD"` with `version = "VERSION"`.

### `docs/changelog.md`
- Add a new top section:

```md
## VERSION

Current development version for the next ConfUSIus release.
```

- Place it above the most recent released version section.

### `CITATION.cff`
- Replace `version: OLD` with `version: VERSION`.
- Do not invent a new release date for a development version.
- If the file already contains a release date for the last release, preserve it unless the
  user asks for a different convention.

### `README.md` and `docs/citing.md`
- Update citation version strings only if the repository convention is to keep them aligned
  with the in-repo development version.
- If updating them would make the citation text misleading for users, stop and ask the
  user before changing those files.

### Step 3 — Sync and verify

Run:

```bash
uv sync
just pre-commit
```

Fix failures before continuing.

### Step 4 — Commit and open PR

Use commit message:

```text
chore: start VERSION development
```

Create a PR targeting `main`.

Recommended PR title:

```text
chore: start VERSION development
```

Recommended PR body:

```markdown
## Summary
- bump the repository version to `VERSION`
- start the next changelog section
- restore development-version metadata after the release
```

Return the PR URL when done.

---

## Notes

- Prefer `0.2.0.dev0` style versions over non-standard forms such as `0.2.0.dev`.
- The release tag must always point to the merged `main` commit that represents the final
  released state.
- If the merge strategy was squash or rebase, never tag the PR branch commit.
