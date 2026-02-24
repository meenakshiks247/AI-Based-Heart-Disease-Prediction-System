# Commit Convention — Heart Disease Prediction System

## Format

```
<type>: <WHAT changed> — <WHY it was done>
```

**Subject line must be under 80 characters.**

## Allowed Types

| Type       | When to use                                  |
|------------|----------------------------------------------|
| `feat`     | New feature or capability                    |
| `fix`      | Bug fix or correction                        |
| `refactor` | Code restructuring (no behavior change)      |
| `perf`     | Performance improvement                      |
| `docs`     | Documentation only                           |
| `test`     | Adding or updating tests                     |
| `style`    | Formatting, linting, whitespace (no logic)   |

## Rules

1. **Explain WHAT** was done — the subject starts after `<type>:`.
2. **Explain WHY** it was necessary — follows the em dash `—`.
3. **Under 80 characters** — keep it scannable in `git log --oneline`.
4. **Imperative mood** — "Add", "Fix", "Remove" (not "Added", "Fixes").
5. **No vague messages** — never write "update", "changes", "misc", "fix stuff".

## Examples

```
feat: Add cardio dataset training pipeline — enable large-scale risk prediction
fix: Remove duplicate patient records before training — prevent memorization
refactor: Centralize model loading logic — avoid double preprocessing
docs: Add experiment comparison explanation — clarify performance differences
test: Add model artifact existence checks — catch missing files early
perf: Cache model bundle as singleton — eliminate repeated disk I/O
style: Apply black formatting to ml/ scripts — enforce consistent code style
```

## Setup

The repo includes a `.gitmessage` template. Activate it locally:

```bash
git config commit.template .gitmessage
```

This pre-fills the template (as comments) every time you run `git commit`.
