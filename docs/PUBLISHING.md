# Publishing under a new GitHub account

This tree is independent of the previous repository.
It contains no Git history, remotes, credentials, or account-specific workflow secrets.
The previous source origin is retained only for attribution and traceability.

## Before publishing

Run the result reconstruction, tests, and source audit.
Review the README result scope and asset-access description.
Choose the intended owner and repository name, for example `YOUR_ACCOUNT/Maze`.
Keep the code license and source attributions.
Do not add the original update packet or private working directories to the public repository.

## Create and push

From this source tree, after authenticating GitHub CLI to the new account:

```bash
git init -b main
git add .
git commit -m "Initialize Maze project"
gh repo create YOUR_ACCOUNT/Maze --public --source=. --remote=origin --push
```

The commit uses your configured Git identity.
No command in the source tree creates a repository or pushes automatically.
The README uses relative project links, so it does not depend on a guessed new account name.
After publication, add the repository URL to your package metadata, citation metadata, and paper code-availability statement.
Set the GitHub description to:

> Prompt-trajectory planning for efficient LLM reasoning. Maze methods, variants, reproducible experiments, and deployment tools.

Suggested topics: `test-time-scaling`, `llm-reasoning`, `in-context-learning`, `prompt-optimization`, `neurips-2026`, `reproducible-research`.

## Maintain a project, not a single-paper snapshot

Use normal versioned releases for source changes and stable experiment identifiers for measurements.
A release can support multiple research milestones without rewriting an earlier result record.
Publish large authorized trained assets separately when available, with their configurations and access conditions.
Keep source, datasets, and checkpoint permissions explicit.

The included GitHub Actions workflow has read-only repository permissions and needs no model API key.
It reconstructs the numeric evidence and runs CPU tests; it never launches an external language model.
