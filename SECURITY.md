# Security

Keep model-provider credentials in environment variables or your own secret manager.
The collector uses HTTPS except for a loopback inference server, rejects redirects, and does not record provider exception strings.
Do not post secrets or private benchmark content in public issues.

New checkpoints use numerical NPZ files with `allow_pickle=False` plus JSON metadata.
Treat externally supplied code and assets as untrusted, verify their source, and review them before execution.
The project does not automatically execute remote model code.

For a security concern involving private material, contact a project maintainer through the contact information on the linked paper rather than opening a public issue containing the material.
