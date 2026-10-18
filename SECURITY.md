# Security Policy

Security fixes are handled on the current development branch. This repository does not promise maintenance of older releases or a fixed response deadline.

## Report a vulnerability privately

Use **Security → Report a vulnerability** on this repository if GitHub private vulnerability reporting is enabled. If that option is unavailable, use a private contact channel explicitly listed by the repository maintainer. No dedicated security email or alternative private channel is established by this file.

Avoid posting exploit details, credentials, or private document contents in public issues. If no private channel is listed, a public issue can request a private reporting channel without disclosing the vulnerability itself.

Include affected versions, reproduction steps using a synthetic file where possible, and the security impact. Maintainer availability determines response and release timing.

## Local processing

The scope includes the Python package and repository automation. PDFs are parsed by third-party libraries; process untrusted inputs with appropriate filesystem permissions and keep dependencies updated. Optional engines may download model files during setup or first use.

Keep private papers, extracted text, logs, and notes in local ignored directories. Review files before sharing a bundle or submitting a pull request, and never commit credentials.
