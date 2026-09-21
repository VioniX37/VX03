# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | Yes       |
| < 2.0   | No        |

## What counts as a security issue

- Vulnerabilities in the REST API (`sovereign_opt/server.py`) or the dashboard, for example unsafe
  handling of uploaded model files.
- Inputs that make the independent validator certify a result that is not actually feasible or optimal.
- Vulnerable dependencies.

Wrong answers that the validator correctly reports as `FAILED` are ordinary bugs; please open a normal issue.

## Reporting a vulnerability

1. **Do not open a public issue.**
2. Report it privately through GitHub: **Security → Report a vulnerability** on this repository
   (<https://github.com/VioniX37/VX03/security/advisories/new>).
3. Include a description and impact, a minimal reproduction (for example an `.mps` or `.lp` file), and
   the behaviour you expected.

We aim to acknowledge reports within 7 days and will coordinate a fix and disclosure with you.

## Deployment note

The API server has no authentication and is intended for local or trusted-network use. Do not expose it
directly to the internet.
