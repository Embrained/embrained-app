# Security Policy

## Reporting a Vulnerability

The Embrained platform controls physical robots over WiFi networks. We take security seriously and appreciate responsible disclosure of any vulnerabilities.

**Please do NOT report security vulnerabilities through public GitHub Issues.**

Instead, please report them through one of the following channels:

- **Email:** Send details to the team via our [Discord server](https://discord.com/channels/1487132795833684228/1487132796920004640) using a private/direct message to a team member.
- **GitHub:** Use [GitHub Security Advisories](https://github.com/Embrained/embrained-app/security/advisories/new) to report a vulnerability privately.

### What to include

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any relevant logs, screenshots, or proof-of-concept code
- Your suggested fix (if any)

### What to expect

- **Acknowledgment** within 5 business days of your report
- **Status update** within 14 business days with our assessment
- **Credit** in the fix announcement (unless you prefer to remain anonymous)

## Scope

The following areas are in scope for security reports:

| Area | Examples |
|------|----------|
| **Network communication** | WiFi robot-to-PC protocol, WebSocket connections, FastAPI endpoints |
| **Local server** | CORS configuration, file path traversal, command injection |
| **Firmware** | ESP32-S3 OTA update integrity, serial protocol vulnerabilities |
| **Data handling** | Training data exposure, model weight exfiltration |

## Out of Scope

- Physical attacks requiring direct hardware access
- Social engineering attacks
- Denial of service attacks against local development servers
- Issues in third-party dependencies (please report those upstream)

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest `main` branch | ✅ |
| Older releases | ❌ |
