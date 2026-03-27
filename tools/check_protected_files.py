"""
Pre-commit hook: blocks commits that expose secrets or modify auth machinery.

Checks:
  1. Staged diff does not contain known secret patterns (Google OAuth,
     Stripe live/secret keys, Postgres connection strings with credentials).
  2. app/auth.py is not being modified (additions are allowed; the file is
     protected once it exists in the repo).
"""

import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Secret patterns — adjust regexes as needed
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    # Google OAuth client secret / token files
    (r'"client_secret"\s*:\s*"[^"]+"', "Google OAuth client_secret"),
    (r'"refresh_token"\s*:\s*"[^"]+"', "Google OAuth refresh_token"),
    (r'"access_token"\s*:\s*"[^"]+"', "Google OAuth access_token"),
    # Stripe live or restricted keys
    (r'\bsk_live_[A-Za-z0-9]+', "Stripe live secret key (sk_live_...)"),
    (r'\brk_live_[A-Za-z0-9]+', "Stripe restricted live key (rk_live_...)"),
    # Postgres connection strings that include a password
    (r'postgres(?:ql)?://[^:]+:[^@]+@', "Postgres connection string with credentials"),
]

# ---------------------------------------------------------------------------
# Protected files (modification blocked; new-file additions are allowed)
# ---------------------------------------------------------------------------
PROTECTED_FILES = ["app/auth.py"]


def get_staged_files():
    """Return list of (status, path) tuples for staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    entries = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            entries.append((parts[0].strip(), parts[1].strip()))
    return entries


def get_staged_diff():
    """Return the full unified diff of staged changes (text files only)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--diff-filter=ACMR", "--text"],
        capture_output=True, encoding="utf-8", errors="replace"
    )
    return result.stdout or ""


def check_protected_file_modifications(staged_files):
    violations = []
    for status, path in staged_files:
        if path in PROTECTED_FILES and status.startswith("M"):
            violations.append(path)
    return violations


def check_secrets(diff_text):
    violations = []
    for pattern, label in SECRET_PATTERNS:
        # Only check lines being added (prefixed with +)
        for line in diff_text.splitlines():
            if not line.startswith("+"):
                continue
            if re.search(pattern, line):
                violations.append(label)
                break  # one hit per pattern is enough
    return violations


def main():
    errors = []

    staged_files = get_staged_files()
    diff_text = get_staged_diff()

    protected_violations = check_protected_file_modifications(staged_files)
    for path in protected_violations:
        errors.append(
            f"BLOCKED: '{path}' is protected. Auth machinery must not be "
            f"modified without explicit approval."
        )

    secret_violations = check_secrets(diff_text)
    for label in secret_violations:
        errors.append(f"BLOCKED: Possible secret detected in staged diff — {label}")

    if errors:
        print("\n[pre-commit] Commit blocked:\n")
        for msg in errors:
            print(f"  {msg}")
        print()
        sys.exit(1)

    print("[pre-commit] Secret and auth checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
