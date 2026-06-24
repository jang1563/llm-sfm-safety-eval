#!/usr/bin/env bash
set -euo pipefail

anthropic_prefix='sk-ant-'
openai_prefix='sk-'
hf_prefix='hf_'

github_token='(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}'
github_pat='github_pat_[A-Za-z0-9_]{20,}'
aws_access_key='AKIA[0-9A-Z]{16}'
google_api_key='AIza[0-9A-Za-z_-]{35}'
private_key='BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY'
slack_token='xox[baprs]-[A-Za-z0-9-]{10,}'
sendgrid_key='SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}'

pattern="(${anthropic_prefix}[A-Za-z0-9_-]{20,}|${openai_prefix}[A-Za-z0-9_-]{30,}|${hf_prefix}[A-Za-z0-9]{20,}|${github_token}|${github_pat}|${aws_access_key}|${google_api_key}|${private_key}|${slack_token}|${sendgrid_key}|OPENAI_API_KEY=${openai_prefix}|ANTHROPIC_API_KEY=${openai_prefix})"

set +e
if command -v rg >/dev/null 2>&1; then
  rg -n --hidden "$pattern" . \
    -g '!.git/**' \
    -g '!pilot/__pycache__/**' \
    -g '!paper/**/.gitignore'
  status=$?
else
  grep -RInE --binary-files=without-match \
    --exclude-dir=.git \
    --exclude-dir=__pycache__ \
    "$pattern" .
  status=$?
fi
set -e

if [ "$status" -eq 0 ]; then
  echo "Potential secret patterns found." >&2
  exit 1
fi

if [ "$status" -eq 1 ]; then
  echo "No secret patterns found."
  exit 0
fi

exit "$status"
