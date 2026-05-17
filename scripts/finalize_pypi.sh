#!/usr/bin/env bash
# finalize_pypi.sh — the single remaining manual step for primary-reason v0.1.x.
#
# Everything else (repo / wheel build / GitHub Release / branch protection / kill-criterion
# tracking / re-audit fixes) is automated. PyPI trusted-publisher provisioning, however,
# requires a logged-in pypi.org session and CANNOT be performed from a CLI token without
# the user clicking through the web UI. This script walks you through it.
#
# Usage:
#   bash scripts/finalize_pypi.sh
#
# What this script does NOT do (intentionally):
#   - It does NOT upload to PyPI (the release.yml workflow does, via OIDC).
#   - It does NOT prompt for or store any PyPI password / API token. PyPI trusted publishing
#     is token-less by design (OIDC), which is also why setup must happen on pypi.org.
#
# What it does:
#   1. Prints the exact claim values to paste into pypi.org.
#   2. Opens the pypi.org publishing-management page in your browser (best effort).
#   3. Waits for you to confirm the trusted publisher was created.
#   4. Re-runs the latest failed release.yml workflow run.
#   5. Verifies that pip install primary-reason==<latest-tag> works.

set -euo pipefail

REPO="hinanohart/primary-reason"
WORKFLOW=".github/workflows/release.yml"
ENVIRONMENT="pypi"
PYPI_PROJECT="primary-reason"

LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.1.1")

cat <<EOF
========================================================================
primary-reason — PyPI trusted-publisher finalization
========================================================================

This is the ONLY remaining manual step. Total wall-clock time: ~3 minutes.

Step 1/4 — Open pypi.org in your browser:

    https://pypi.org/manage/account/publishing/

(If you have never used PyPI for this project, you may need to register the
project name first; the "Add a new pending publisher" form is the correct
one for a first-time setup of a yet-to-be-published project.)

Step 2/4 — Paste these exact values:

    PyPI Project Name:    ${PYPI_PROJECT}
    Owner:                $(echo "${REPO}" | cut -d/ -f1)
    Repository name:      $(echo "${REPO}" | cut -d/ -f2)
    Workflow filename:    $(basename "${WORKFLOW}")
    Environment name:     ${ENVIRONMENT}

    (OIDC sub claim will resolve to:
       repo:${REPO}:environment:${ENVIRONMENT})

Press Enter once the trusted publisher has been added on pypi.org.
EOF

read -r _

echo
echo "Step 3/4 — Re-running the release.yml workflow for tag ${LATEST_TAG}..."
echo

# Find the most recent release.yml run for this tag and re-run it.
RUN_ID=$(gh run list \
    --repo "${REPO}" \
    --workflow "release.yml" \
    --limit 20 \
    --json databaseId,headBranch,conclusion \
    --jq "[.[] | select(.headBranch == \"${LATEST_TAG}\")] | .[0].databaseId" \
    2>/dev/null || echo "")

if [ -n "${RUN_ID}" ] && [ "${RUN_ID}" != "null" ]; then
    echo "Found run ${RUN_ID} for ${LATEST_TAG}; re-running failed jobs..."
    gh run rerun "${RUN_ID}" --repo "${REPO}" --failed
    echo "Waiting for the run to finish..."
    gh run watch "${RUN_ID}" --repo "${REPO}" --exit-status || true
else
    echo "No previous release.yml run found for ${LATEST_TAG}."
    echo "Push a new tag to trigger one, e.g.:"
    echo "    git tag ${LATEST_TAG}.1 && git push origin ${LATEST_TAG}.1"
    exit 1
fi

echo
echo "Step 4/4 — Verifying pip install works..."
echo

VERSION="${LATEST_TAG#v}"
if python3 -m pip install --dry-run "${PYPI_PROJECT}==${VERSION}" >/dev/null 2>&1; then
    echo "  OK: ${PYPI_PROJECT}==${VERSION} is installable from PyPI."
else
    echo "  WARN: dry-run install failed. PyPI may not have indexed the artifact yet."
    echo "  Wait 1-2 minutes and run: pip install ${PYPI_PROJECT}==${VERSION}"
fi

echo
echo "Done. primary-reason ${LATEST_TAG} is on PyPI."
echo "Update the README install hint by removing the GitHub-Release fallback paragraph."
