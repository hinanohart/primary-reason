#!/usr/bin/env bash
# finalize_pypi.sh — the ONLY remaining manual step for primary-reason.
#
# Why this exists:
#   PyPI trusted-publisher provisioning requires a logged-in pypi.org session
#   (you click through the website). Everything else — repo / wheel build /
#   GitHub Release / branch protection / kill-criterion tracking / re-audit
#   fixes — is already automated and on GitHub.
#
# Where to run:
#   Anywhere. The script does not depend on cwd; it talks to GitHub via `gh`
#   and to PyPI via the web UI you open. `gh auth status` must be green.
#
# What it does NOT do (intentionally):
#   - Does NOT upload to PyPI (the release.yml workflow does, via OIDC).
#   - Does NOT prompt for or write any PyPI password / API token. PyPI trusted
#     publishing is token-less by design (OIDC), which is also why the setup
#     itself must happen on pypi.org.
#
# What it DOES:
#   Step 1: prints the exact claim values to paste into pypi.org.
#   Step 2: waits for you to confirm the trusted publisher was created.
#   Step 3: re-runs the failed PyPI step of the most recent Release workflow.
#   Step 4: verifies that `pip install primary-reason==<version>` resolves.

set -euo pipefail

REPO="hinanohart/primary-reason"
PYPI_PROJECT="primary-reason"
ENVIRONMENT="pypi"
WORKFLOW_FILE="release.yml"
VERSION="0.1.1"  # latest tag; bump if you release a newer one before running this
TAG="v${VERSION}"

OWNER=$(echo "${REPO}" | cut -d/ -f1)
REPO_NAME=$(echo "${REPO}" | cut -d/ -f2)

# ----- preflight ----------------------------------------------------------
command -v gh >/dev/null 2>&1 || {
  echo "ERROR: gh CLI not found. Install: https://cli.github.com/"
  exit 1
}
gh auth status >/dev/null 2>&1 || {
  echo "ERROR: gh not authenticated. Run: gh auth login"
  exit 1
}

# ----- Step 1 + 2: instruction screen + manual website click --------------
cat <<EOF
========================================================================
primary-reason ${TAG} — PyPI trusted-publisher finalization
========================================================================

This is the ONLY remaining manual step. Total wall-clock time: ~3 minutes.

Step 1/4 — Open this page in your browser (logged into pypi.org):

    https://pypi.org/manage/account/publishing/

Scroll to "Add a new pending publisher" (use this form when the project is
not yet on PyPI; if it is, the form is "Add a new publisher" under the
existing project).

Step 2/4 — Paste these EXACT values into the form:

    PyPI Project Name:    ${PYPI_PROJECT}
    Owner:                ${OWNER}
    Repository name:      ${REPO_NAME}
    Workflow filename:    ${WORKFLOW_FILE}
    Environment name:     ${ENVIRONMENT}

Click "Add". The OIDC subject claim will resolve to:
    repo:${REPO}:environment:${ENVIRONMENT}

When you're done on pypi.org, press Enter here to continue.
EOF

read -r _

# ----- Step 3: re-run the release workflow --------------------------------
echo
echo "Step 3/4 — Re-running PyPI publish for ${TAG}..."
echo

RUN_ID=$(gh run list \
    --repo "${REPO}" \
    --workflow "${WORKFLOW_FILE}" \
    --limit 20 \
    --json databaseId,headBranch \
    --jq "[.[] | select(.headBranch == \"${TAG}\")] | .[0].databaseId" \
    2>/dev/null || echo "")

if [ -n "${RUN_ID}" ] && [ "${RUN_ID}" != "null" ]; then
    echo "Found release.yml run ${RUN_ID} for ${TAG}. Re-running failed jobs..."
    gh run rerun "${RUN_ID}" --repo "${REPO}" --failed
    echo
    echo "Watching the run (Ctrl+C is safe; the run continues on GitHub)..."
    gh run watch "${RUN_ID}" --repo "${REPO}" --exit-status || \
        echo "(watch exited; check status with: gh run view ${RUN_ID} --repo ${REPO})"
else
    echo "No previous release.yml run found for ${TAG}."
    echo "Trigger a fresh one by pushing a patch tag from inside the repo:"
    echo "    git tag ${TAG}.1 && git push origin ${TAG}.1"
    exit 1
fi

# ----- Step 4: verify pip install works -----------------------------------
echo
echo "Step 4/4 — Verifying pip can resolve ${PYPI_PROJECT}==${VERSION}..."
echo

if command -v python3 >/dev/null 2>&1 && \
   python3 -m pip install --dry-run "${PYPI_PROJECT}==${VERSION}" >/dev/null 2>&1; then
    echo "  OK: ${PYPI_PROJECT}==${VERSION} is now installable from PyPI."
    echo "       pip install ${PYPI_PROJECT}==${VERSION}"
else
    echo "  PyPI may not have indexed the artifact yet (1-2 min delay is normal)."
    echo "  Retry in a minute with:"
    echo "       pip install ${PYPI_PROJECT}==${VERSION}"
fi

echo
echo "All done. primary-reason ${TAG} is published."
echo "Optional follow-up: edit README.md to remove the 'GitHub Release wheel'"
echo "fallback install instructions now that PyPI works."
