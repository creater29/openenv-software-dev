#!/usr/bin/env bash
# validate-submission.sh
# Pre-submission validation script.
# Usage:  bash validate-submission.sh [hf_space_url]
# Example: bash validate-submission.sh https://huggingface.co/spaces/EndlessMarathon/openenv-software-dev

set -euo pipefail

HF_SPACE_URL="${1:-https://huggingface.co/spaces/EndlessMarathon/openenv-software-dev}"
PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  ✅  $desc"
        PASS=$((PASS + 1))
    else
        echo "  ❌  $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " OpenEnv Software-Dev — Submission Validator"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📦 Checking required files..."
check "openenv.yaml exists"           "test -f openenv.yaml"
check "Dockerfile exists"             "test -f Dockerfile"
check "pyproject.toml exists"         "test -f pyproject.toml"
check "inference.py exists"           "test -f inference.py"
check "README.md exists"              "test -f README.md"
check "configs/default.yaml exists"   "test -f configs/default.yaml"
check "validate-submission.sh exists" "test -f validate-submission.sh"

echo ""
echo "🐍 Checking Python package structure..."
check "__init__.py present"           "test -f openenv_software_dev/__init__.py"
check "env.py present"                "test -f openenv_software_dev/env.py"
check "actions.py present"            "test -f openenv_software_dev/actions.py"
check "reward.py present"             "test -f openenv_software_dev/reward.py"
check "graders/composite.py present"  "test -f openenv_software_dev/graders/composite.py"
check "tasks/bug_fix.py present"      "test -f openenv_software_dev/tasks/bug_fix.py"
check "tasks/feature_impl.py present" "test -f openenv_software_dev/tasks/feature_impl.py"
check "sandbox/executor.py present"   "test -f openenv_software_dev/sandbox/executor.py"

echo ""
echo "⚙️  Checking Python imports..."
check "Package imports cleanly" \
    "python3 -c 'from openenv_software_dev.env import SoftwareDevEnv'"
check "Gymnasium registration works" \
    "python3 -c 'import gymnasium as gym; gym.make(\"SoftwareDev-v0\")'"
check "Reset + step cycle works" \
    "python3 -c '
from openenv_software_dev.env import SoftwareDevEnv
env = SoftwareDevEnv()
obs, _ = env.reset()
env.step({\"type\": 3, \"target_file\": \"\", \"text_input\": \"\"})
'"

echo ""
echo "🧪 Running test suite..."
if python3 -m pytest tests/ -q --tb=short 2>&1; then
    echo "  ✅  All tests passed"
    PASS=$((PASS + 1))
else
    echo "  ❌  Some tests failed (see above)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "📋 Checking inference script output format..."
INFERENCE_OUT=$(timeout 30 python3 inference.py 2>/dev/null || true)
check "[START] line present" "echo '$INFERENCE_OUT' | grep -q '\\[START\\]'"
check "[STEP] line present"  "echo '$INFERENCE_OUT' | grep -q '\\[STEP\\]'"
check "[END] line present"   "echo '$INFERENCE_OUT' | grep -q '\\[END\\]'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo " 🎉 Submission is VALID — ready to push!"
else
    echo " ⚠️  Fix the issues above before submitting."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit $FAIL
