#!/usr/bin/env bash
set -euo pipefail

# One-time host setup for shared persistent assets under /data.
# Run as a user that has permission to create/chown/chmod these paths.

DATA_ROOT="${1:-/data/llm_spine_parse}"
SHARED_GROUP="${2:-llmtrain}"

echo "Preparing campus data root: ${DATA_ROOT}"

sudo mkdir -p "${DATA_ROOT}"/{hf_home,checkpoints,input,img,logs}

# Group-shared strategy: both users join SHARED_GROUP.
sudo chgrp -R "${SHARED_GROUP}" "${DATA_ROOT}"

# setgid preserves group on new files/dirs; group rwx for collaboration.
sudo chmod 2775 "${DATA_ROOT}"
sudo chmod -R 2775 "${DATA_ROOT}"/{hf_home,checkpoints,logs}

# Input and image assets are usually managed by admins; keep writable by group by default.
sudo chmod -R 2775 "${DATA_ROOT}"/{input,img}

cat <<'EOF'
Done.
Next steps:
1) Ensure both users are in the shared group.
2) Start new shell/session so group membership is active.
3) Edit docker_campus/.env.campus and adjust values.
EOF
