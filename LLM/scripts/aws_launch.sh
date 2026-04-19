#!/usr/bin/env bash
# Launch a g5.xlarge spot instance that runs the full REVIVE training pipeline
# and uploads GGUF artifacts to S3 when done.
#
# Prerequisites:
#   aws CLI configured (aws configure)
#   Permissions: ec2:RunInstances, ec2:DescribeImages, iam:CreateRole,
#                iam:AttachRolePolicy, iam:CreateInstanceProfile,
#                iam:AddRoleToInstanceProfile, s3:CreateBucket
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-...
#   export S3_BUCKET=my-revive-models
#   bash LLM/scripts/aws_launch.sh
#   bash LLM/scripts/aws_launch.sh --dry-run   # print config, don't launch
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
: "${S3_BUCKET:?Set S3_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
DRY_RUN="${1:-}"

# Infer GitHub repo from remote origin
GITHUB_REPO="$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]\(.*\)\.git|\1|' || echo 'your-org/revive')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Resolve latest Deep Learning AMI ─────────────────────────────────────────
echo "Resolving latest Deep Learning AMI in ${AWS_REGION}..."
AMI_ID="$(aws ec2 describe-images \
    --owners amazon \
    --region "${AWS_REGION}" \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)*" \
        "Name=architecture,Values=x86_64" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)"

if [ -z "${AMI_ID}" ] || [ "${AMI_ID}" = "None" ]; then
    echo "ERROR: Could not find Deep Learning AMI. Check AWS region and permissions."
    exit 1
fi
echo "  AMI: ${AMI_ID}"

# ── IAM role for S3 access ────────────────────────────────────────────────────
ROLE_NAME="revive-training-role"
PROFILE_NAME="revive-training-profile"

if ! aws iam get-role --role-name "${ROLE_NAME}" --region "${AWS_REGION}" &>/dev/null; then
    echo "Creating IAM role ${ROLE_NAME}..."
    aws iam create-role --role-name "${ROLE_NAME}" \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
        }' --output text --query "Role.RoleName" > /dev/null

    aws iam attach-role-policy --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    aws iam attach-role-policy --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonEC2FullAccess"

    aws iam create-instance-profile --instance-profile-name "${PROFILE_NAME}" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "${PROFILE_NAME}" \
        --role-name "${ROLE_NAME}"

    echo "  Waiting for IAM propagation..."
    sleep 15
fi

# ── Ensure S3 bucket exists ───────────────────────────────────────────────────
if ! aws s3api head-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "Creating S3 bucket s3://${S3_BUCKET}..."
    if [ "${AWS_REGION}" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}"
    else
        aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}" \
            --create-bucket-configuration LocationConstraint="${AWS_REGION}"
    fi
fi

# ── Prepare user-data script ──────────────────────────────────────────────────
USER_DATA_TEMPLATE="${SCRIPT_DIR}/aws_user_data.sh"
USER_DATA_TMP="$(mktemp)"
trap 'rm -f "${USER_DATA_TMP}"' EXIT

sed \
    -e "s|__ANTHROPIC_API_KEY__|${ANTHROPIC_API_KEY}|g" \
    -e "s|__S3_BUCKET__|${S3_BUCKET}|g" \
    -e "s|__GITHUB_REPO__|${GITHUB_REPO}|g" \
    -e "s|__AWS_REGION__|${AWS_REGION}|g" \
    "${USER_DATA_TEMPLATE}" > "${USER_DATA_TMP}"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== Launch config ==="
echo "  Instance:   ${INSTANCE_TYPE}"
echo "  AMI:        ${AMI_ID}"
echo "  Region:     ${AWS_REGION}"
echo "  S3 output:  s3://${S3_BUCKET}/revive-models/"
echo "  Repo:       https://github.com/${GITHUB_REPO}"
echo "  Est. cost:  \$1.50–6 (spot) | \$5–6 (on-demand)"
echo "  Est. time:  5–6 hours"
echo ""

if [ "${DRY_RUN}" = "--dry-run" ]; then
    echo "[dry-run] Skipping launch. User-data preview:"
    head -20 "${USER_DATA_TMP}"
    echo "..."
    exit 0
fi

# ── Launch spot instance ──────────────────────────────────────────────────────
echo "Launching spot instance..."
INSTANCE_ID="$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --region "${AWS_REGION}" \
    --iam-instance-profile Name="${PROFILE_NAME}" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --user-data "file://${USER_DATA_TMP}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=revive-training},{Key=Project,Value=revive}]" \
    --query "Instances[0].InstanceId" \
    --output text)"

echo ""
echo "=== Instance launched ==="
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Region:      ${AWS_REGION}"
echo ""
echo "Monitor logs:"
echo "  aws ec2 get-console-output --instance-id ${INSTANCE_ID} --region ${AWS_REGION}"
echo ""
echo "When training completes (~6hrs), pull models:"
echo "  S3_BUCKET=${S3_BUCKET} bash LLM/scripts/aws_pull.sh"
