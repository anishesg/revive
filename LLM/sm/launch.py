#!/usr/bin/env python3
"""Launch a REVIVE role-training job on SageMaker.

Uses the default AWS profile already configured on this host. Bootstraps
missing resources on first run: execution role (revive-sagemaker-execution-role)
and S3 bucket (sagemaker-<region>-<account>). Then submits a managed training
job that runs LLM/sagemaker/train_entry.py on a single GPU instance and writes
model.tar.gz to S3.

Typical usage:
    python LLM/sagemaker/launch.py --roles reasoner --epochs 1
    python LLM/sagemaker/launch.py --roles all --epochs 3 --instance ml.g5.2xlarge
    python LLM/sagemaker/launch.py --dry-run   # print plan, don't submit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.huggingface import HuggingFace

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = REPO_ROOT / "LLM"

DEFAULT_ROLE_NAME = "revive-sagemaker-execution-role"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}


def ensure_execution_role(iam, role_name: str) -> str:
    try:
        role = iam.get_role(RoleName=role_name)
        return role["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
    print(f"[iam] creating role {role_name}")
    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
        Description="REVIVE SageMaker training execution role",
    )
    for arn in (
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ):
        iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)
    # IAM is eventually consistent — SageMaker may refuse for ~10s after create.
    print("[iam] waiting 15s for role propagation")
    time.sleep(15)
    return role["Role"]["Arn"]


def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code not in ("404", "NoSuchBucket", "403"):
            # 403 means bucket exists but we don't own it — surface that.
            if code == "403":
                raise RuntimeError(f"bucket {bucket} exists in another account")
            raise
    print(f"[s3] creating bucket {bucket}")
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roles", default="reasoner",
                   help="Comma-separated role names, or 'all'.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--synthetic-n", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--instance", default="ml.g5.xlarge",
                   help="SageMaker training instance type.")
    p.add_argument("--max-run-hours", type=float, default=4.0)
    p.add_argument("--spot", action="store_true",
                   help="Use managed spot training (cheaper, interruptible).")
    p.add_argument("--role-name", default=DEFAULT_ROLE_NAME)
    p.add_argument("--bucket", default=None,
                   help="S3 bucket for code/output. Default: sagemaker-<region>-<account>.")
    p.add_argument("--job-name", default=None)
    p.add_argument("--wait", action="store_true",
                   help="Stream logs until the job finishes.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    session = boto3.Session()
    region = session.region_name or "us-east-1"
    sts = session.client("sts")
    account = sts.get_caller_identity()["Account"]
    iam = session.client("iam")
    s3 = session.client("s3", region_name=region)

    role_arn = ensure_execution_role(iam, args.role_name)
    bucket = args.bucket or f"sagemaker-{region}-{account}"
    ensure_bucket(s3, bucket, region)

    sm_session = sagemaker.Session(boto_session=session, default_bucket=bucket)

    job_name = args.job_name or f"revive-{args.roles.replace(',', '-')}-{int(time.time())}"
    # SageMaker caps job names at 63 chars.
    job_name = job_name[:63].rstrip("-")

    hyperparameters = {
        "roles": args.roles,
        "epochs": args.epochs,
        "batch": args.batch,
        "lora-r": args.lora_r,
        "lr": args.lr,
        "synthetic-n": args.synthetic_n,
        "max-seq-len": args.max_seq_len,
    }

    # AWS HuggingFace DLC: pytorch 2.5.1 + transformers 4.49 + CUDA 12.4 +
    # py311. Unsloth installs cleanly on top via requirements.txt.
    estimator_kwargs = dict(
        entry_point="train_entry.py",
        source_dir=str(LLM_DIR),
        dependencies=[],
        role=role_arn,
        instance_type=args.instance,
        instance_count=1,
        transformers_version="4.49.0",
        pytorch_version="2.5.1",
        py_version="py311",
        hyperparameters=hyperparameters,
        max_run=int(args.max_run_hours * 3600),
        base_job_name="revive",
        sagemaker_session=sm_session,
        output_path=f"s3://{bucket}/revive-sagemaker/output/",
        code_location=f"s3://{bucket}/revive-sagemaker/code/",
        environment={
            "TRANSFORMERS_VERBOSITY": "info",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        },
        disable_profiler=True,
    )
    if args.spot:
        estimator_kwargs.update(
            use_spot_instances=True,
            max_wait=int(args.max_run_hours * 3600) + 1800,
            checkpoint_s3_uri=f"s3://{bucket}/revive-sagemaker/checkpoints/{job_name}/",
        )

    # source_dir points at LLM/; entry script lives under sm/ to avoid
    # shadowing the installed `sagemaker` package on PYTHONPATH.
    estimator_kwargs["entry_point"] = "sm/train_entry.py"

    estimator = HuggingFace(**estimator_kwargs)

    print("=== SageMaker launch plan ===")
    print(f"  account:   {account}")
    print(f"  region:    {region}")
    print(f"  role:      {role_arn}")
    print(f"  bucket:    s3://{bucket}/revive-sagemaker/")
    print(f"  instance:  {args.instance} (spot={args.spot})")
    print(f"  job name:  {job_name}")
    print(f"  roles:     {args.roles}")
    print(f"  epochs:    {args.epochs}   batch: {args.batch}   lora_r: {args.lora_r}")
    print(f"  max_run:   {args.max_run_hours}h")

    if args.dry_run:
        print("[dry-run] not submitting.")
        return 0

    estimator.fit(wait=args.wait, job_name=job_name)

    print("\n=== Submitted ===")
    print(f"  job:     {job_name}")
    print(f"  console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    print(f"  logs:    aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")
    print(f"  model:   s3://{bucket}/revive-sagemaker/output/{job_name}/output/model.tar.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
