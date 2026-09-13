# AWS deployment foundation

This CDK app provisions the AWS boundary for the backend:

- versioned private S3 artifact bucket
- encrypted DynamoDB run-state table
- immutable ECR repository
- VPC, ECS/Fargate cluster, private coordinator task, CIDR-restricted coordinator ALB, and CloudWatch logs
- task and SageMaker IAM roles

The coordinator image must be present in the stack's ECR repository before the
ECS service starts. Because this stack creates that repository, bootstrap it
with zero coordinator tasks, then push the three images and update the stack
with their immutable digests. The trainer and evaluator images are consumed by
SageMaker; build them for `linux/amd64` as well.

```bash
cd backend
docker buildx build --platform linux/amd64 --load -t post-training-backend:local -f Dockerfile .
docker buildx build --platform linux/amd64 --load -t post-training-trainer:local -f workers/trainer/Dockerfile .
docker buildx build --platform linux/amd64 --load -t post-training-evaluator:local -f workers/evaluator/Dockerfile .
cd ../infra/cdk
```

### Smoke the trainer and evaluator images

These checks require the exact, verified FunctionGemma snapshot staged from
S3 and unpacked locally. The trainer smoke also requires an NVIDIA GPU runtime
on the Docker host. Use an empty scratch directory for its one-step throwaway
adapter; this adapter has no training manifest and is not promotion-eligible.
Both containers run with networking disabled, and their smoke entrypoints force
Transformers/Hugging Face offline mode.

```bash
export FUNCTIONGEMMA_DIR=/absolute/path/to/unpacked/verified-functiongemma-snapshot
export FUNCTIONGEMMA_SMOKE_OUTPUT="$(mktemp -d)"

docker run --rm --network none --gpus all \
  -v "${FUNCTIONGEMMA_DIR}:/models/functiongemma:ro" \
  -v "${FUNCTIONGEMMA_SMOKE_OUTPUT}:/smoke-output" \
  --entrypoint python post-training-trainer:local \
  /opt/ml/code/image_smoke.py \
  --base-model-dir /models/functiongemma \
  --output-dir /smoke-output

docker run --rm --network none \
  -v "${FUNCTIONGEMMA_DIR}:/models/functiongemma:ro" \
  -v "${FUNCTIONGEMMA_SMOKE_OUTPUT}:/smoke-output:ro" \
  --entrypoint python post-training-evaluator:local \
  /opt/ml/code/image_smoke.py \
  --base-model-dir /models/functiongemma \
  --adapter-dir /smoke-output \
  --device cpu
```

The trainer check imports CUDA PyTorch and bitsandbytes, locally loads the base
in 4-bit, runs exactly one LoRA optimizer update, and writes the temporary
adapter. The evaluator independently loads that local base plus adapter and
runs a forward pass without reading sealed tasks. These are image integration
smokes only; they do not test a SageMaker job, the sealed report contract, or
checkpoint promotion.

First create the ECR repositories without starting the coordinator. Supply
syntactically valid placeholder digests for this bootstrap only; no task will
pull them while `desired_count=0`. The coordinator ALB is internet-facing on
HTTPS port 443 and its security group is restricted to an explicit operator
IPv4 CIDR. Set this from a trusted network; `/0` and networks broader than
`/16` are rejected. The same CIDR context must be passed on each deploy:

```bash
DEMO_OPERATOR_CIDR="$(curl --fail --silent --show-error -4 https://checkip.amazonaws.com | tr -d '\r\n')/32"
```

The stack outputs `CoordinatorUrl` for laptop access. This hackathon endpoint
uses HTTPS on port 443 with an imported ACM certificate and matching public
Route 53 hostname/zone. The ALB security group still restricts access to the
explicit operator IPv4 CIDR. The certificate must be issued in the stack's
region, cover the exact configured hostname (or a matching one-label wildcard),
and the supplied hosted-zone ID must refer to the named public zone.

Set the following values from an already-issued certificate and existing public
Route 53 zone before bootstrapping the stack:

```bash
COORDINATOR_CERT_ARN="arn:aws:acm:us-east-1:123456789012:certificate/replace-me"
COORDINATOR_HOST="posttraining.example.com"
COORDINATOR_ZONE_NAME="example.com"
COORDINATOR_ZONE_ID="Z0123456789EXAMPLE"
COORDINATOR_CERT_SAN="posttraining.example.com"
DEMO_OPERATOR_CIDR="198.51.100.10/32"
OBJECTIVE_WORKER_URL="https://objective.example.com/"
```

`OBJECTIVE_WORKER_URL` must identify a real authenticated HTTPS objective
service reachable from the coordinator VPC. To deploy the in-stack objective
service instead, omit that URL and provide the complete objective certificate,
private DNS, and SAN context described below; the immutable base checkpoint
must already be staged and version-pinned for this mode. Before starting a live
run in either mode, stage FunctionGemma and pass its exact versioned
`CHECKPOINT_VERSION_REF`, SHA-256, repository ID, and immutable revision.

```bash
cdk deploy \
  -c backend_image_digest=sha256:0000000000000000000000000000000000000000000000000000000000000000 \
  -c trainer_image_digest=sha256:1111111111111111111111111111111111111111111111111111111111111111 \
  -c evaluator_image_digest=sha256:2222222222222222222222222222222222222222222222222222222222222222 \
  -c coordinator_ingress_cidrs="$DEMO_OPERATOR_CIDR" \
  -c coordinator_certificate_arn="$COORDINATOR_CERT_ARN" \
  -c coordinator_public_dns_name="$COORDINATOR_HOST" \
  -c coordinator_public_hosted_zone_name="$COORDINATOR_ZONE_NAME" \
  -c coordinator_public_hosted_zone_id="$COORDINATOR_ZONE_ID" \
  -c coordinator_certificate_san="$COORDINATOR_CERT_SAN" \
  -c objective_worker_url="$OBJECTIVE_WORKER_URL" \
  -c desired_count=0
```

After pushing the local images to the three repository URIs output by CDK,
resolve each ECR `imageDigest` and deploy again with the real values and
`-c desired_count=1` plus the same ingress, certificate, DNS-name, and zone
contexts shown above. On the task-starting deploy, also pass the staged
`checkpoint_s3_uri`, `checkpoint_sha256`, `hf_repo_id`, and `hf_revision`;
`checkpoint_s3_uri` must be the exact immutable S3 `version_ref`, not a bare
key or a mutable default path.
For the internal objective-worker mode, the same deploy context passes that
versioned checkpoint and its bundle SHA-256 into the objective task. Its image
entrypoint downloads and verifies the pinned object before Uvicorn starts; a
missing object version, permission/KMS error, SHA mismatch, or invalid
FunctionGemma snapshot prevents the service from starting. No checkpoint is
baked into the image, and the coordinator container does not download it.
Pass only lowercase `sha256:` digests; mutable `:latest`
tags are rejected by the stack's image contract. Do not start a SageMaker run
until both worker images have been pushed and pinned as well.

The stack is infrastructure only. The application still requires the durable
DynamoDB/S3/provider orchestration path to be wired before a live training or
promotion result can be claimed. Local fixtures remain `EXPLANATION` evidence.

An external objective worker must be configured with an explicit HTTPS
`objective_worker_url`. For an in-stack private objective worker, supply an
issued `objective_certificate_arn`, `objective_private_dns_name`,
`objective_private_hosted_zone_name`, and `objective_certificate_san` together.
The stack creates a private Route 53 zone and ALB alias for that hostname and
fails closed unless the configured certificate SAN covers it; the ALB-generated
AWS DNS name is never used for TLS. The coordinator HTTPS certificate and public
DNS zone are separate required inputs in either objective-worker mode.
