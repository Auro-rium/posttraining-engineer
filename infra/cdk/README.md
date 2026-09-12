# AWS deployment foundation

This CDK app provisions the AWS boundary for the backend:

- versioned private S3 artifact bucket
- encrypted DynamoDB run-state table
- immutable ECR repository
- VPC, ECS/Fargate cluster, task, and CloudWatch logs
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

First create the ECR repositories without starting the coordinator. Supply
syntactically valid placeholder digests for this bootstrap only; no task will
pull them while `desired_count=0`.

```bash
cdk deploy \
  -c backend_image_digest=sha256:0000000000000000000000000000000000000000000000000000000000000000 \
  -c trainer_image_digest=sha256:1111111111111111111111111111111111111111111111111111111111111111 \
  -c evaluator_image_digest=sha256:2222222222222222222222222222222222222222222222222222222222222222 \
  -c desired_count=0
```

After pushing the local images to the three repository URIs output by CDK,
resolve each ECR `imageDigest` and deploy again with the real values and
`-c desired_count=1`. Pass only lowercase `sha256:` digests; mutable `:latest`
tags are rejected by the stack's image contract. Do not start a SageMaker run
until both worker images have been pushed and pinned as well.

The stack is infrastructure only. The application still requires the durable
DynamoDB/S3/provider orchestration path to be wired before a live training or
promotion result can be claimed. Local fixtures remain `EXPLANATION` evidence.

An external objective worker must be configured with an explicit HTTPS
`objective_worker_url`. For an in-stack private objective worker, supply
`objective_certificate_arn`, `objective_private_dns_name`,
`objective_private_hosted_zone_name`, and `objective_certificate_san` together.
The stack creates a private Route 53 zone and ALB alias for that hostname and
fails closed unless the configured certificate SAN covers it; the ALB-generated
AWS DNS name is never used for coordinator TLS.
