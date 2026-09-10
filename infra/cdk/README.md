# AWS deployment foundation

This CDK app provisions the AWS boundary for the backend:

- versioned private S3 artifact bucket
- encrypted DynamoDB run-state table
- immutable ECR repository
- VPC, ECS/Fargate cluster, task, and CloudWatch logs
- task and SageMaker IAM roles

The backend image must be pushed before the ECS service is started.

```bash
cd backend
uv sync --extra cloud
cd ../infra/cdk
../../backend/.venv/bin/python app.py
cdk deploy -c image_tag=latest \
  -c training_image_uri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/trainer:latest \
  -c evaluation_image_uri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/evaluator:latest
```

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
