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
