"""AWS deployment entry point for the backend-only hackathon service."""

import aws_cdk as cdk
from stacks.post_training_stack import PostTrainingStack

app = cdk.App()
PostTrainingStack(
    app,
    "AutonomousPostTrainingStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)
app.synth()
