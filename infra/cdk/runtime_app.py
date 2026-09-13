"""CDK entry point for the immutable-artifact-consuming runtime phase."""

import aws_cdk as cdk

from stacks.runtime_stack import PostTrainingRuntimeStack

app = cdk.App()
PostTrainingRuntimeStack(
    app,
    "AutonomousPostTrainingRuntime",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    deployment_id=app.node.try_get_context("deployment_id") or "demo",
)
app.synth()
