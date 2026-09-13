"""Focused contract tests for the AWS bootstrap stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Template

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stacks.bootstrap_stack import PostTrainingBootstrapStack  # noqa: I001


OUTPUT_SUFFIXES = {
    "ArtifactBucketName",
    "ArtifactKeyArn",
    "StateTableName",
    "SageMakerRoleArn",
    "BackendRepositoryName",
    "TrainerRepositoryName",
    "EvaluatorRepositoryName",
    "RuntimeLogGroupName",
    "ObjectiveLogGroupName",
    "VpcId",
    "AvailabilityZones",
    "PrivateSubnetIds",
    "PrivateSubnetRouteTableIds",
    "PublicSubnetIds",
    "PublicSubnetRouteTableIds",
    "ApprovalSecretArn",
    "ObjectiveCredentialSecretArn",
}


def template(
    context: dict[str, str] | None = None, *, deployment_id: str | None = None
) -> Template:
    app = App(context=context or {})
    stack = PostTrainingBootstrapStack(
        app,
        "AutonomousPostTrainingBootstrap",
        env=Environment(account="123456789012", region="us-east-1"),
        deployment_id=deployment_id,
    )
    return Template.from_stack(stack)


def test_synthesizes_with_empty_runtime_context_and_default_deployment_id() -> None:
    rendered = template().to_json()

    assert rendered["Resources"]
    exports = {
        output["Export"]["Name"]
        for output in rendered["Outputs"].values()
        if "Export" in output
    }
    assert exports == {f"aptraining-demo-{suffix}" for suffix in OUTPUT_SUFFIXES}

def test_bootstrap_provisions_encrypted_versioned_artifacts_and_durable_state() -> None:
    rendered = template()
    buckets = rendered.find_resources("AWS::S3::Bucket")
    assert len(buckets) == 1
    bucket = next(iter(buckets.values()))["Properties"]
    assert bucket["VersioningConfiguration"] == {"Status": "Enabled"}
    assert bucket["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"
    ]["SSEAlgorithm"] == "aws:kms"
    assert bucket["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }

    keys = rendered.find_resources("AWS::KMS::Key")
    assert len(keys) == 1
    assert next(iter(keys.values()))["Properties"]["EnableKeyRotation"] is True

    tables = rendered.find_resources("AWS::DynamoDB::Table")
    assert len(tables) == 1
    table = next(iter(tables.values()))["Properties"]
    assert table["BillingMode"] == "PAY_PER_REQUEST"
    assert table["KeySchema"] == [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ]
    assert table["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True
    assert table["SSESpecification"]["SSEEnabled"] is True

    bucket_id, bucket_resource = next(iter(buckets.items()))
    assert "BucketName" not in bucket_resource["Properties"]
    artifact_bucket_output = next(
        output
        for output in rendered.to_json()["Outputs"].values()
        if output.get("Export", {}).get("Name") == "aptraining-demo-ArtifactBucketName"
    )
    assert artifact_bucket_output["Value"] == {"Ref": bucket_id}


def test_repositories_are_immutable_and_scanned_without_image_dependencies() -> None:
    repositories = template().find_resources("AWS::ECR::Repository")
    assert len(repositories) == 3
    names = {item["Properties"]["RepositoryName"] for item in repositories.values()}
    assert names == {
        "aptraining-demo-backend",
        "aptraining-demo-trainer",
        "aptraining-demo-evaluator",
    }
    for repository in repositories.values():
        properties = repository["Properties"]
        assert properties["ImageTagMutability"] == "IMMUTABLE"
        assert properties["ImageScanningConfiguration"] == {"ScanOnPush": True}


def test_sagemaker_role_has_scoped_execution_permissions() -> None:
    rendered = template()
    roles = rendered.find_resources("AWS::IAM::Role")
    assert len(roles) == 1
    role = next(iter(roles.values()))["Properties"]
    assert role["AssumeRolePolicyDocument"]["Statement"][0]["Principal"] == {
        "Service": "sagemaker.amazonaws.com"
    }

    policies = rendered.find_resources("AWS::IAM::Policy")
    assert len(policies) == 1
    document = next(iter(policies.values()))["Properties"]["PolicyDocument"]
    serialized = json.dumps(document)
    for action in (
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:PutObject",
        "kms:Decrypt",
        "kms:GenerateDataKey",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    ):
        assert action in serialized
    wildcard_statements = [
        statement
        for statement in document["Statement"]
        if statement.get("Resource") == "*"
    ]
    assert all(
        statement.get("Action") == "ecr:GetAuthorizationToken"
        for statement in wildcard_statements
    )


def test_logs_secrets_and_network_are_ready_for_later_private_runtime() -> None:
    rendered = template()
    groups = rendered.find_resources("AWS::Logs::LogGroup")
    assert len(groups) == 2
    group_names = {group["Properties"]["LogGroupName"] for group in groups.values()}
    assert group_names == {
        "/aws/sagemaker/aptraining-demo/runtime",
        "/aws/sagemaker/aptraining-demo/objective",
    }

    secrets = rendered.find_resources("AWS::SecretsManager::Secret")
    assert len(secrets) == 2
    for secret in secrets.values():
        assert "GenerateSecretString" in secret["Properties"]
        assert "SecretString" not in secret["Properties"]
        assert secret["DeletionPolicy"] == "Retain"

    assert len(rendered.find_resources("AWS::EC2::VPC")) == 1
    assert len(rendered.find_resources("AWS::EC2::NatGateway")) == 0
    assert len(rendered.find_resources("AWS::EC2::Subnet")) == 4


def test_nat_gateway_is_enabled_only_with_explicit_budget_approval() -> None:
    rendered = template({"enable_nat_gateway": "true"})
    assert len(rendered.find_resources("AWS::EC2::NatGateway")) == 1


def test_list_outputs_are_comma_delimited_and_custom_id_is_exported() -> None:
    rendered = template(deployment_id="demo-2").to_json()
    expected = {f"aptraining-demo-2-{suffix}" for suffix in OUTPUT_SUFFIXES}
    values_by_export = {
        output["Export"]["Name"]: output["Value"]
        for output in rendered["Outputs"].values()
    }
    assert set(values_by_export) == expected

    for suffix in (
        "AvailabilityZones",
        "PrivateSubnetIds",
        "PrivateSubnetRouteTableIds",
        "PublicSubnetIds",
        "PublicSubnetRouteTableIds",
    ):
        value = values_by_export[f"aptraining-demo-2-{suffix}"]
        if isinstance(value, dict) and "Fn::Join" in value:
            delimiter, values = value["Fn::Join"]
            assert delimiter == ","
            assert len(values) == 2
        else:
            assert isinstance(value, str) and "," in value


def test_bootstrap_has_no_runtime_or_custom_domain_resources() -> None:
    rendered = template().to_json()
    resource_types = {resource["Type"] for resource in rendered["Resources"].values()}
    forbidden = {
        "AWS::ACM::Certificate",
        "AWS::Route53::HostedZone",
        "AWS::Route53::RecordSet",
        "AWS::ApiGateway::RestApi",
        "AWS::ApiGatewayV2::Api",
        "AWS::ECS::Service",
        "AWS::ECS::TaskDefinition",
    }
    assert resource_types.isdisjoint(forbidden)
    assert "sha256:" not in json.dumps(rendered)


@pytest.mark.parametrize("deployment_id", ["", "Bad_ID", "-demo", "demo-"])
def test_rejects_unsafe_deployment_ids(deployment_id: str) -> None:
    app = App(context={})
    with pytest.raises(ValueError, match="deployment_id"):
        PostTrainingBootstrapStack(
            app,
            "AutonomousPostTrainingBootstrap",
            env=Environment(account="123456789012", region="us-east-1"),
            deployment_id=deployment_id,
        )
