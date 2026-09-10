"""Contract tests for the synthesizable post-training AWS boundary."""

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

from stacks.post_training_stack import PostTrainingStack


def template(extra_context: dict[str, str] | None = None) -> Template:
    context = {
        "account": "123456789012",
        "region": "us-east-1",
        "backend_image_digest": "sha256:" + "0" * 64,
        "trainer_image_digest": "sha256:" + "1" * 64,
        "evaluator_image_digest": "sha256:" + "2" * 64,
        "objective_worker_url": "https://objective.example.test",
    }
    context.update(extra_context or {})
    app = App(
        context=context
    )
    stack = PostTrainingStack(
        app,
        "AutonomousPostTrainingStack",
        env=Environment(account="123456789012", region="us-east-1"),
    )
    return Template.from_stack(stack)


def test_has_three_immutable_ecr_repositories() -> None:
    template().resource_count_is("AWS::ECR::Repository", 3)


def test_ecr_repositories_are_immutable_and_scan_on_push() -> None:
    resources = template().find_resources("AWS::ECR::Repository")
    assert len(resources) == 3
    for resource in resources.values():
        assert resource["Properties"]["ImageTagMutability"] == "IMMUTABLE"
        assert resource["Properties"]["ImageScanningConfiguration"] == {
            "ScanOnPush": True
        }


def test_artifacts_are_encrypted_versioned_and_private() -> None:
    bucket = template().find_resources("AWS::S3::Bucket")
    assert len(bucket) == 1
    properties = next(iter(bucket.values()))["Properties"]
    assert properties["VersioningConfiguration"] == {"Status": "Enabled"}
    assert properties["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"
    ]["SSEAlgorithm"] == "aws:kms"
    assert properties["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }


def test_state_table_matches_recovery_scan_without_unused_gsi() -> None:
    table = template().find_resources("AWS::DynamoDB::Table")
    assert len(table) == 1
    properties = next(iter(table.values()))["Properties"]
    assert properties["KeySchema"] == [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ]
    assert {item["AttributeName"] for item in properties["AttributeDefinitions"]} == {
        "pk",
        "sk",
    }
    assert "GlobalSecondaryIndexes" not in properties


def test_secrets_are_created_and_injected_without_plaintext_values() -> None:
    secrets = template().find_resources("AWS::SecretsManager::Secret")
    assert len(secrets) == 2
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    serialized = json.dumps(task_definitions)
    assert "LIVE_APPROVAL_SECRET" in serialized
    assert "OBJECTIVE_WORKER_AUTH_TOKEN" in serialized
    assert "secret-value" not in serialized


def test_objective_service_is_internal_and_token_authenticated() -> None:
    stack_template = template(
        {
            "objective_worker_url": "",
            "objective_certificate_arn": (
                "arn:aws:acm:us-east-1:123456789012:certificate/"
                "abcdef01-2345-6789-abcd-ef0123456789"
            ),
            "objective_private_dns_name": "objective.internal.example.test",
            "objective_private_hosted_zone_name": "internal.example.test",
            "objective_certificate_san": "objective.internal.example.test",
        }
    )
    stack_template.resource_count_is("AWS::ECS::Service", 2)
    load_balancers = stack_template.find_resources("AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert len(load_balancers) == 1
    assert next(iter(load_balancers.values()))["Properties"]["Scheme"] == "internal"
    task_definitions = stack_template.find_resources("AWS::ECS::TaskDefinition")
    objective = [
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Objective"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    ]
    assert len(objective) == 1
    assert "OBJECTIVE_AUTH_TOKEN" in json.dumps(objective[0])
    ingress = stack_template.find_resources("AWS::EC2::SecurityGroupIngress")
    assert ingress
    assert all("CidrIp" not in item["Properties"] for item in ingress.values())
    assert any(
        item["Properties"].get("SourceSecurityGroupId")
        for item in ingress.values()
    )
    listeners = stack_template.find_resources("AWS::ElasticLoadBalancingV2::Listener")
    assert len(listeners) == 1
    assert next(iter(listeners.values()))["Properties"]["Port"] == 443


def test_external_objective_url_does_not_create_internal_service() -> None:
    stack_template = template()
    stack_template.resource_count_is("AWS::ECS::Service", 1)
    assert not stack_template.find_resources("AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert not stack_template.find_resources("AWS::ElasticLoadBalancingV2::Listener")
    assert len(stack_template.find_resources("AWS::ECS::TaskDefinition")) == 1


def test_runtime_task_injects_all_live_readiness_configuration() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    coordinator = [
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Backend"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    ]
    assert len(coordinator) == 1
    container = next(
        item
        for item in coordinator[0]["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Backend"
    )
    env_names = {item["Name"] for item in container["Environment"]}
    required = {
        "APP_MODE",
        "SERVICE_ROLE",
        "AWS_REGION",
        "TARGET_MODEL",
        "STRANDS_MODEL",
        "OBJECTIVE_SUITE",
        "OBJECTIVE_SUITE_VERSION",
        "S3_ARTIFACT_BUCKET",
        "S3_ARTIFACT_PREFIX",
        "DYNAMODB_TABLE_NAME",
        "SAGEMAKER_TRAINING_ROLE_ARN",
        "SAGEMAKER_TRAINING_IMAGE_URI",
        "SAGEMAKER_EVALUATION_IMAGE_URI",
        "OBJECTIVE_WORKER_URL",
        "HF_REPO_ID",
        "HF_REVISION",
        "TRAINING_INPUT_S3_URI",
        "EVALUATION_INPUT_S3_URI",
        "CHECKPOINT_S3_URI",
        "CHECKPOINT_SHA256",
        "SAGEMAKER_INSTANCE_TYPE",
        "GPU_INSTANCE_ALLOWLIST",
        "SAGEMAKER_GPU_QUOTA_CODE",
        "MINIMUM_GPU_QUOTA",
        "SAGEMAKER_INSTANCE_COUNT",
        "SAGEMAKER_VOLUME_SIZE_GB",
        "LIVE_APPROVAL_SECRET_ENV",
        "LIVE_APPROVAL_TTL_SECONDS",
        "MAX_EXPERIMENTS",
        "MAX_COST_USD",
        "MAX_TRAINING_TIME_MIN",
    }
    assert required <= env_names


def test_outputs_expose_resources_and_objective_endpoint() -> None:
    outputs = template().find_outputs("*")
    for name in (
        "ArtifactBucketName",
        "StateTableName",
        "BackendRepositoryUri",
        "TrainerRepositoryUri",
        "EvaluatorRepositoryUri",
        "ObjectiveWorkerUrl",
        "SageMakerTrainingRoleArn",
    ):
        assert name in outputs


def test_sagemaker_permissions_are_tag_bound_and_prefix_scoped() -> None:
    policies = template().find_resources("AWS::IAM::Policy")
    policy_documents = [item["Properties"]["PolicyDocument"] for item in policies.values()]
    sagemaker_statements = [
        statement
        for document in policy_documents
        for statement in document["Statement"]
        if any(
            str(action).startswith("sagemaker:")
            for action in (
                statement.get("Action", [])
                if isinstance(statement.get("Action", []), list)
                else [statement.get("Action")]
            )
        )
    ]
    assert any(
        statement.get("Condition", {}).get("StringEquals", {}).get("aws:RequestTag/project")
        == "autonomous-post-training"
        for statement in sagemaker_statements
    )
    assert any(
        "/post-training/*" in json.dumps(statement.get("Resource"))
        for document in policy_documents
        for statement in document["Statement"]
    )


def test_context_can_pin_sagemaker_worker_images_by_digest() -> None:
    rendered = json.dumps(
        template(
            {
                "trainer_image_digest": "sha256:" + "a" * 64,
                "evaluator_image_digest": "sha256:" + "b" * 64,
            }
        ).find_resources("AWS::ECS::TaskDefinition")
    )
    assert "sha256:" + "a" * 64 in rendered
    assert "sha256:" + "b" * 64 in rendered


def test_cloudwatch_metric_filters_are_declared() -> None:
    metric_filters = template().find_resources("AWS::Logs::MetricFilter")
    assert len(metric_filters) == 3
    assert {
        item["Properties"]["MetricTransformations"][0]["MetricName"]
        for item in metric_filters.values()
    } == {"PhaseStarts", "ProviderFailures", "RunFailures"}


def test_stack_fails_closed_without_image_digests() -> None:
    with pytest.raises(ValueError, match="backend_image_digest"):
        template({"backend_image_digest": "latest"})


def test_cdk_defaults_are_explicitly_development_only() -> None:
    config = json.loads((ROOT / "cdk.json").read_text())
    context = config["context"]
    assert context["config_note"].startswith("development-only;")
    assert not any(name.endswith("_image_digest") for name in context)
    assert "objective_worker_url" not in context
    assert "objective_certificate_arn" not in context


def test_stack_fails_closed_without_https_objective_endpoint() -> None:
    with pytest.raises(ValueError, match="objective_worker_url"):
        template({"objective_worker_url": "http://objective.internal"})


def test_valid_certificate_can_supply_https_objective_endpoint() -> None:
    stack_template = template(
        {
            "objective_worker_url": "",
            "objective_certificate_arn": (
                "arn:aws:acm:us-east-1:123456789012:certificate/"
                "abcdef01-2345-6789-abcd-ef0123456789"
            ),
            "objective_private_dns_name": "objective.internal.example.test",
            "objective_private_hosted_zone_name": "internal.example.test",
            "objective_certificate_san": "objective.internal.example.test",
        }
    )
    listener = next(
        iter(stack_template.find_resources("AWS::ElasticLoadBalancingV2::Listener").values())
    )
    assert listener["Properties"]["Protocol"] == "HTTPS"
    assert listener["Properties"]["Port"] == 443
    serialized = json.dumps(stack_template.to_json())
    assert "https://" in serialized
    assert '"DNSName"' in serialized


def test_certificate_only_objective_uses_private_alias_matching_certificate_san() -> None:
    stack_template = template(
        {
            "objective_worker_url": "",
            "objective_certificate_arn": (
                "arn:aws:acm:us-east-1:123456789012:certificate/"
                "abcdef01-2345-6789-abcd-ef0123456789"
            ),
            "objective_private_dns_name": "objective.internal.example.test",
            "objective_private_hosted_zone_name": "internal.example.test",
            "objective_certificate_san": "objective.internal.example.test",
        }
    )
    zones = stack_template.find_resources("AWS::Route53::HostedZone")
    assert len(zones) == 1
    assert next(iter(zones.values()))["Properties"]["Name"] == "internal.example.test."
    records = stack_template.find_resources("AWS::Route53::RecordSet")
    assert len(records) == 1
    record = next(iter(records.values()))["Properties"]
    assert record["Name"] == "objective.internal.example.test."
    assert record["Type"] == "A"
    coordinator = next(
        value
        for value in stack_template.find_resources("AWS::ECS::TaskDefinition").values()
        if any(
            container.get("Name") == "Backend"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    environment = {
        item["Name"]: item["Value"] for item in coordinator["Properties"]["ContainerDefinitions"][0]["Environment"]
    }
    assert environment["OBJECTIVE_WORKER_URL"] == "https://objective.internal.example.test"


def test_certificate_only_objective_fails_closed_on_missing_or_mismatched_dns_contract() -> None:
    certificate = (
        "arn:aws:acm:us-east-1:123456789012:certificate/"
        "abcdef01-2345-6789-abcd-ef0123456789"
    )
    with pytest.raises(ValueError, match="objective_private_dns_name"):
        template({"objective_worker_url": "", "objective_certificate_arn": certificate})
    with pytest.raises(ValueError, match="must match objective_certificate_san"):
        template(
            {
                "objective_worker_url": "",
                "objective_certificate_arn": certificate,
                "objective_private_dns_name": "objective.internal.example.test",
                "objective_private_hosted_zone_name": "internal.example.test",
                "objective_certificate_san": "other.internal.example.test",
            }
        )


def test_objective_role_can_read_write_versioned_encrypted_artifacts() -> None:
    stack_template = template(
        {
            "objective_worker_url": "",
            "objective_certificate_arn": (
                "arn:aws:acm:us-east-1:123456789012:certificate/"
                "abcdef01-2345-6789-abcd-ef0123456789"
            ),
            "objective_private_dns_name": "objective.internal.example.test",
            "objective_private_hosted_zone_name": "internal.example.test",
            "objective_certificate_san": "objective.internal.example.test",
        }
    )
    policies = stack_template.find_resources("AWS::IAM::Policy")
    objective_policy = next(
        item
        for key, item in policies.items()
        if key.startswith("ObjectiveTaskRoleDefaultPolicy")
    )
    statements = objective_policy["Properties"]["PolicyDocument"]["Statement"]
    actions = {
        action
        for statement in statements
        for action in (
            statement.get("Action", [])
            if isinstance(statement.get("Action", []), list)
            else [statement.get("Action")]
        )
    }
    assert {
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObject",
        "s3:ListBucketVersions",
        "s3:GetBucketVersioning",
        "kms:Encrypt",
        "kms:Decrypt",
        "kms:GenerateDataKey*",
    } <= actions
    object_statements = [
        statement
        for statement in statements
        if "/post-training/*" in json.dumps(statement.get("Resource"))
    ]
    assert object_statements
    assert all("s3:DeleteObject" not in json.dumps(statement) for statement in statements)


def test_task_role_has_only_scoped_readonly_preflight_permissions() -> None:
    policies = template().find_resources("AWS::IAM::Policy")
    coordinator_policy = next(
        item
        for key, item in policies.items()
        if key.startswith("TaskRoleDefaultPolicy")
    )
    statements = coordinator_policy["Properties"]["PolicyDocument"]["Statement"]
    by_sid = {statement["Sid"]: statement for statement in statements if "Sid" in statement}
    assert set(by_sid["ReadOnlyPreflightS3"]["Action"]) == {
        "s3:GetBucketLocation",
        "s3:GetEncryptionConfiguration",
        "s3:GetBucketVersioning",
    }
    assert by_sid["ReadOnlyPreflightIam"]["Action"] == "iam:GetRole"
    assert by_sid["ReadOnlyPreflightEcr"]["Action"] == "ecr:DescribeImages"
    assert by_sid["ReadOnlyPreflightServiceQuota"]["Action"] == "servicequotas:GetServiceQuota"
    assert by_sid["ReadOnlyPreflightServiceQuota"]["Resource"] == "*"
    assert by_sid["ReadOnlyPreflightS3"]["Resource"]["Fn::GetAtt"][1] == "Arn"
    assert by_sid["ReadOnlyPreflightIam"]["Resource"]["Fn::GetAtt"][1] == "Arn"
    ecr_resources = by_sid["ReadOnlyPreflightEcr"]["Resource"]
    assert len(ecr_resources) == 2
    assert all(resource["Fn::GetAtt"][1] == "Arn" for resource in ecr_resources)


def test_configured_s3_inputs_must_match_named_bucket_and_prefix() -> None:
    with pytest.raises(ValueError, match="training_input_s3_uri"):
        template(
            {
                "artifact_bucket_name": "autonomous-post-training-test",
                "training_input_s3_uri": "s3://other-bucket/post-training/input",
            }
        )


def test_task_roles_do_not_read_runtime_secrets() -> None:
    policies = template(
        {
            "objective_worker_url": "",
            "objective_certificate_arn": (
                "arn:aws:acm:us-east-1:123456789012:certificate/"
                "abcdef01-2345-6789-abcd-ef0123456789"
            ),
            "objective_private_dns_name": "objective.internal.example.test",
            "objective_private_hosted_zone_name": "internal.example.test",
            "objective_certificate_san": "objective.internal.example.test",
        }
    ).find_resources("AWS::IAM::Policy")
    coordinator_policy = next(
        item
        for key, item in policies.items()
        if key.startswith("TaskRoleDefaultPolicy")
    )
    objective_policy = next(
        item
        for key, item in policies.items()
        if key.startswith("ObjectiveTaskRoleDefaultPolicy")
    )
    for policy in (coordinator_policy, objective_policy):
        assert "secretsmanager:GetSecretValue" not in json.dumps(policy)
