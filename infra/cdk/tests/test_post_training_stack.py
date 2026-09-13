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

from stacks.post_training_stack import PostTrainingStack  # noqa: E402

OBJECTIVE_MODEL_CONTEXT = {
    "artifact_bucket_name": "post-training-test-artifacts",
    "checkpoint_s3_uri": (
        "s3://post-training-test-artifacts/post-training/checkpoints/base.tar.gz"
        "?versionId=base-v1"
    ),
    "checkpoint_sha256": "a" * 64,
    "hf_revision": "b" * 40,
}


def template(extra_context: dict[str, str] | None = None) -> Template:
    context = {
        "account": "123456789012",
        "region": "us-east-1",
        "backend_image_digest": "sha256:" + "0" * 64,
        "trainer_image_digest": "sha256:" + "1" * 64,
        "evaluator_image_digest": "sha256:" + "2" * 64,
        "objective_worker_url": "https://objective.example.test",
        "coordinator_ingress_cidrs": "198.51.100.14/32",
        "coordinator_certificate_arn": (
            "arn:aws:acm:us-east-1:123456789012:certificate/"
            "12345678-abcd-1234-abcd-1234567890ab"
        ),
        "coordinator_public_dns_name": "api.example.test",
        "coordinator_public_hosted_zone_name": "example.test",
        "coordinator_public_hosted_zone_id": "Z1234567890ABC",
        "coordinator_certificate_san": "api.example.test",
        # A version-pinned checkpoint is part of the required live deployment
        # contract even when the objective worker is external.
        "artifact_bucket_name": "post-training-test-artifacts",
        "checkpoint_s3_uri": (
            "s3://post-training-test-artifacts/post-training/checkpoints/base.tar.gz"
            "?versionId=base-v1"
        ),
        "checkpoint_sha256": "a" * 64,
        "hf_revision": "b" * 40,
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


def test_zero_desired_count_synthesizes_repositories_without_starting_tasks() -> None:
    stack_template = template({"desired_count": "0"})

    stack_template.resource_count_is("AWS::ECR::Repository", 3)
    services = stack_template.find_resources("AWS::ECS::Service")
    assert len(services) == 1
    service = next(iter(services.values()))
    assert service["Properties"]["DesiredCount"] == 0


def test_default_desired_count_keeps_one_coordinator_task() -> None:
    services = template().find_resources("AWS::ECS::Service")
    assert len(services) == 1
    service = next(iter(services.values()))
    assert service["Properties"]["DesiredCount"] == 1


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
            **OBJECTIVE_MODEL_CONTEXT,
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
    assert {item["Properties"]["Scheme"] for item in load_balancers.values()} == {
        "internal",
        "internet-facing",
    }
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
    objective_container = next(
        item
        for item in objective[0]["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Objective"
    )
    objective_environment = {
        item["Name"]: item["Value"]
        for item in objective_container["Environment"]
    }
    assert objective_environment["SERVICE_ROLE"] == "objective"
    assert objective_environment["OBJECTIVE_MODEL_CHECKPOINT_DIR"] == "/opt/models/functiongemma"
    assert objective_environment["OBJECTIVE_MODEL_REVISION"] == "b" * 40
    assert objective_environment["OBJECTIVE_BASE_MODEL_URI"] == (
        "s3://post-training-test-artifacts/post-training/checkpoints/base.tar.gz"
        "?versionId=base-v1"
    )
    assert objective_environment["OBJECTIVE_BASE_MODEL_SHA256"] == "a" * 64
    assert objective_environment["S3_ARTIFACT_BUCKET"]
    assert objective_environment["S3_ARTIFACT_PREFIX"] == "post-training"
    ingress = stack_template.find_resources("AWS::EC2::SecurityGroupIngress")
    assert ingress
    assert all(
        item["Properties"]["FromPort"] == 443
        and item["Properties"]["ToPort"] == 443
        for item in ingress.values()
        if "CidrIp" in item["Properties"]
    )
    assert any(
        item["Properties"].get("SourceSecurityGroupId")
        for item in ingress.values()
    )
    listeners = stack_template.find_resources("AWS::ElasticLoadBalancingV2::Listener")
    assert {item["Properties"]["Port"] for item in listeners.values()} == {443}


def test_internal_objective_worker_uses_hackathon_task_sizing() -> None:
    stack_template = template(
        {
            **OBJECTIVE_MODEL_CONTEXT,
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
    task_definitions = stack_template.find_resources("AWS::ECS::TaskDefinition")
    objective = [
        task
        for task in task_definitions.values()
        if any(
            container.get("Name") == "Objective"
            for container in task["Properties"]["ContainerDefinitions"]
        )
    ]
    coordinator = [
        task
        for task in task_definitions.values()
        if any(
            container.get("Name") == "Backend"
            for container in task["Properties"]["ContainerDefinitions"]
        )
    ]

    assert len(objective) == 1
    assert objective[0]["Properties"]["Cpu"] == "2048"
    assert objective[0]["Properties"]["Memory"] == "4096"
    assert len(coordinator) == 1
    assert coordinator[0]["Properties"]["Cpu"] == "1024"
    assert coordinator[0]["Properties"]["Memory"] == "2048"


def test_external_objective_url_does_not_create_internal_service() -> None:
    stack_template = template()
    stack_template.resource_count_is("AWS::ECS::Service", 1)
    load_balancers = stack_template.find_resources("AWS::ElasticLoadBalancingV2::LoadBalancer")
    assert len(load_balancers) == 1
    coordinator_lb = next(iter(load_balancers.values()))
    assert coordinator_lb["Properties"]["Scheme"] == "internet-facing"
    listeners = stack_template.find_resources("AWS::ElasticLoadBalancingV2::Listener")
    assert len(listeners) == 1
    listener = next(iter(listeners.values()))["Properties"]
    assert listener["Port"] == 443
    assert listener["Protocol"] == "HTTPS"
    assert len(listener["Certificates"]) == 1
    records = stack_template.find_resources("AWS::Route53::RecordSet")
    assert len(records) == 1
    record = next(iter(records.values()))["Properties"]
    assert record["Name"] == "api.example.test."
    assert record["Type"] == "A"
    assert len(stack_template.find_resources("AWS::ECS::TaskDefinition")) == 1


def test_coordinator_ingress_is_public_but_restricted_to_explicit_cidr() -> None:
    security_groups = template().find_resources("AWS::EC2::SecurityGroup")
    coordinator_ingress = [
        rule
        for item in security_groups.values()
        for rule in item["Properties"].get("SecurityGroupIngress", [])
        if rule.get("CidrIp") == "198.51.100.14/32"
    ]
    assert len(coordinator_ingress) == 1
    assert coordinator_ingress[0]["FromPort"] == 443
    assert coordinator_ingress[0]["ToPort"] == 443
    assert coordinator_ingress[0]["IpProtocol"] == "tcp"
    listeners = template().find_resources("AWS::ElasticLoadBalancingV2::Listener")
    assert len(listeners) == 1
    listener = next(iter(listeners.values()))["Properties"]
    assert listener["Port"] == 443
    assert listener["Protocol"] == "HTTPS"


@pytest.mark.parametrize(
    ("cidr", "message"),
    [
        ("", "coordinator_ingress_cidrs"),
        ("0.0.0.0/0", "prefix /16"),
        ("198.51.100.1", "valid IPv4 CIDRs"),
        ("2001:db8::/64", "valid IPv4 CIDRs"),
    ],
)
def test_coordinator_ingress_requires_a_narrow_explicit_ipv4_cidr(
    cidr: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        template({"coordinator_ingress_cidrs": cidr})


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
    runtime_environment = {
        item["Name"]: item["Value"] for item in container["Environment"]
    }
    assert runtime_environment["LIVE_APPROVAL_TTL_SECONDS"] == "86400"


def test_outputs_expose_resources_and_objective_endpoint() -> None:
    outputs = template().find_outputs("*")
    for name in (
        "ArtifactBucketName",
        "StateTableName",
        "BackendRepositoryUri",
        "TrainerRepositoryUri",
        "EvaluatorRepositoryUri",
        "ObjectiveWorkerUrl",
        "CoordinatorUrl",
        "SageMakerTrainingRoleArn",
    ):
        assert name in outputs
    coordinator_output = outputs["CoordinatorUrl"]["Value"]
    assert coordinator_output == "https://api.example.test"


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
    list_tags = [
        statement
        for statement in sagemaker_statements
        if statement.get("Action") == "sagemaker:ListTags"
    ]
    assert len(list_tags) == 1
    assert len(list_tags[0]["Resource"]) == 2
    assert "processing-job/*" in json.dumps(list_tags[0]["Resource"])
    assert "training-job/*" in json.dumps(list_tags[0]["Resource"])
    assert list_tags[0]["Condition"]["StringEquals"]["sagemaker:ResourceTag/project"] == (
        "autonomous-post-training"
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


def test_default_cdk_app_is_bootstrap_only_and_has_no_runtime_inputs() -> None:
    config = json.loads((ROOT / "cdk.json").read_text())
    context = config["context"]
    assert context["config_note"].startswith("bootstrap-only;")
    assert config["app"].endswith("app.py")
    assert not any(name.endswith("_image_digest") for name in context)
    assert "objective_worker_url" not in context
    assert "objective_certificate_arn" not in context
    assert not any("model" in name or "checkpoint" in name for name in context)
    assert not any(
        marker in name
        for name in context
        for marker in ("evaluation", "certificate", "dns", "ingress")
    )


def test_stack_fails_closed_without_https_objective_endpoint() -> None:
    with pytest.raises(ValueError, match="objective_worker_url"):
        template({"objective_worker_url": "http://objective.internal"})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"coordinator_certificate_arn": ""}, "coordinator_certificate_arn"),
        ({"coordinator_public_dns_name": ""}, "coordinator_public_dns_name"),
        (
            {"coordinator_public_hosted_zone_name": ""},
            "coordinator_public_hosted_zone_name",
        ),
        (
            {"coordinator_public_hosted_zone_id": ""},
            "coordinator_public_hosted_zone_id",
        ),
        ({"coordinator_certificate_san": "wrong.example.test"}, "must match"),
        (
            {
                "coordinator_certificate_arn": (
                    "arn:aws:acm:us-west-2:123456789012:certificate/"
                    "12345678-abcd-1234-abcd-1234567890ab"
                )
            },
            "stack's AWS region",
        ),
    ],
)
def test_coordinator_https_requires_valid_certificate_and_public_dns_contract(
    config: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        template(config)


def test_valid_certificate_can_supply_https_objective_endpoint() -> None:
    stack_template = template(
        {
            **OBJECTIVE_MODEL_CONTEXT,
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
            **OBJECTIVE_MODEL_CONTEXT,
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
    assert len(records) == 2
    record = next(
        value["Properties"]
        for value in records.values()
        if value["Properties"]["Name"] == "objective.internal.example.test."
    )
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
        item["Name"]: item["Value"]
        for item in coordinator["Properties"]["ContainerDefinitions"][0]["Environment"]
    }
    assert environment["OBJECTIVE_WORKER_URL"] == "https://objective.internal.example.test"


def test_certificate_only_objective_fails_closed_on_missing_or_mismatched_dns_contract() -> None:
    certificate = (
        "arn:aws:acm:us-east-1:123456789012:certificate/"
        "abcdef01-2345-6789-abcd-ef0123456789"
    )
    with pytest.raises(ValueError, match="objective_private_dns_name"):
        template(
            {
                **OBJECTIVE_MODEL_CONTEXT,
                "objective_worker_url": "",
                "objective_certificate_arn": certificate,
            }
        )
    with pytest.raises(ValueError, match="must match objective_certificate_san"):
        template(
            {
                **OBJECTIVE_MODEL_CONTEXT,
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
            **OBJECTIVE_MODEL_CONTEXT,
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
    artifact_prefix = "scoped-preflight"
    stack_template = template(
        {
            "artifact_prefix": artifact_prefix,
            "checkpoint_s3_uri": (
                "s3://post-training-test-artifacts/scoped-preflight/"
                "checkpoints/base.tar.gz?versionId=base-v1"
            ),
        }
    )
    policies = stack_template.find_resources("AWS::IAM::Policy")
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
    versioned_object_preflight = by_sid["ReadOnlyPreflightS3ObjectVersions"]
    assert versioned_object_preflight["Action"] == "s3:GetObjectVersion"
    artifact_bucket_id = next(iter(stack_template.find_resources("AWS::S3::Bucket")))
    assert versioned_object_preflight["Resource"] == {
        "Fn::Join": [
            "",
            [
                {"Fn::GetAtt": [artifact_bucket_id, "Arn"]},
                f"/{artifact_prefix}/*",
            ],
        ]
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


def test_live_checkpoint_uri_requires_explicit_immutable_version() -> None:
    with pytest.raises(ValueError, match="checkpoint_s3_uri.*versionId"):
        template({"checkpoint_s3_uri": ""})

    with pytest.raises(ValueError, match="exactly one immutable versionId"):
        template(
            {
                "checkpoint_s3_uri": (
                    "s3://post-training-test-artifacts/"
                    "post-training/checkpoints/base.tar.gz"
                )
            }
        )


def test_task_roles_do_not_read_runtime_secrets() -> None:
    policies = template(
        {
            **OBJECTIVE_MODEL_CONTEXT,
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
