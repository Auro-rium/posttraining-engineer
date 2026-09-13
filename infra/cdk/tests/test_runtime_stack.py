"""Synthesis contracts for the imported-resource AWS runtime stack."""

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

from stacks.runtime_stack import PostTrainingRuntimeStack


def context(extra: dict[str, str] | None = None) -> dict[str, str]:
    values = {
        "account": "123456789012",
        "region": "us-east-1",
        "backend_image_digest": "sha256:" + "0" * 64,
        "trainer_image_digest": "sha256:" + "1" * 64,
        "evaluator_image_digest": "sha256:" + "2" * 64,
        "git_sha": "d" * 40,
        "build_id": "aws-codebuild:build-123",
        "artifact_bucket_name": "bootstrap-generated-test-bucket",
        "checkpoint_s3_uri": (
            "s3://bootstrap-generated-test-bucket/post-training/checkpoint.tar"
            "?versionId=checkpoint-v1"
        ),
        "checkpoint_sha256": "a" * 64,
        "hf_revision": "b" * 40,
        "evaluation_input_s3_uri": (
            "s3://bootstrap-generated-test-bucket/post-training/inputs/evaluation/"
            + "c" * 64
        ),
        "evaluation_manifest_sha256": "c" * 64,
        "target_model": "google/functiongemma-270m-it",
        "objective_suite": "AgentGym/AgentEval",
        "objective_suite_version": "agent-eval-v1",
        "posttraining_seed": "7",
        "sagemaker_instance_type": "ml.g5.xlarge",
        "sagemaker_gpu_quota_code": "L-B6D80D9C",
        "sagemaker_processing_gpu_quota_code": "L-ABCDEF12",
        "gpu_instance_allowlist": "ml.g5.xlarge",
        "sagemaker_instance_count": "1",
        "sagemaker_volume_size_gb": "30",
        "minimum_gpu_quota": "1",
        "max_cost_usd": "25",
        "max_experiments": "5",
        "max_training_time_min": "120",
        "approval_ttl_seconds": "86400",
        "runtime_nat_gateway_acknowledged": "true",
    }
    values.update(extra or {})
    return values


def template(
    extra: dict[str, str] | None = None, *, region: str = "us-east-1"
) -> Template:
    app = App(context=context(extra))
    stack = PostTrainingRuntimeStack(
        app,
        "AutonomousPostTrainingRuntimeStack",
        env=Environment(account="123456789012", region=region),
    )
    return Template.from_stack(stack)


def test_api_gateway_default_synthesizes_private_http_api_fronted_runtime() -> None:
    stack_template = template()

    stack_template.resource_count_is("AWS::ApiGatewayV2::Api", 1)
    stack_template.resource_count_is("AWS::ApiGatewayV2::VpcLink", 1)
    stack_template.resource_count_is("AWS::ECS::Service", 2)
    stack_template.resource_count_is("AWS::EC2::NatGateway", 1)
    stack_template.resource_count_is("AWS::EC2::EIP", 1)
    assert not stack_template.find_resources("AWS::CertificateManager::Certificate")
    assert not stack_template.find_resources("AWS::Route53::RecordSet")
    assert not stack_template.find_resources("AWS::Route53::HostedZone")

    api = next(iter(stack_template.find_resources("AWS::ApiGatewayV2::Api").values()))
    assert api["Properties"]["ProtocolType"] == "HTTP"
    assert api["Properties"].get("DisableExecuteApiEndpoint", False) is False

    routes = stack_template.find_resources("AWS::ApiGatewayV2::Route")
    route_targets = {
        item["Properties"]["RouteKey"]: item["Properties"]["Target"]
        for item in routes.values()
    }
    assert set(route_targets) == {
        "ANY /api/{proxy+}",
        "ANY /health",
        "ANY /v1/{proxy+}",
    }
    assert route_targets["ANY /api/{proxy+}"] == route_targets["ANY /health"]
    assert route_targets["ANY /v1/{proxy+}"] != route_targets["ANY /health"]

    for service in stack_template.find_resources("AWS::ECS::Service").values():
        network = service["Properties"]["NetworkConfiguration"]["AwsvpcConfiguration"]
        assert network["AssignPublicIp"] == "DISABLED"

    load_balancers = stack_template.find_resources(
        "AWS::ElasticLoadBalancingV2::LoadBalancer"
    )
    assert len(load_balancers) == 2
    assert {item["Properties"]["Scheme"] for item in load_balancers.values()} == {
        "internal"
    }


def test_runtime_region_is_not_artificially_limited_by_egress_design() -> None:
    template(region="ap-south-1").resource_count_is("AWS::EC2::NatGateway", 1)


def test_coordinator_calls_objective_through_its_api_gateway_v1_path() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    coordinator = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Coordinator"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    container = next(
        item
        for item in coordinator["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Coordinator"
    )
    environment = {item["Name"]: item["Value"] for item in container["Environment"]}
    endpoint = environment["OBJECTIVE_WORKER_URL"]
    assert endpoint["Fn::Join"][0] == ""
    join_parts = endpoint["Fn::Join"][1]
    assert join_parts[0]["Fn::GetAtt"][1] == "ApiEndpoint"
    assert join_parts[1] == "/v1/"


def test_coordinator_receives_imported_bucket_and_sealed_manifest_provenance() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    coordinator = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Coordinator"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    container = next(
        item
        for item in coordinator["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Coordinator"
    )
    environment = {item["Name"]: item["Value"] for item in container["Environment"]}
    assert environment["S3_ARTIFACT_BUCKET"] == {
        "Fn::ImportValue": "aptraining-demo-ArtifactBucketName"
    }
    assert environment["LIVE_BENCHMARK_MANIFEST_SHA256"] == "c" * 64
    assert environment["EVALUATION_INPUT_S3_URI"].endswith("/" + "c" * 64)
    assert "?" not in environment["EVALUATION_INPUT_S3_URI"]


def test_coordinator_receives_immutable_build_provenance_and_single_baseline_episode() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    coordinator = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Coordinator"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    container = next(
        item
        for item in coordinator["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Coordinator"
    )
    environment = {item["Name"]: item["Value"] for item in container["Environment"]}

    assert environment["GIT_SHA"] == "d" * 40
    assert environment["BUILD_ID"] == "aws-codebuild:build-123"
    assert environment["IMAGE_DIGEST"] == "sha256:" + "0" * 64
    assert environment["BASELINE_EPISODES"] == "1"


def test_coordinator_receives_validated_sagemaker_gpu_quota_codes() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    coordinator = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Coordinator"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    container = next(
        item
        for item in coordinator["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Coordinator"
    )
    environment = {item["Name"]: item["Value"] for item in container["Environment"]}
    assert environment["SAGEMAKER_GPU_QUOTA_CODE"] == "L-B6D80D9C"
    assert environment["SAGEMAKER_PROCESSING_GPU_QUOTA_CODE"] == "L-ABCDEF12"


def test_coordinator_bedrock_access_is_scoped_to_its_foundation_model() -> None:
    policies = template().find_resources("AWS::IAM::Policy")
    coordinator_policy = next(
        item
        for logical_id, item in policies.items()
        if logical_id.startswith("CoordinatorTaskRoleDefaultPolicy")
    )
    statements = coordinator_policy["Properties"]["PolicyDocument"]["Statement"]
    bedrock_statement = next(
        statement
        for statement in statements
        if statement.get("Sid") == "InvokePinnedBedrockFoundationModel"
    )

    assert set(bedrock_statement["Action"]) == {
        "bedrock:GetFoundationModel",
        "bedrock:InvokeModel",
    }
    assert bedrock_statement["Resource"] == {
        "Fn::Join": [
            "",
            [
                "arn:",
                {"Ref": "AWS::Partition"},
                ":bedrock:us-east-1::foundation-model/nvidia.nemotron-super-3-120b",
            ],
        ]
    }
    assert bedrock_statement["Resource"] != "*"


def test_coordinator_readiness_access_is_readonly_and_scoped_to_runtime_inputs() -> (
    None
):
    policies = template().find_resources("AWS::IAM::Policy")
    coordinator_policy = next(
        item
        for logical_id, item in policies.items()
        if logical_id.startswith("CoordinatorTaskRoleDefaultPolicy")
    )
    statements = coordinator_policy["Properties"]["PolicyDocument"]["Statement"]
    by_sid = {
        statement["Sid"]: statement for statement in statements if "Sid" in statement
    }

    bucket_metadata = by_sid["ReadinessArtifactBucketMetadata"]
    assert set(bucket_metadata["Action"]) == {
        "s3:GetBucketLocation",
        "s3:GetEncryptionConfiguration",
        "s3:GetBucketVersioning",
    }
    assert "aptraining-demo-ArtifactBucketName" in json.dumps(
        bucket_metadata["Resource"]
    )
    assert bucket_metadata["Resource"] != "*"

    training_role = by_sid["ReadinessSageMakerRole"]
    assert training_role["Action"] == "iam:GetRole"
    assert training_role["Resource"] == {
        "Fn::ImportValue": "aptraining-demo-SageMakerRoleArn"
    }

    worker_images = by_sid["ReadinessWorkerImages"]
    assert worker_images["Action"] == "ecr:DescribeImages"
    assert len(worker_images["Resource"]) == 2
    worker_image_resources = json.dumps(worker_images["Resource"])
    assert "aptraining-demo-TrainerRepositoryName" in worker_image_resources
    assert "aptraining-demo-EvaluatorRepositoryName" in worker_image_resources
    assert worker_images["Resource"] != "*"

    service_quotas = by_sid["ReadinessServiceQuotas"]
    assert service_quotas["Action"] == "servicequotas:GetServiceQuota"
    assert {
        json.dumps(resource, sort_keys=True) for resource in service_quotas["Resource"]
    } == {
        json.dumps(
            {
                "Fn::Join": [
                    "",
                    [
                        "arn:",
                        {"Ref": "AWS::Partition"},
                        ":servicequotas:us-east-1:123456789012:sagemaker/L-B6D80D9C",
                    ],
                ]
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "Fn::Join": [
                    "",
                    [
                        "arn:",
                        {"Ref": "AWS::Partition"},
                        ":servicequotas:us-east-1:123456789012:sagemaker/L-ABCDEF12",
                    ],
                ]
            },
            sort_keys=True,
        ),
    }


def test_objective_worker_can_encrypt_new_verified_artifacts() -> None:
    policies = template().find_resources("AWS::IAM::Policy")
    objective_policy = next(
        item
        for logical_id, item in policies.items()
        if logical_id.startswith("ObjectiveTaskRoleDefaultPolicy")
    )
    statements = objective_policy["Properties"]["PolicyDocument"]["Statement"]
    kms_actions = {
        action
        for statement in statements
        for action in (
            statement.get("Action", [])
            if isinstance(statement.get("Action"), list)
            else [statement.get("Action")]
        )
        if isinstance(action, str) and action.startswith("kms:")
    }

    assert {"kms:Decrypt", "kms:Encrypt"} <= kms_actions
    assert {"kms:GenerateDataKey", "kms:GenerateDataKey*"} & kms_actions


def test_objective_bearer_credential_is_injected_from_imported_secret() -> None:
    task_definitions = template().find_resources("AWS::ECS::TaskDefinition")
    objective = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Objective"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    container = next(
        item
        for item in objective["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Objective"
    )
    secrets = {item["Name"]: item["ValueFrom"] for item in container["Secrets"]}
    assert "OBJECTIVE_AUTH_TOKEN" in secrets
    assert "Fn::ImportValue" in json.dumps(secrets["OBJECTIVE_AUTH_TOKEN"])

    coordinator = next(
        value
        for value in task_definitions.values()
        if any(
            container.get("Name") == "Coordinator"
            for container in value["Properties"]["ContainerDefinitions"]
        )
    )
    coordinator_container = next(
        item
        for item in coordinator["Properties"]["ContainerDefinitions"]
        if item["Name"] == "Coordinator"
    )
    coordinator_secrets = {
        item["Name"]: item["ValueFrom"] for item in coordinator_container["Secrets"]
    }
    assert "OBJECTIVE_WORKER_AUTH_TOKEN" in coordinator_secrets
    assert "LIVE_APPROVAL_SECRET" in coordinator_secrets
    assert all(
        "Fn::ImportValue" in json.dumps(value) for value in coordinator_secrets.values()
    )


def test_runtime_imports_bootstrap_exports_under_default_deployment_id() -> None:
    serialized = json.dumps(template().to_json())
    for suffix in (
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
    ):
        assert f"aptraining-demo-{suffix}" in serialized


def test_runtime_uses_nat_for_private_subnet_egress_and_gateway_endpoints_for_data() -> (
    None
):
    stack_template = template()
    endpoints = stack_template.find_resources("AWS::EC2::VPCEndpoint")
    assert len(endpoints) == 2
    assert {item["Properties"]["VpcEndpointType"] for item in endpoints.values()} == {
        "Gateway"
    }
    routes = stack_template.find_resources("AWS::EC2::Route")
    nat_routes = [
        item
        for item in routes.values()
        if item["Properties"].get("NatGatewayId")
        and item["Properties"].get("DestinationCidrBlock") == "0.0.0.0/0"
    ]
    assert len(nat_routes) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend_image_digest", "SHA256:" + "0" * 64, "backend_image_digest"),
        ("git_sha", "not-a-commit", "git_sha"),
        ("build_id", "has spaces", "build_id"),
        ("baseline_episodes", "0", "baseline_episodes"),
        ("baseline_episodes", "11", "baseline_episodes"),
        ("trainer_image_digest", "sha256:" + "A" * 64, "trainer_image_digest"),
        ("evaluator_image_digest", "latest", "evaluator_image_digest"),
        (
            "checkpoint_s3_uri",
            "s3://bootstrap-generated-test-bucket/post-training/checkpoint.tar",
            "versionId",
        ),
        (
            "checkpoint_s3_uri",
            "s3://bootstrap-generated-test-bucket/post-training/checkpoint.tar?versionId=a&versionId=b",
            "exactly one",
        ),
        ("checkpoint_sha256", "A" * 64, "checkpoint_sha256"),
        ("hf_revision", "b" * 39, "hf_revision"),
        (
            "evaluation_input_s3_uri",
            "s3://bootstrap-generated-test-bucket/post-training/inputs/evaluation/"
            + "c" * 64
            + "?versionId=sealed-v1",
            "without query",
        ),
        ("evaluation_manifest_sha256", "not-a-digest", "evaluation_manifest_sha256"),
        (
            "evaluation_input_s3_uri",
            "s3://bootstrap-generated-test-bucket/post-training/inputs/evaluation/mutable",
            "content-addressed",
        ),
        ("target_model", "google/functiongemma-latest", "target_model"),
        ("objective_suite", "", "objective_suite"),
        ("objective_suite_version", "", "objective_suite_version"),
        ("posttraining_seed", "-1", "posttraining_seed"),
        ("sagemaker_instance_type", "ml.p4d.24xlarge", "gpu_instance_allowlist"),
        ("sagemaker_gpu_quota_code", "not-a-quota-code", "sagemaker_gpu_quota_code"),
        (
            "sagemaker_processing_gpu_quota_code",
            "L-abc12345",
            "sagemaker_processing_gpu_quota_code",
        ),
        ("max_experiments", "6", "max_experiments"),
        ("max_cost_usd", "26", "max_cost_usd"),
    ],
)
def test_synthesis_rejects_mutable_or_out_of_bounds_runtime_inputs(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        template({field: value})


def test_synthesis_rejects_missing_required_immutable_artifact_contracts() -> None:
    with pytest.raises(ValueError, match="checkpoint_s3_uri"):
        template({"checkpoint_s3_uri": ""})

    with pytest.raises(ValueError, match="evaluation_input_s3_uri"):
        template({"evaluation_input_s3_uri": ""})

    with pytest.raises(ValueError, match="evaluation_manifest_sha256"):
        template({"evaluation_manifest_sha256": ""})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("artifact_bucket_name", "artifact_bucket_name"),
        ("backend_image_digest", "backend_image_digest"),
        ("trainer_image_digest", "trainer_image_digest"),
        ("evaluator_image_digest", "evaluator_image_digest"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("hf_revision", "hf_revision"),
        ("evaluation_manifest_sha256", "evaluation_manifest_sha256"),
        ("target_model", "target_model"),
        ("objective_suite", "objective_suite"),
        ("objective_suite_version", "objective_suite_version"),
        ("git_sha", "git_sha"),
        ("build_id", "build_id"),
        ("posttraining_seed", "posttraining_seed"),
        ("sagemaker_instance_type", "sagemaker_instance_type"),
        ("sagemaker_gpu_quota_code", "sagemaker_gpu_quota_code"),
        ("sagemaker_processing_gpu_quota_code", "sagemaker_processing_gpu_quota_code"),
        ("gpu_instance_allowlist", "gpu_instance_allowlist"),
        ("max_experiments", "max_experiments"),
        ("max_cost_usd", "max_cost_usd"),
        ("max_training_time_min", "max_training_time_min"),
        ("runtime_nat_gateway_acknowledged", "runtime_nat_gateway_acknowledged"),
    ],
)
def test_synthesis_rejects_missing_runtime_contract_values(
    field: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        template({field: ""})


def test_runtime_requires_an_explicit_acknowledgement_for_nat_gateway_cost() -> None:
    with pytest.raises(
        ValueError,
        match=r"runtime_nat_gateway_acknowledged.*0\.045 per gateway-hour.*Bootstrap remains NAT-free",
    ):
        template({"runtime_nat_gateway_acknowledged": "false"})


def test_runtime_rejects_artifact_bucket_or_prefix_that_disagrees_with_bootstrap_contract() -> (
    None
):
    with pytest.raises(ValueError, match="artifact_bucket_name"):
        template(
            {
                "checkpoint_s3_uri": (
                    "s3://unrelated-bucket/post-training/checkpoint.tar"
                    "?versionId=checkpoint-v1"
                )
            }
        )
    with pytest.raises(ValueError, match="post-training"):
        template({"artifact_prefix": "other-prefix"})

    with pytest.raises(ValueError, match="artifact_bucket_name"):
        template(
            {
                "evaluation_input_s3_uri": (
                    "s3://unrelated-bucket/post-training/inputs/evaluation/" + "c" * 64
                )
            }
        )


def test_non_api_gateway_ingress_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="direct_tls mode requires"):
        template({"ingress_mode": "direct_tls"})
