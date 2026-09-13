"""Long-lived AWS prerequisites for the autonomous post-training runtime.

This stack is independent of deployable runtime images and model artifacts. It
can be synthesized before live model configuration exists and creates no ECS
services or task definitions. Private subnets are isolated by default to avoid
an always-on NAT bill; a consuming runtime must either supply VPC endpoints or
enable NAT explicitly with ``enable_nat_gateway=true`` in CDK context.
"""

from __future__ import annotations

import re
from typing import Any

from aws_cdk import CfnOutput, Fn, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct


class PostTrainingBootstrapStack(Stack):
    """Provision durable storage, image repositories, IAM, logs, and network."""

    deployment_id_pattern = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        deployment_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if deployment_id is None:
            configured_id = self.node.try_get_context("deployment_id")
            deployment_id = "demo" if configured_id is None else str(configured_id)
        if (
            len(deployment_id) > 32
            or not self.deployment_id_pattern.fullmatch(deployment_id)
        ):
            raise ValueError(
                "deployment_id must be 1-32 lowercase letters or digits, "
                "with single hyphens between groups"
            )
        self.deployment_id = deployment_id
        self.export_prefix = f"aptraining-{deployment_id}"

        self.artifact_key = kms.Key(
            self,
            "ArtifactKey",
            alias=f"alias/{self.export_prefix}-artifacts",
            description=f"Artifact encryption key for {self.export_prefix}",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.artifact_bucket = s3.Bucket(
            self,
            "ArtifactBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.artifact_key,
            enforce_ssl=True,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.state_table = dynamodb.Table(
            self,
            "StateTable",
            table_name=f"{self.export_prefix}-state",
            partition_key=dynamodb.Attribute(
                name="pk", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.artifact_key,
            point_in_time_recovery_specification=(
                dynamodb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True
                )
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.repositories = {
            name: ecr.Repository(
                self,
                f"{name.title()}Repository",
                repository_name=f"{self.export_prefix}-{name}",
                image_tag_mutability=ecr.TagMutability.IMMUTABLE,
                image_scan_on_push=True,
                removal_policy=RemovalPolicy.RETAIN,
            )
            for name in ("backend", "trainer", "evaluator")
        }

        self.runtime_log_group = logs.LogGroup(
            self,
            "RuntimeLogGroup",
            log_group_name=f"/aws/sagemaker/{self.export_prefix}/runtime",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.objective_log_group = logs.LogGroup(
            self,
            "ObjectiveLogGroup",
            log_group_name=f"/aws/sagemaker/{self.export_prefix}/objective",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"{self.export_prefix}-sagemaker-execution",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description=f"Scoped SageMaker execution role for {self.export_prefix}",
        )
        self._add_sagemaker_permissions()

        enable_nat_gateway = self._nat_gateway_enabled()
        self.vpc = ec2.Vpc(
            self,
            "RuntimeVpc",
            max_azs=2,
            # The managed NAT Gateway is intentionally opt-in: its hourly
            # charge alone exceeds the demo's $25 budget when left running.
            nat_gateways=1 if enable_nat_gateway else 0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=(
                        ec2.SubnetType.PRIVATE_WITH_EGRESS
                        if enable_nat_gateway
                        else ec2.SubnetType.PRIVATE_ISOLATED
                    ),
                    cidr_mask=24,
                ),
            ],
        )

        self.approval_secret = secretsmanager.Secret(
            self,
            "ApprovalSecret",
            secret_name=f"{self.export_prefix}/approval",
            description="Generated approval credential for controlled live execution",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=48,
                exclude_punctuation=True,
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.objective_credential_secret = secretsmanager.Secret(
            self,
            "ObjectiveCredentialSecret",
            secret_name=f"{self.export_prefix}/objective-credential",
            description="Generated bearer credential for the objective worker",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=48,
                exclude_punctuation=True,
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        self._add_outputs()

    def _nat_gateway_enabled(self) -> bool:
        """Enable managed NAT egress only when explicitly requested."""

        raw_value = self.node.try_get_context("enable_nat_gateway")
        if raw_value is None:
            return False
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError("enable_nat_gateway must be true or false")

    def _add_sagemaker_permissions(self) -> None:
        artifact_object_arn = self.artifact_bucket.arn_for_objects("post-training/*")
        log_stream_arns = [
            f"{group.log_group_arn}:*"
            for group in (self.runtime_log_group, self.objective_log_group)
        ]
        repository_arns = [
            repository.repository_arn for repository in self.repositories.values()
        ]

        policy = iam.Policy(
            self,
            "SageMakerExecutionPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[self.artifact_bucket.bucket_arn],
                    conditions={
                        "StringLike": {"s3:prefix": ["post-training", "post-training/*"]}
                    },
                ),
                iam.PolicyStatement(
                    actions=["s3:GetBucketLocation"],
                    resources=[self.artifact_bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:AbortMultipartUpload",
                        "s3:GetObject",
                        "s3:GetObjectVersion",
                        "s3:PutObject",
                    ],
                    resources=[artifact_object_arn],
                ),
                iam.PolicyStatement(
                    actions=[
                        "kms:Decrypt",
                        "kms:DescribeKey",
                        "kms:Encrypt",
                        "kms:GenerateDataKey",
                        "kms:ReEncryptFrom",
                        "kms:ReEncryptTo",
                    ],
                    resources=[self.artifact_key.key_arn],
                ),
                # ECR requires '*' for GetAuthorizationToken; image access below
                # is constrained to this deployment's three repositories.
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=[
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:DescribeImages",
                        "ecr:DescribeRepositories",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    resources=repository_arns,
                ),
                iam.PolicyStatement(
                    actions=[
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                    ],
                    resources=log_stream_arns,
                ),
            ],
        )
        policy.attach_to_role(self.sagemaker_role)

    def _add_outputs(self) -> None:
        # With NAT opt-in the VPC classifies these as PRIVATE_WITH_EGRESS;
        # otherwise they are PRIVATE_ISOLATED while retaining the same IDs and
        # route tables for the later runtime deployment contract.
        private_subnets = self.vpc.private_subnets or self.vpc.isolated_subnets
        public_subnets = self.vpc.public_subnets
        outputs = {
            "ArtifactBucketName": self.artifact_bucket.bucket_name,
            "ArtifactKeyArn": self.artifact_key.key_arn,
            "StateTableName": self.state_table.table_name,
            "SageMakerRoleArn": self.sagemaker_role.role_arn,
            "BackendRepositoryName": self.repositories["backend"].repository_name,
            "TrainerRepositoryName": self.repositories["trainer"].repository_name,
            "EvaluatorRepositoryName": self.repositories["evaluator"].repository_name,
            "RuntimeLogGroupName": self.runtime_log_group.log_group_name,
            "ObjectiveLogGroupName": self.objective_log_group.log_group_name,
            "VpcId": self.vpc.vpc_id,
            "AvailabilityZones": Fn.join(",", self.vpc.availability_zones),
            "PrivateSubnetIds": Fn.join(",", [subnet.subnet_id for subnet in private_subnets]),
            "PrivateSubnetRouteTableIds": Fn.join(
                ",", [subnet.route_table.route_table_id for subnet in private_subnets]
            ),
            "PublicSubnetIds": Fn.join(",", [subnet.subnet_id for subnet in public_subnets]),
            "PublicSubnetRouteTableIds": Fn.join(
                ",", [subnet.route_table.route_table_id for subnet in public_subnets]
            ),
            "ApprovalSecretArn": self.approval_secret.secret_arn,
            "ObjectiveCredentialSecretArn": self.objective_credential_secret.secret_arn,
        }
        for suffix, value in outputs.items():
            CfnOutput(
                self,
                f"Export{suffix}",
                value=value,
                export_name=f"{self.export_prefix}-{suffix}",
            )
