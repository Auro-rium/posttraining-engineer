"""Synthesizable AWS boundary for the autonomous post-training backend.

This stack creates the durable, authenticated boundaries used by the live
controller. It deliberately does not push images, stage checkpoints, submit
SageMaker jobs, or otherwise perform an execution-side effect.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from aws_cdk import Arn, CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct


class PostTrainingStack(Stack):
    """Provision storage, worker images, service boundaries, and IAM."""

    project_tag = "autonomous-post-training"
    metric_namespace = "AutonomousPostTraining"
    image_digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")

    def _context(self, name: str, default: Any = None) -> Any:
        value = self.node.try_get_context(name)
        return default if value is None else value

    def _text(self, name: str, default: str = "") -> str:
        value = self._context(name, default)
        return str(value) if value is not None else default

    def _required_digest(self, name: str) -> str:
        digest = self._text(name)
        if not self.image_digest_pattern.fullmatch(digest):
            raise ValueError(f"{name} must be a lowercase sha256 image digest")
        return digest

    def _image(self, repository: ecr.Repository, digest_key: str) -> ecs.ContainerImage:
        digest = self._required_digest(digest_key)
        return ecs.ContainerImage.from_registry(f"{repository.repository_uri}@{digest}")

    def _image_uri(self, repository: ecr.Repository, digest_key: str) -> str:
        digest = self._required_digest(digest_key)
        return f"{repository.repository_uri}@{digest}"

    def _configured_image_uri(
        self, name: str, repository: ecr.Repository, digest_key: str
    ) -> str:
        value = self._text(name)
        if not value:
            return self._image_uri(repository, digest_key)
        if not re.search(r"@sha256:[0-9a-f]{64}$", value):
            raise ValueError(f"{name} must be an ECR URI pinned by a sha256 digest")
        return value

    def _artifact_uri(
        self,
        name: str,
        *,
        bucket: s3.Bucket,
        prefix: str,
        suffix: str,
        require_version: bool = False,
    ) -> str:
        """Constrain configured input URIs to this bucket and artifact prefix."""

        configured = self._text(name)
        if not configured:
            return f"s3://{bucket.bucket_name}/{prefix}/{suffix}"
        parsed = urlparse(configured)
        expected_bucket = self._text("artifact_bucket_name")
        expected_path = "/" + prefix + "/"
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or (expected_bucket and parsed.netloc != expected_bucket)
            or (not expected_bucket)
            or not parsed.path.startswith(expected_path)
            or ".." in parsed.path.split("/")
        ):
            raise ValueError(
                f"{name} must target the configured artifact bucket and prefix"
            )
        if require_version and not parse_qs(parsed.query).get("versionId"):
            raise ValueError(f"{name} must include an immutable versionId")
        return configured

    def _require_https_url(self, value: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("objective_worker_url must be a credential-free HTTPS URL")

    def _require_certificate_arn(self, value: str) -> None:
        if not re.fullmatch(
            r"arn:(?:aws|aws-us-gov|aws-cn):acm:[a-z0-9-]+:\d{12}:certificate/[a-f0-9-]+",
            value,
        ):
            raise ValueError("objective_certificate_arn must be a valid ACM certificate ARN")

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = self._text("artifact_prefix", "post-training").strip("/") or "post-training"
        target_model = self._text("target_model", "google/functiongemma-270m-it")
        reasoning_model = self._text("strands_model", "nvidia.nemotron-super-3-120b")
        objective_suite = self._text("objective_suite", "AgentGym/AgentEval")
        objective_suite_version = self._text("objective_suite_version", "agent-eval-v1")
        project = self._text("project_tag", self.project_tag)
        Tags.of(self).add("Project", project)
        Tags.of(self).add("ManagedBy", "autonomous-post-training")

        artifact_key = kms.Key(
            self,
            "ArtifactKey",
            alias="alias/autonomous-post-training-artifacts",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        bucket_name = self._text("artifact_bucket_name") or None
        artifacts = s3.Bucket(
            self,
            "Artifacts",
            bucket_name=bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            bucket_key_enabled=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=artifact_key,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        state = dynamodb.Table(
            self,
            "RunState",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=artifact_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        state.add_global_secondary_index(
            index_name="LeaseDispatcherIndex",
            partition_key=dynamodb.Attribute(name="status", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(
                name="event_sequence", type=dynamodb.AttributeType.NUMBER
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        backend_repository = ecr.Repository(
            self,
            "BackendRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.RETAIN,
        )
        trainer_repository = ecr.Repository(
            self,
            "TrainerRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.RETAIN,
        )
        evaluator_repository = ecr.Repository(
            self,
            "EvaluatorRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.RETAIN,
        )

        approval_secret = secretsmanager.Secret(
            self,
            "ApprovalSecret",
            description="HMAC secret for one bounded autonomous-run approval",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=64,
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )
        objective_secret = secretsmanager.Secret(
            self,
            "ObjectiveCredential",
            description="Bearer credential for the internal objective worker",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=64,
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        vpc = ec2.Vpc(self, "RuntimeVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "RuntimeCluster", vpc=vpc, container_insights=True)
        runtime_logs = logs.LogGroup(
            self,
            "RuntimeLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )
        objective_logs = logs.LogGroup(
            self,
            "ObjectiveLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )

        training_role = iam.Role(
            self,
            "SageMakerTrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        training_role.add_to_policy(
            iam.PolicyStatement(
                sid="ArtifactPrefixReadWrite",
                actions=["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
                resources=[f"{artifacts.bucket_arn}/{prefix}/*"],
            )
        )
        training_role.add_to_policy(
            iam.PolicyStatement(
                sid="ArtifactPrefixList",
                actions=["s3:ListBucket"],
                resources=[artifacts.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}},
            )
        )
        training_role.add_to_policy(
            iam.PolicyStatement(
                sid="TrainingLogWrite",
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[runtime_logs.log_group_arn + ":*"],
            )
        )
        artifact_key.grant_encrypt_decrypt(training_role)
        trainer_repository.grant_pull(training_role)
        evaluator_repository.grant_pull(training_role)

        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ArtifactPrefixReadWrite",
                actions=["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
                resources=[f"{artifacts.bucket_arn}/{prefix}/*"],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ArtifactPrefixList",
                actions=["s3:ListBucket"],
                resources=[artifacts.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}},
            )
        )
        state.grant_read_write_data(task_role)
        artifact_key.grant_encrypt_decrypt(task_role)
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokePinnedNemotron",
                actions=["bedrock:InvokeModel"],
                resources=[
                    Arn.format(
                        {
                            "partition": self.partition,
                            "service": "bedrock",
                            "region": self.region,
                            "account": "",
                            "resource": "foundation-model/nvidia.nemotron-super-3-120b",
                        },
                        stack=self,
                    )
                ],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerTaggedJobs",
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:CreateTrainingJob",
                ],
                resources=["*"],
                conditions={"StringEquals": {"aws:RequestTag/project": project}},
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="DescribeTaggedSageMakerJobs",
                actions=[
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:StopTrainingJob",
                ],
                resources=[
                    f"arn:{self.partition}:sagemaker:{self.region}:{self.account}:processing-job/*",
                    f"arn:{self.partition}:sagemaker:{self.region}:{self.account}:training-job/*",
                ],
                conditions={"StringEquals": {"sagemaker:ResourceTag/project": project}},
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="PassSageMakerRole",
                actions=["iam:PassRole"],
                resources=[training_role.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            )
        )

        objective_role = iam.Role(
            self,
            "ObjectiveTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        objective_role.add_to_policy(
            iam.PolicyStatement(
                sid="ObjectiveReadPrefix",
                actions=["s3:GetObject"],
                resources=[f"{artifacts.bucket_arn}/{prefix}/*"],
            )
        )
        objective_role.add_to_policy(
            iam.PolicyStatement(
                sid="ObjectiveListPrefix",
                actions=["s3:ListBucket"],
                resources=[artifacts.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}},
            )
        )
        artifact_key.grant_decrypt(objective_role)

        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        backend_repository.grant_pull(execution_role)
        approval_secret.grant_read(execution_role)
        objective_secret.grant_read(execution_role)

        objective_execution_role = iam.Role(
            self,
            "ObjectiveExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        backend_repository.grant_pull(objective_execution_role)
        objective_secret.grant_read(objective_execution_role)

        objective_task_definition = ecs.FargateTaskDefinition(
            self,
            "ObjectiveTaskDefinition",
            cpu=512,
            memory_limit_mib=1024,
            task_role=objective_role,
            execution_role=objective_execution_role,
        )
        objective_task_definition.add_container(
            "Objective",
            image=self._image(backend_repository, "backend_image_digest"),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="objective", log_group=objective_logs),
            environment={
                "APP_MODE": "aws",
                "SERVICE_ROLE": "objective",
                "AWS_REGION": self.region or "us-east-1",
                "TARGET_MODEL": target_model,
                "OBJECTIVE_SUITE": objective_suite,
                "OBJECTIVE_SUITE_VERSION": objective_suite_version,
                "S3_ARTIFACT_BUCKET": artifacts.bucket_name,
                "S3_ARTIFACT_PREFIX": prefix,
            },
            secrets={
                "OBJECTIVE_AUTH_TOKEN": ecs.Secret.from_secrets_manager(objective_secret),
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        objective_worker_url = self._text("objective_worker_url")
        certificate_arn = self._text("objective_certificate_arn")
        if not objective_worker_url and not certificate_arn:
            raise ValueError(
                "objective_worker_url or objective_certificate_arn is required"
            )
        if objective_worker_url:
            self._require_https_url(objective_worker_url)
        objective_service_options: dict[str, Any] = {
            "listener_port": 443 if certificate_arn else 80,
            "open_listener": False,
        }
        if certificate_arn:
            self._require_certificate_arn(certificate_arn)
            objective_service_options.update(
                {
                    "protocol": elbv2.ApplicationProtocol.HTTPS,
                    "certificate": acm.Certificate.from_certificate_arn(
                        self, "ObjectiveCertificate", certificate_arn
                    ),
                }
            )
        objective_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ObjectiveService",
            cluster=cluster,
            task_definition=objective_task_definition,
            desired_count=int(self._context("objective_desired_count", 1)),
            public_load_balancer=False,
            assign_public_ip=False,
            health_check_grace_period=Duration.seconds(60),
            **objective_service_options,
        )
        objective_service.target_group.configure_health_check(path="/health", port="8080")
        objective_service.service.connections.allow_from(
            objective_service.load_balancer,
            ec2.Port.tcp(8080),
            "Allow the internal load balancer to reach the objective worker",
        )
        if not objective_worker_url:
            objective_worker_url = "https://" + objective_service.load_balancer.load_balancer_dns_name
        coordinator_security_group = ec2.SecurityGroup(
            self,
            "CoordinatorSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="Coordinator egress to private objective worker only",
        )
        objective_service.load_balancer.connections.allow_from(
            coordinator_security_group,
            ec2.Port.tcp(443 if certificate_arn else 80),
            "Allow only the coordinator to call the objective worker",
        )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDefinition",
            cpu=1024,
            memory_limit_mib=2048,
            task_role=task_role,
            execution_role=execution_role,
        )
        task_definition.add_container(
            "Backend",
            image=self._image(backend_repository, "backend_image_digest"),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="coordinator", log_group=runtime_logs),
            environment=self._coordinator_environment(
                artifacts=artifacts,
                state=state,
                training_role=training_role,
                trainer_repository=trainer_repository,
                evaluator_repository=evaluator_repository,
                objective_worker_url=objective_worker_url,
                prefix=prefix,
                target_model=target_model,
                reasoning_model=reasoning_model,
                objective_suite=objective_suite,
                objective_suite_version=objective_suite_version,
            ),
            secrets={
                "LIVE_APPROVAL_SECRET": ecs.Secret.from_secrets_manager(approval_secret),
                "OBJECTIVE_AUTH_TOKEN": ecs.Secret.from_secrets_manager(objective_secret),
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        service = ecs.FargateService(
            self,
            "RuntimeService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=int(self._context("desired_count", 1)),
            assign_public_ip=False,
            security_groups=[coordinator_security_group],
            health_check_grace_period=Duration.seconds(60),
        )
        service.connections.allow_to(
            objective_service.load_balancer,
            ec2.Port.tcp(443 if certificate_arn else 80),
            "Allow the coordinator to call the internal objective worker",
        )

        logs.MetricFilter(
            self,
            "PhaseTransitionMetric",
            log_group=runtime_logs,
            filter_pattern=logs.FilterPattern.literal('{ $.event_type = "phase.started" }'),
            metric_namespace=self.metric_namespace,
            metric_name="PhaseStarts",
            metric_value="1",
        )
        logs.MetricFilter(
            self,
            "ProviderFailureMetric",
            log_group=runtime_logs,
            filter_pattern=logs.FilterPattern.literal('{ $.event_type = "job.failed" }'),
            metric_namespace=self.metric_namespace,
            metric_name="ProviderFailures",
            metric_value="1",
        )
        logs.MetricFilter(
            self,
            "RunFailureMetric",
            log_group=runtime_logs,
            filter_pattern=logs.FilterPattern.literal('{ $.event_type = "run.failed" }'),
            metric_namespace=self.metric_namespace,
            metric_name="RunFailures",
            metric_value="1",
        )

        CfnOutput(self, "ArtifactBucketName", value=artifacts.bucket_name)
        CfnOutput(self, "StateTableName", value=state.table_name)
        CfnOutput(self, "BackendRepositoryUri", value=backend_repository.repository_uri)
        CfnOutput(self, "TrainerRepositoryUri", value=trainer_repository.repository_uri)
        CfnOutput(self, "EvaluatorRepositoryUri", value=evaluator_repository.repository_uri)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=service.service_name)
        CfnOutput(
            self,
            "ObjectiveWorkerUrl",
            value=objective_worker_url,
            description="Private objective worker endpoint; requests require the objective token",
        )
        CfnOutput(self, "ApprovalSecretArn", value=approval_secret.secret_arn)
        CfnOutput(self, "ObjectiveCredentialSecretArn", value=objective_secret.secret_arn)
        CfnOutput(self, "SageMakerTrainingRoleArn", value=training_role.role_arn)
        CfnOutput(self, "RuntimeLogGroupName", value=runtime_logs.log_group_name)

    def _coordinator_environment(
        self,
        *,
        artifacts: s3.Bucket,
        state: dynamodb.Table,
        training_role: iam.Role,
        trainer_repository: ecr.Repository,
        evaluator_repository: ecr.Repository,
        objective_worker_url: str,
        prefix: str,
        target_model: str,
        reasoning_model: str,
        objective_suite: str,
        objective_suite_version: str,
    ) -> dict[str, str]:
        """Inject the complete live readiness contract into the coordinator."""

        training_uri = self._configured_image_uri(
            "training_image_uri", trainer_repository, "trainer_image_digest"
        )
        evaluation_uri = self._configured_image_uri(
            "evaluation_image_uri", evaluator_repository, "evaluator_image_digest"
        )
        training_input_uri = self._artifact_uri(
            "training_input_s3_uri",
            bucket=artifacts,
            prefix=prefix,
            suffix="inputs/training",
        )
        evaluation_input_uri = self._artifact_uri(
            "evaluation_input_s3_uri",
            bucket=artifacts,
            prefix=prefix,
            suffix="inputs/evaluation",
        )
        checkpoint_uri = self._artifact_uri(
            "checkpoint_s3_uri",
            bucket=artifacts,
            prefix=prefix,
            suffix="checkpoints/base.tar.gz",
            require_version=True,
        )
        return {
            "APP_MODE": "aws",
            "SERVICE_ROLE": "coordinator",
            "AWS_REGION": self.region or "us-east-1",
            "TARGET_MODEL": target_model,
            "STRANDS_MODEL": reasoning_model,
            "OBJECTIVE_SUITE": objective_suite,
            "OBJECTIVE_SUITE_VERSION": objective_suite_version,
            "S3_ARTIFACT_BUCKET": artifacts.bucket_name,
            "S3_ARTIFACT_PREFIX": prefix,
            "DYNAMODB_TABLE_NAME": state.table_name,
            "SAGEMAKER_TRAINING_ROLE_ARN": training_role.role_arn,
            "SAGEMAKER_TRAINING_IMAGE_URI": training_uri,
            "SAGEMAKER_EVALUATION_IMAGE_URI": evaluation_uri,
            "OBJECTIVE_WORKER_URL": objective_worker_url,
            "HF_REPO_ID": self._text("hf_repo_id"),
            "HF_REVISION": self._text("hf_revision"),
            "TRAINING_INPUT_S3_URI": training_input_uri,
            "EVALUATION_INPUT_S3_URI": evaluation_input_uri,
            "CHECKPOINT_S3_URI": checkpoint_uri,
            "CHECKPOINT_SHA256": self._text("checkpoint_sha256"),
            "SAGEMAKER_INSTANCE_TYPE": self._text("sagemaker_instance_type", "ml.g5.xlarge"),
            "GPU_INSTANCE_ALLOWLIST": self._text("gpu_instance_allowlist", "ml.g5.xlarge"),
            "SAGEMAKER_GPU_QUOTA_CODE": self._text("sagemaker_gpu_quota_code"),
            "MINIMUM_GPU_QUOTA": self._text("minimum_gpu_quota", "1"),
            "SAGEMAKER_INSTANCE_COUNT": self._text("sagemaker_instance_count", "1"),
            "SAGEMAKER_VOLUME_SIZE_GB": self._text("sagemaker_volume_size_gb", "30"),
            "LIVE_APPROVAL_SECRET_ENV": self._text(
                "approval_secret_env", "LIVE_APPROVAL_SECRET"
            ),
            "LIVE_APPROVAL_TTL_SECONDS": self._text("approval_ttl_seconds", "900"),
            "MAX_EXPERIMENTS": self._text("max_experiments", "5"),
            "MAX_COST_USD": self._text("max_cost_usd", "25"),
            "MAX_TRAINING_TIME_MIN": self._text("max_training_time_min", "120"),
            "POSTTRAINING_SEED": self._text("posttraining_seed", "7"),
            "TELEMETRY_ENABLED": self._text("telemetry_enabled", "true"),
            "TELEMETRY_EXPORTER": self._text("telemetry_exporter", "logging"),
            "TELEMETRY_OTLP_ENDPOINT": self._text("telemetry_otlp_endpoint"),
            "BEDROCK_AUTH_MODE": self._text("bedrock_auth_mode", "sigv4"),
            "AGENTCORE_ENABLED": self._text("agentcore_enabled", "false"),
            "AGENTCORE_RUNTIME_ARN": self._text("agentcore_runtime_arn"),
        }


__all__ = ["PostTrainingStack"]
