"""Private, version-pinned AWS runtime consuming bootstrap-stack exports."""

from __future__ import annotations

import re
from typing import Any, ClassVar
from urllib.parse import parse_qsl, urlparse

from aws_cdk import CfnOutput, Duration, Fn, Stack, Tags
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as apigw_integrations
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct


class PostTrainingRuntimeStack(Stack):
    """Deploy the coordinator and objective worker into imported private subnets."""

    _IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
    _SHA256 = re.compile(r"^[0-9a-f]{64}$")
    _HF_COMMIT = re.compile(r"^[0-9a-f]{40}$")
    _GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
    _BUILD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$")
    _SERVICE_QUOTA_CODE = re.compile(r"^L-[A-F0-9]{8}$")
    _DEPLOYMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
    _MAX_EXPERIMENTS = 5
    _MAX_COST_USD = 25.0
    _MAX_TRAINING_TIME_MIN = 120
    _GPU_ALLOWLIST = frozenset({"ml.g5.xlarge"})
    _deployment_id_input: str | None = None

    _EXPORTS: ClassVar[dict[str, str]] = {
        "artifact_bucket_name": "ArtifactBucketName",
        "artifact_key_arn": "ArtifactKeyArn",
        "state_table_name": "StateTableName",
        "sagemaker_role_arn": "SageMakerRoleArn",
        "backend_repository_name": "BackendRepositoryName",
        "trainer_repository_name": "TrainerRepositoryName",
        "evaluator_repository_name": "EvaluatorRepositoryName",
        "runtime_log_group_name": "RuntimeLogGroupName",
        "objective_log_group_name": "ObjectiveLogGroupName",
        "vpc_id": "VpcId",
        "availability_zones": "AvailabilityZones",
        "private_subnet_ids": "PrivateSubnetIds",
        "private_subnet_route_table_ids": "PrivateSubnetRouteTableIds",
        "public_subnet_ids": "PublicSubnetIds",
        "public_subnet_route_table_ids": "PublicSubnetRouteTableIds",
        "approval_secret_arn": "ApprovalSecretArn",
        "objective_credential_secret_arn": "ObjectiveCredentialSecretArn",
    }

    def _context(self, name: str, default: Any = None) -> Any:
        value = self.node.try_get_context(name)
        return default if value is None else value

    def _text(self, name: str, default: str | None = None) -> str:
        value = self._context(name, default)
        if value is None:
            return ""
        return str(value).strip()

    def _required_text(self, name: str) -> str:
        value = self._text(name)
        if not value:
            raise ValueError(f"{name} is required for the live AWS runtime")
        return value

    def _integer(
        self,
        name: str,
        *,
        minimum: int,
        maximum: int,
        default: int | None = None,
    ) -> int:
        raw = self._text(name, str(default) if default is not None else None)
        if not raw:
            raise ValueError(f"{name} is required")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer from {minimum} to {maximum}"
            ) from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        return value

    def _cost_limit(self) -> float:
        raw = self._required_text("max_cost_usd")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"max_cost_usd must be a finite value between 0 and {self._MAX_COST_USD:g}"
            ) from exc
        if not 0 < value <= self._MAX_COST_USD:
            raise ValueError(
                f"max_cost_usd must be greater than 0 and at most {self._MAX_COST_USD:g}"
            )
        return value

    def _digest(self, name: str) -> str:
        value = self._required_text(name)
        if not self._IMAGE_DIGEST.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase sha256:<64-hex> image digest")
        return value

    def _sha256(self, name: str) -> str:
        value = self._required_text(name)
        if not self._SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
        return value

    def _artifact_uri(self, name: str, *, prefix: str, versioned: bool) -> str:
        value = self._required_text(name)
        parsed = urlparse(value)
        path = parsed.path.strip("/")
        expected_bucket = self._required_text("artifact_bucket_name")
        if parsed.netloc != expected_bucket:
            raise ValueError(
                f"{name} bucket must equal artifact_bucket_name from the bootstrap stack output"
            )
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not path
            or parsed.fragment
            or ".." in path.split("/")
        ):
            raise ValueError(f"{name} must be a valid S3 artifact URI")
        if parsed.username or parsed.password or parsed.port:
            raise ValueError(
                f"{name} must be an S3 URI without user information or port"
            )

        if versioned:
            query = parse_qsl(parsed.query, keep_blank_values=True)
            versions = [value for key, value in query if key == "versionId"]
            if len(versions) != 1 or not versions[0] or len(query) != 1:
                raise ValueError(f"{name} must include exactly one non-empty versionId")
        elif parsed.query:
            raise ValueError(
                f"{name} must be an immutable content-addressed S3 prefix without query parameters"
            )

        normalized_prefix = prefix.strip("/")
        if not normalized_prefix or not path.startswith(normalized_prefix + "/"):
            raise ValueError(f"{name} must be beneath the configured artifact_prefix")
        return value

    def _deployment_id(self) -> str:
        deployment_id = (
            self._deployment_id_input or self._text("deployment_id", "demo")
        ).lower()
        if not self._DEPLOYMENT_ID.fullmatch(deployment_id):
            raise ValueError(
                "deployment_id must be a lowercase identifier of letters, digits, and hyphens"
            )
        return deployment_id

    def _imports(self, deployment_id: str) -> dict[str, str]:
        return {
            key: Fn.import_value(f"aptraining-{deployment_id}-{suffix}")
            for key, suffix in self._EXPORTS.items()
        }

    def _validate_inputs(self) -> dict[str, Any]:
        mode = self._text("ingress_mode", "api_gateway").lower()
        if mode == "direct_tls":
            required_direct_tls = (
                "coordinator_ingress_cidrs",
                "coordinator_certificate_arn",
                "coordinator_public_dns_name",
            )
            missing_direct_tls = [
                name for name in required_direct_tls if not self._text(name)
            ]
            if missing_direct_tls:
                raise ValueError(
                    "direct_tls mode requires coordinator_ingress_cidrs, "
                    "coordinator_certificate_arn, and coordinator_public_dns_name"
                )
            raise ValueError(
                "direct_tls mode is not implemented in PostTrainingRuntimeStack"
            )
        if mode != "api_gateway":
            raise ValueError(
                "this runtime stack supports only ingress_mode=api_gateway"
            )

        prefix = self._text("artifact_prefix", "post-training").strip("/")
        if prefix != "post-training":
            raise ValueError(
                "artifact_prefix is pinned to the bootstrap IAM scope post-training"
            )
        if self._text("runtime_nat_gateway_acknowledged").lower() != "true":
            raise ValueError(
                "runtime_nat_gateway_acknowledged=true is required; this runtime-only NAT Gateway has recurring hourly charges ($0.045 per gateway-hour is AWS's US East (Ohio) example; regional rates vary), per-GB processing charges, and applicable data-transfer fees. Bootstrap remains NAT-free"
            )

        digests = {
            "backend_image_digest": self._digest("backend_image_digest"),
            "trainer_image_digest": self._digest("trainer_image_digest"),
            "evaluator_image_digest": self._digest("evaluator_image_digest"),
        }
        git_sha = self._required_text("git_sha")
        if not self._GIT_SHA.fullmatch(git_sha):
            raise ValueError("git_sha must be an exact lowercase 40-character commit SHA")
        build_id = self._required_text("build_id")
        if not self._BUILD_ID.fullmatch(build_id):
            raise ValueError(
                "build_id must be 1-128 safe alphanumeric, colon, dot, underscore, or hyphen characters"
            )
        checkpoint_uri = self._artifact_uri(
            "checkpoint_s3_uri", prefix=prefix, versioned=True
        )
        checkpoint_sha = self._sha256("checkpoint_sha256")
        hf_revision = self._required_text("hf_revision")
        if not self._HF_COMMIT.fullmatch(hf_revision):
            raise ValueError(
                "hf_revision must be an exact lowercase 40-character Hugging Face commit"
            )

        evaluation_uri = self._artifact_uri(
            "evaluation_input_s3_uri", prefix=prefix, versioned=False
        )
        evaluation_sha = self._sha256("evaluation_manifest_sha256")
        evaluation_path_parts = urlparse(evaluation_uri).path.strip("/").split("/")
        if evaluation_sha not in evaluation_path_parts:
            raise ValueError(
                "evaluation_input_s3_uri must be a content-addressed prefix containing evaluation_manifest_sha256"
            )

        model = self._required_text("target_model")
        if model != "google/functiongemma-270m-it":
            raise ValueError("target_model is pinned to google/functiongemma-270m-it")
        suite = self._required_text("objective_suite")
        suite_version = self._required_text("objective_suite_version")
        seed = self._integer("posttraining_seed", minimum=0, maximum=2**31 - 1)
        baseline_episodes = self._integer(
            "baseline_episodes", minimum=1, maximum=10, default=1
        )

        instance_type = self._required_text("sagemaker_instance_type")
        gpu_quota_code = self._required_text("sagemaker_gpu_quota_code")
        if not self._SERVICE_QUOTA_CODE.fullmatch(gpu_quota_code):
            raise ValueError(
                "sagemaker_gpu_quota_code must be an AWS Service Quotas code in L-XXXXXXXX format"
            )
        processing_gpu_quota_code = self._required_text(
            "sagemaker_processing_gpu_quota_code"
        )
        if not self._SERVICE_QUOTA_CODE.fullmatch(processing_gpu_quota_code):
            raise ValueError(
                "sagemaker_processing_gpu_quota_code must be an AWS Service Quotas code in L-XXXXXXXX format"
            )
        allowlist = tuple(
            sorted(
                {
                    item.strip()
                    for item in self._required_text("gpu_instance_allowlist").split(",")
                    if item.strip()
                }
            )
        )
        if not allowlist or not set(allowlist).issubset(self._GPU_ALLOWLIST):
            raise ValueError(
                "gpu_instance_allowlist must stay within the approved bounded GPU set"
            )
        if instance_type not in allowlist:
            raise ValueError(
                "sagemaker_instance_type must be included in gpu_instance_allowlist"
            )
        if instance_type not in self._GPU_ALLOWLIST:
            raise ValueError(
                "sagemaker_instance_type is outside the approved bounded GPU set"
            )

        max_experiments = self._integer(
            "max_experiments", minimum=1, maximum=self._MAX_EXPERIMENTS
        )
        max_cost = self._cost_limit()
        max_training_time = self._integer(
            "max_training_time_min", minimum=1, maximum=self._MAX_TRAINING_TIME_MIN
        )
        instance_count = self._integer("sagemaker_instance_count", minimum=1, maximum=1)
        volume_size = self._integer("sagemaker_volume_size_gb", minimum=1, maximum=100)
        minimum_gpu_quota = self._required_text("minimum_gpu_quota")
        try:
            quota = float(minimum_gpu_quota)
        except ValueError as exc:
            raise ValueError("minimum_gpu_quota must be finite and positive") from exc
        if not 0 < quota <= 1:
            raise ValueError("minimum_gpu_quota must be greater than 0 and at most 1")
        approval_ttl = self._integer("approval_ttl_seconds", minimum=60, maximum=86400)

        return {
            "artifact_prefix": prefix,
            **digests,
            "git_sha": git_sha,
            "build_id": build_id,
            "checkpoint_s3_uri": checkpoint_uri,
            "checkpoint_sha256": checkpoint_sha,
            "hf_revision": hf_revision,
            "evaluation_input_s3_uri": evaluation_uri,
            "evaluation_manifest_sha256": evaluation_sha,
            "target_model": model,
            "objective_suite": suite,
            "objective_suite_version": suite_version,
            "posttraining_seed": seed,
            "baseline_episodes": baseline_episodes,
            "sagemaker_instance_type": instance_type,
            "sagemaker_gpu_quota_code": gpu_quota_code,
            "sagemaker_processing_gpu_quota_code": processing_gpu_quota_code,
            "gpu_instance_allowlist": ",".join(allowlist),
            "minimum_gpu_quota": quota,
            "sagemaker_instance_count": instance_count,
            "sagemaker_volume_size_gb": volume_size,
            "max_experiments": max_experiments,
            "max_cost_usd": max_cost,
            "max_training_time_min": max_training_time,
            "approval_ttl_seconds": approval_ttl,
        }

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        deployment_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._deployment_id_input = deployment_id
        deployment_id = self._deployment_id()
        config = self._validate_inputs()
        imported = self._imports(deployment_id)
        self._configure_runtime(deployment_id, config, imported)

    def _configure_runtime(
        self, deployment_id: str, config: dict[str, Any], imported: dict[str, str]
    ) -> None:
        Tags.of(self).add("ManagedBy", "autonomous-post-training")
        Tags.of(self).add("DeploymentId", deployment_id)

        azs = Fn.split(",", imported["availability_zones"], assumed_length=2)
        private_ids = Fn.split(",", imported["private_subnet_ids"], assumed_length=2)
        private_route_tables = Fn.split(
            ",", imported["private_subnet_route_table_ids"], assumed_length=2
        )
        public_ids = Fn.split(",", imported["public_subnet_ids"], assumed_length=2)
        public_route_tables = Fn.split(
            ",", imported["public_subnet_route_table_ids"], assumed_length=2
        )
        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "ImportedRuntimeVpc",
            vpc_id=imported["vpc_id"],
            availability_zones=azs,
            private_subnet_ids=private_ids,
            private_subnet_route_table_ids=private_route_tables,
            public_subnet_ids=public_ids,
            public_subnet_route_table_ids=public_route_tables,
        )
        private_subnets = ec2.SubnetSelection(subnets=vpc.private_subnets)

        # The bootstrap stack intentionally remains NAT-free. Runtime-only
        # egress is explicit and requires the synthesis acknowledgement above.
        nat_eip = ec2.CfnEIP(self, "RuntimeNatEip", domain="vpc")
        nat_gateway = ec2.CfnNatGateway(
            self,
            "RuntimeNatGateway",
            allocation_id=nat_eip.attr_allocation_id,
            subnet_id=Fn.select(0, public_ids),
            connectivity_type="public",
        )
        for subnet_index in range(2):
            ec2.CfnRoute(
                self,
                f"PrivateSubnetDefaultRoute{subnet_index + 1}",
                route_table_id=Fn.select(subnet_index, private_route_tables),
                destination_cidr_block="0.0.0.0/0",
                nat_gateway_id=nat_gateway.ref,
            )

        bucket = s3.Bucket.from_bucket_name(
            self, "ImportedArtifactBucket", imported["artifact_bucket_name"]
        )
        artifact_key = kms.Key.from_key_arn(
            self, "ImportedArtifactKey", imported["artifact_key_arn"]
        )
        state_table = dynamodb.Table.from_table_name(
            self, "ImportedRunState", imported["state_table_name"]
        )
        ec2.GatewayVpcEndpoint(
            self,
            "ArtifactS3GatewayEndpoint",
            vpc=vpc,
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[private_subnets],
        )
        ec2.GatewayVpcEndpoint(
            self,
            "RunStateDynamoDbGatewayEndpoint",
            vpc=vpc,
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
            subnets=[private_subnets],
        )
        sage_role_arn = imported["sagemaker_role_arn"]
        backend_repository = ecr.Repository.from_repository_name(
            self, "ImportedBackendRepository", imported["backend_repository_name"]
        )
        trainer_repository = ecr.Repository.from_repository_name(
            self, "ImportedTrainerRepository", imported["trainer_repository_name"]
        )
        evaluator_repository = ecr.Repository.from_repository_name(
            self, "ImportedEvaluatorRepository", imported["evaluator_repository_name"]
        )
        runtime_logs = logs.LogGroup.from_log_group_name(
            self, "ImportedRuntimeLogs", imported["runtime_log_group_name"]
        )
        objective_logs = logs.LogGroup.from_log_group_name(
            self, "ImportedObjectiveLogs", imported["objective_log_group_name"]
        )
        approval_secret = secretsmanager.Secret.from_secret_complete_arn(
            self,
            "ImportedApprovalSecret",
            secret_complete_arn=imported["approval_secret_arn"],
        )
        objective_secret = secretsmanager.Secret.from_secret_complete_arn(
            self,
            "ImportedObjectiveCredential",
            secret_complete_arn=imported["objective_credential_secret_arn"],
        )

        cluster = ecs.Cluster(
            self,
            "RuntimeCluster",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )
        task_role = iam.Role(
            self,
            "CoordinatorTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        objective_role = iam.Role(
            self,
            "ObjectiveTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        coordinator_execution_role = iam.Role(
            self,
            "CoordinatorExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
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
        backend_repository.grant_pull(coordinator_execution_role)
        backend_repository.grant_pull(objective_execution_role)
        approval_secret.grant_read(coordinator_execution_role)
        objective_secret.grant_read(coordinator_execution_role)
        objective_secret.grant_read(objective_execution_role)

        prefix = config["artifact_prefix"]
        bucket.grant_read_write(task_role, f"{prefix}/*")
        bucket.grant_read_write(objective_role, f"{prefix}/*")
        state_table.grant_read_write_data(task_role)
        artifact_key.grant_encrypt_decrypt(task_role)
        # The objective worker persists newly verified trajectories and datasets
        # into the KMS-encrypted artifact bucket, so it needs data-key/encrypt
        # permissions in addition to decrypting the staged base checkpoint.
        artifact_key.grant_encrypt_decrypt(objective_role)

        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadinessArtifactBucketMetadata",
                actions=[
                    "s3:GetBucketLocation",
                    "s3:GetEncryptionConfiguration",
                    "s3:GetBucketVersioning",
                ],
                resources=[bucket.bucket_arn],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadinessSageMakerRole",
                actions=["iam:GetRole"],
                resources=[sage_role_arn],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadinessWorkerImages",
                actions=["ecr:DescribeImages"],
                resources=[
                    trainer_repository.repository_arn,
                    evaluator_repository.repository_arn,
                ],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadinessServiceQuotas",
                actions=["servicequotas:GetServiceQuota"],
                resources=[
                    f"arn:{self.partition}:servicequotas:{self.region}:{self.account}:sagemaker/{config['sagemaker_gpu_quota_code']}",
                    f"arn:{self.partition}:servicequotas:{self.region}:{self.account}:sagemaker/{config['sagemaker_processing_gpu_quota_code']}",
                ],
            )
        )
        objective_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadPinnedObjectiveCheckpointVersion",
                actions=["s3:GetObjectVersion"],
                resources=[f"{bucket.bucket_arn}/{prefix}/*"],
            )
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeBoundedSageMakerJobs",
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:CreateTrainingJob",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "aws:RequestTag/project": "autonomous-post-training"
                    }
                },
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ManageTaggedSageMakerJobs",
                actions=[
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:ListTags",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:StopTrainingJob",
                ],
                resources=[
                    f"arn:{self.partition}:sagemaker:{self.region}:{self.account}:processing-job/*",
                    f"arn:{self.partition}:sagemaker:{self.region}:{self.account}:training-job/*",
                ],
                conditions={
                    "StringEquals": {
                        "sagemaker:ResourceTag/project": "autonomous-post-training"
                    }
                },
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="PassBootstrapSageMakerRole",
                actions=["iam:PassRole"],
                resources=[sage_role_arn],
                conditions={
                    "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
                },
            )
        )
        strands_model = self._text("strands_model", "nvidia.nemotron-super-3-120b")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,127}", strands_model):
            raise ValueError("strands_model must be a Bedrock foundation model ID")
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokePinnedBedrockFoundationModel",
                actions=["bedrock:GetFoundationModel", "bedrock:InvokeModel"],
                resources=[
                    f"arn:{self.partition}:bedrock:{self.region}::foundation-model/{strands_model}"
                ],
            )
        )

        backend_digest = config["backend_image_digest"]
        coordinator_image = ecs.ContainerImage.from_registry(
            f"{backend_repository.repository_uri}@{backend_digest}"
        )
        objective_image = coordinator_image
        trainer_image = (
            f"{trainer_repository.repository_uri}@{config['trainer_image_digest']}"
        )
        evaluator_image = (
            f"{evaluator_repository.repository_uri}@{config['evaluator_image_digest']}"
        )

        vpc_link_sg = ec2.SecurityGroup(
            self,
            "ApiGatewayVpcLinkSecurityGroup",
            vpc=vpc,
            allow_all_outbound=False,
            description="HTTP API VPC Link egress only to the two private ALBs",
        )
        coordinator_alb_sg = ec2.SecurityGroup(
            self,
            "CoordinatorPrivateAlbSecurityGroup",
            vpc=vpc,
            allow_all_outbound=False,
            description="Coordinator private ALB accepts HTTP only from API Gateway VPC Link",
        )
        objective_alb_sg = ec2.SecurityGroup(
            self,
            "ObjectivePrivateAlbSecurityGroup",
            vpc=vpc,
            allow_all_outbound=False,
            description="Objective private ALB accepts HTTP only from API Gateway VPC Link",
        )
        coordinator_task_sg = ec2.SecurityGroup(
            self,
            "CoordinatorTaskSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
        )
        objective_task_sg = ec2.SecurityGroup(
            self,
            "ObjectiveTaskSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
        )

        for alb_sg in (coordinator_alb_sg, objective_alb_sg):
            alb_sg.add_ingress_rule(
                vpc_link_sg,
                ec2.Port.tcp(80),
                "Allow the API Gateway VPC Link only",
            )
        vpc_link_sg.add_egress_rule(
            coordinator_alb_sg, ec2.Port.tcp(80), "Reach the coordinator private ALB"
        )
        vpc_link_sg.add_egress_rule(
            objective_alb_sg, ec2.Port.tcp(80), "Reach the objective private ALB"
        )
        coordinator_alb_sg.add_egress_rule(
            coordinator_task_sg, ec2.Port.tcp(8080), "Forward to coordinator tasks"
        )
        objective_alb_sg.add_egress_rule(
            objective_task_sg, ec2.Port.tcp(8080), "Forward to objective tasks"
        )
        coordinator_task_sg.add_ingress_rule(
            coordinator_alb_sg, ec2.Port.tcp(8080), "Accept coordinator ALB traffic"
        )
        objective_task_sg.add_ingress_rule(
            objective_alb_sg, ec2.Port.tcp(8080), "Accept objective ALB traffic"
        )

        coordinator_alb = elbv2.ApplicationLoadBalancer(
            self,
            "CoordinatorPrivateAlb",
            vpc=vpc,
            internet_facing=False,
            security_group=coordinator_alb_sg,
            vpc_subnets=private_subnets,
        )
        objective_alb = elbv2.ApplicationLoadBalancer(
            self,
            "ObjectivePrivateAlb",
            vpc=vpc,
            internet_facing=False,
            security_group=objective_alb_sg,
            vpc_subnets=private_subnets,
        )
        coordinator_listener = coordinator_alb.add_listener(
            "CoordinatorHttpListener",
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
        )
        objective_listener = objective_alb.add_listener(
            "ObjectiveHttpListener",
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
        )

        http_api = apigw.HttpApi(
            self,
            "RuntimeHttpApi",
            api_name=f"aptraining-{deployment_id}-runtime",
            description="HTTPS API Gateway entry point to private coordinator and objective services",
            create_default_stage=True,
            disable_execute_api_endpoint=False,
        )
        vpc_link = apigw.VpcLink(
            self,
            "RuntimeVpcLink",
            vpc=vpc,
            vpc_link_name=f"aptraining-{deployment_id}-runtime-link",
            security_groups=[vpc_link_sg],
            subnets=private_subnets,
        )
        coordinator_integration = apigw_integrations.HttpAlbIntegration(
            "CoordinatorAlbIntegration",
            coordinator_listener,
            method=apigw.HttpMethod.ANY,
            vpc_link=vpc_link,
        )
        objective_integration = apigw_integrations.HttpAlbIntegration(
            "ObjectiveAlbIntegration",
            objective_listener,
            method=apigw.HttpMethod.ANY,
            vpc_link=vpc_link,
        )
        http_api.add_routes(
            path="/api/{proxy+}",
            methods=[apigw.HttpMethod.ANY],
            integration=coordinator_integration,
        )
        http_api.add_routes(
            path="/health",
            methods=[apigw.HttpMethod.ANY],
            integration=coordinator_integration,
        )
        http_api.add_routes(
            path="/v1/{proxy+}",
            methods=[apigw.HttpMethod.ANY],
            integration=objective_integration,
        )
        objective_worker_url = f"{http_api.api_endpoint}/v1/"

        coordinator_task = ecs.FargateTaskDefinition(
            self,
            "CoordinatorTaskDefinition",
            cpu=1024,
            memory_limit_mib=2048,
            task_role=task_role,
            execution_role=coordinator_execution_role,
        )
        coordinator_task.add_container(
            "Coordinator",
            image=coordinator_image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="coordinator", log_group=runtime_logs
            ),
            environment={
                "APP_MODE": "aws",
                "SERVICE_ROLE": "coordinator",
                "AWS_REGION": self.region or "us-east-1",
                "TARGET_MODEL": config["target_model"],
                "GIT_SHA": config["git_sha"],
                "BUILD_ID": config["build_id"],
                "IMAGE_DIGEST": config["backend_image_digest"],
                "STRANDS_MODEL": strands_model,
                "OBJECTIVE_SUITE": config["objective_suite"],
                "OBJECTIVE_SUITE_VERSION": config["objective_suite_version"],
                "OBJECTIVE_WORKER_URL": objective_worker_url,
                "S3_ARTIFACT_BUCKET": imported["artifact_bucket_name"],
                "S3_ARTIFACT_PREFIX": config["artifact_prefix"],
                "DYNAMODB_TABLE_NAME": imported["state_table_name"],
                "SAGEMAKER_TRAINING_ROLE_ARN": sage_role_arn,
                "SAGEMAKER_TRAINING_IMAGE_URI": trainer_image,
                "SAGEMAKER_EVALUATION_IMAGE_URI": evaluator_image,
                "HF_REPO_ID": config["target_model"],
                "HF_REVISION": config["hf_revision"],
                "CHECKPOINT_S3_URI": config["checkpoint_s3_uri"],
                "CHECKPOINT_SHA256": config["checkpoint_sha256"],
                "EVALUATION_INPUT_S3_URI": config["evaluation_input_s3_uri"],
                "LIVE_BENCHMARK_MANIFEST_SHA256": config["evaluation_manifest_sha256"],
                "SAGEMAKER_INSTANCE_TYPE": config["sagemaker_instance_type"],
                "SAGEMAKER_GPU_QUOTA_CODE": config["sagemaker_gpu_quota_code"],
                "SAGEMAKER_PROCESSING_GPU_QUOTA_CODE": config[
                    "sagemaker_processing_gpu_quota_code"
                ],
                "GPU_INSTANCE_ALLOWLIST": config["gpu_instance_allowlist"],
                "MINIMUM_GPU_QUOTA": str(config["minimum_gpu_quota"]),
                "SAGEMAKER_INSTANCE_COUNT": str(config["sagemaker_instance_count"]),
                "SAGEMAKER_VOLUME_SIZE_GB": str(config["sagemaker_volume_size_gb"]),
                "MAX_EXPERIMENTS": str(config["max_experiments"]),
                "MAX_COST_USD": str(config["max_cost_usd"]),
                "MAX_TRAINING_TIME_MIN": str(config["max_training_time_min"]),
                "POSTTRAINING_SEED": str(config["posttraining_seed"]),
                "BASELINE_EPISODES": str(config["baseline_episodes"]),
                "LIVE_APPROVAL_SECRET_ENV": "LIVE_APPROVAL_SECRET",
                "LIVE_APPROVAL_TTL_SECONDS": str(config["approval_ttl_seconds"]),
            },
            secrets={
                "LIVE_APPROVAL_SECRET": ecs.Secret.from_secrets_manager(
                    approval_secret
                ),
                "OBJECTIVE_WORKER_AUTH_TOKEN": ecs.Secret.from_secrets_manager(
                    objective_secret
                ),
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        coordinator_service = ecs.FargateService(
            self,
            "CoordinatorService",
            cluster=cluster,
            task_definition=coordinator_task,
            desired_count=self._integer("desired_count", minimum=0, maximum=2)
            if self._text("desired_count")
            else 1,
            assign_public_ip=False,
            security_groups=[coordinator_task_sg],
            vpc_subnets=private_subnets,
            health_check_grace_period=Duration.seconds(60),
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            min_healthy_percent=100,
        )
        coordinator_listener.add_targets(
            "CoordinatorTargets",
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[coordinator_service],
            health_check=elbv2.HealthCheck(path="/health", port="8080"),
        )

        objective_task = ecs.FargateTaskDefinition(
            self,
            "ObjectiveTaskDefinition",
            cpu=2048,
            memory_limit_mib=4096,
            task_role=objective_role,
            execution_role=objective_execution_role,
        )
        objective_task.add_container(
            "Objective",
            image=objective_image,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="objective", log_group=objective_logs
            ),
            environment={
                "APP_MODE": "aws",
                "SERVICE_ROLE": "objective",
                "AWS_REGION": self.region or "us-east-1",
                "TARGET_MODEL": config["target_model"],
                "OBJECTIVE_MODEL_CHECKPOINT_DIR": self._text(
                    "objective_model_checkpoint_dir", "/opt/models/functiongemma"
                ),
                "OBJECTIVE_MODEL_REVISION": config["hf_revision"],
                "OBJECTIVE_BASE_MODEL_URI": config["checkpoint_s3_uri"],
                "OBJECTIVE_BASE_MODEL_SHA256": config["checkpoint_sha256"],
                "OBJECTIVE_SUITE": config["objective_suite"],
                "OBJECTIVE_SUITE_VERSION": config["objective_suite_version"],
                "S3_ARTIFACT_BUCKET": imported["artifact_bucket_name"],
                "S3_ARTIFACT_PREFIX": config["artifact_prefix"],
            },
            secrets={
                "OBJECTIVE_AUTH_TOKEN": ecs.Secret.from_secrets_manager(
                    objective_secret
                )
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        objective_service = ecs.FargateService(
            self,
            "ObjectiveService",
            cluster=cluster,
            task_definition=objective_task,
            desired_count=self._integer("objective_desired_count", minimum=0, maximum=2)
            if self._text("objective_desired_count")
            else 1,
            assign_public_ip=False,
            security_groups=[objective_task_sg],
            vpc_subnets=private_subnets,
            health_check_grace_period=Duration.seconds(60),
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            min_healthy_percent=100,
        )
        objective_listener.add_targets(
            "ObjectiveTargets",
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[objective_service],
            health_check=elbv2.HealthCheck(path="/health", port="8080"),
        )

        CfnOutput(self, "HttpApiEndpoint", value=http_api.api_endpoint)
        CfnOutput(
            self,
            "BootstrapAvailabilityZones",
            value=Fn.join(",", azs),
            description="Availability zones imported from the bootstrap VPC contract.",
        )
        CfnOutput(
            self,
            "BootstrapPublicSubnetRouteTableIds",
            value=Fn.join(",", public_route_tables),
            description="Public route tables imported from the bootstrap VPC contract.",
        )
        CfnOutput(
            self,
            "ObjectiveWorkerApiUrl",
            value=objective_worker_url,
            description="Objective worker URL on the HTTPS HTTP API; requests require the bearer secret.",
        )
        CfnOutput(self, "RuntimeClusterName", value=cluster.cluster_name)
        CfnOutput(
            self, "CoordinatorServiceName", value=coordinator_service.service_name
        )
        CfnOutput(self, "ObjectiveServiceName", value=objective_service.service_name)


__all__ = ["PostTrainingRuntimeStack"]
