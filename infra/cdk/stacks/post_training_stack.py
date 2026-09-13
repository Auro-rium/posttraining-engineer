"""Synthesizable AWS boundary for the autonomous post-training backend.

This stack creates the durable, authenticated boundaries used by the live
controller. It deliberately does not push images, stage checkpoints, submit
SageMaker jobs, or otherwise perform an execution-side effect.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from aws_cdk import Arn, CfnOutput, Duration, RemovalPolicy, Stack, Tags, Token
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
from aws_cdk import aws_route53 as route53
from aws_cdk import aws_route53_targets as route53_targets
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
            if require_version:
                raise ValueError(
                    f"{name} must be explicitly configured with an immutable versionId"
                )
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
        version_ids = parse_qs(parsed.query).get("versionId", [])
        if require_version and (len(version_ids) != 1 or not version_ids[0]):
            raise ValueError(f"{name} must include exactly one immutable versionId")
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

    def _coordinator_ingress_ipv4_cidrs(self) -> tuple[str, ...]:
        """Require a narrow, explicit IPv4 allowlist for the public API ALB."""

        raw = self._text("coordinator_ingress_cidrs").strip()
        if not raw:
            raise ValueError(
                "coordinator_ingress_cidrs must explicitly allow the demo operator's IPv4 CIDR"
            )
        networks: set[str] = set()
        for item in raw.split(","):
            candidate = item.strip()
            if "/" not in candidate:
                raise ValueError(
                    "coordinator_ingress_cidrs must contain valid IPv4 CIDRs"
                )
            try:
                network = ipaddress.ip_network(candidate, strict=True)
            except ValueError as exc:
                raise ValueError(
                    "coordinator_ingress_cidrs must contain valid IPv4 CIDRs"
                ) from exc
            if not isinstance(network, ipaddress.IPv4Network):
                raise ValueError(
                    "coordinator_ingress_cidrs must contain valid IPv4 CIDRs"
                )
            if network.prefixlen < 16:
                raise ValueError(
                    "coordinator_ingress_cidrs must be IPv4 networks with prefix /16 or narrower"
                )
            networks.add(str(network))
        return tuple(sorted(networks))

    def _private_dns_contract(self) -> tuple[str, str]:
        """Validate the private DNS and certificate SAN contract.

        ACM certificates are imported by ARN and their SANs are not available
        to CloudFormation at synth time.  The configured SAN is therefore an
        explicit deployment contract: the operator must provide the exact
        private hostname (or a single-label wildcard that covers it).  The
        Route 53 zone is created in this stack and attached to the runtime VPC,
        so the coordinator never receives the ALB-generated AWS hostname.
        """

        hostname = self._text("objective_private_dns_name").strip().rstrip(".").lower()
        zone_name = self._text("objective_private_hosted_zone_name").strip().rstrip(".").lower()
        certificate_san = self._text("objective_certificate_san").strip().rstrip(".").lower()
        label = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        hostname_pattern = re.compile(rf"^(?:{label})(?:\.(?:{label})){{0,126}}$")
        for name, value in (
            ("objective_private_dns_name", hostname),
            ("objective_private_hosted_zone_name", zone_name),
        ):
            if not value or not hostname_pattern.fullmatch(value):
                raise ValueError(f"{name} must be a valid private DNS hostname")
        san_to_validate = certificate_san.removeprefix("*.")
        if not certificate_san or not hostname_pattern.fullmatch(san_to_validate):
            raise ValueError("objective_certificate_san must be a valid DNS SAN")
        if hostname != zone_name and not hostname.endswith("." + zone_name):
            raise ValueError(
                "objective_private_dns_name must be within objective_private_hosted_zone_name"
            )
        if certificate_san.startswith("*."):
            wildcard_suffix = certificate_san[2:]
            wildcard_matches = (
                hostname.endswith("." + wildcard_suffix)
                and hostname.count(".") == wildcard_suffix.count(".") + 1
            )
        else:
            wildcard_matches = hostname == certificate_san
        if not wildcard_matches:
            raise ValueError(
                "objective_private_dns_name must match objective_certificate_san"
            )
        return hostname, zone_name

    def _coordinator_public_dns_contract(self) -> tuple[str, str, str]:
        """Require a public hostname and matching operator-supplied certificate SAN.

        ACM certificate SANs and imported hosted-zone visibility cannot be
        inspected during synthesis. The SAN and explicitly named public zone
        are therefore deployment contracts, just as the zone ID is an
        operator-supplied reference to the public Route 53 zone.
        """

        hostname = self._text("coordinator_public_dns_name").strip().rstrip(".").lower()
        zone_name = (
            self._text("coordinator_public_hosted_zone_name")
            .strip()
            .rstrip(".")
            .lower()
        )
        zone_id = self._text("coordinator_public_hosted_zone_id").strip()
        certificate_san = self._text("coordinator_certificate_san").strip().rstrip(".").lower()
        label = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        hostname_pattern = re.compile(rf"^(?:{label})(?:\.(?:{label})){{0,126}}$")
        if not hostname or not hostname_pattern.fullmatch(hostname):
            raise ValueError("coordinator_public_dns_name must be a valid DNS hostname")
        if not zone_name or not hostname_pattern.fullmatch(zone_name):
            raise ValueError(
                "coordinator_public_hosted_zone_name must be a valid DNS hostname"
            )
        if not re.fullmatch(r"Z[A-Z0-9]+", zone_id):
            raise ValueError(
                "coordinator_public_hosted_zone_id must be a Route 53 zone ID"
            )
        san_to_validate = certificate_san.removeprefix("*.")
        if not certificate_san or not hostname_pattern.fullmatch(san_to_validate):
            raise ValueError("coordinator_certificate_san must be a valid DNS SAN")
        if hostname != zone_name and not hostname.endswith("." + zone_name):
            raise ValueError(
                "coordinator_public_dns_name must be within coordinator_public_hosted_zone_name"
            )
        if certificate_san.startswith("*."):
            wildcard_suffix = certificate_san[2:]
            san_matches = (
                hostname.endswith("." + wildcard_suffix)
                and hostname.count(".") == wildcard_suffix.count(".") + 1
            )
        else:
            san_matches = hostname == certificate_san
        if not san_matches:
            raise ValueError(
                "coordinator_public_dns_name must match coordinator_certificate_san"
            )
        return hostname, zone_name, zone_id

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = self._text("artifact_prefix", "post-training").strip("/") or "post-training"
        coordinator_ingress_cidrs = self._coordinator_ingress_ipv4_cidrs()
        coordinator_certificate_arn = self._text("coordinator_certificate_arn").strip()
        if not coordinator_certificate_arn:
            raise ValueError(
                "coordinator_certificate_arn is required for the public HTTPS API"
            )
        self._require_certificate_arn(coordinator_certificate_arn)
        certificate_region = coordinator_certificate_arn.split(":")[3]
        if (
            self.region
            and not Token.is_unresolved(self.region)
            and certificate_region != self.region
        ):
            raise ValueError(
                "coordinator_certificate_arn must be in the stack's AWS region"
            )
        (
            coordinator_dns_name,
            coordinator_zone_name,
            coordinator_zone_id,
        ) = self._coordinator_public_dns_contract()
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

        # An externally hosted objective worker is already the complete
        # objective boundary. Only certificate-only deployments create the
        # internal worker and its ALB; an external URL must never leave an
        # unused plaintext listener in the stack.
        objective_worker_url = self._text("objective_worker_url").strip()
        certificate_arn = self._text("objective_certificate_arn").strip()
        if objective_worker_url:
            self._require_https_url(objective_worker_url)
        elif certificate_arn:
            self._require_certificate_arn(certificate_arn)
        else:
            raise ValueError(
                "objective_worker_url or objective_certificate_arn is required"
            )
        internal_objective = not objective_worker_url
        objective_checkpoint_uri = ""
        objective_checkpoint_revision = ""
        objective_checkpoint_sha256 = ""
        if internal_objective:
            objective_checkpoint_uri = self._artifact_uri(
                "checkpoint_s3_uri",
                bucket=artifacts,
                prefix=prefix,
                suffix="checkpoints/base.tar.gz",
                require_version=True,
            )
            objective_checkpoint_revision = self._text("hf_revision")
            objective_checkpoint_sha256 = self._text("checkpoint_sha256")
            uri = urlparse(objective_checkpoint_uri)
            version_ids = parse_qs(uri.query).get("versionId", [])
            if not version_ids or not version_ids[0]:
                raise ValueError(
                    "internal objective worker requires a version-pinned checkpoint_s3_uri"
                )
            if not re.fullmatch(r"[0-9a-f]{40}", objective_checkpoint_revision):
                raise ValueError(
                    "internal objective worker requires hf_revision as a lowercase commit SHA"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", objective_checkpoint_sha256):
                raise ValueError(
                    "internal objective worker requires checkpoint_sha256 from checkpoint handoff"
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
                sid="ListTagsForTaggedSageMakerJobs",
                actions=["sagemaker:ListTags"],
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
        # These are the coordinator's bounded, read-only readiness probes.
        # Keep each resource set narrow: the S3 bucket, the SageMaker role,
        # and only the two worker repositories created above.  Service Quotas
        # does not support resource-level authorization, so that one action
        # necessarily uses the documented wildcard resource.
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadOnlyPreflightS3",
                actions=[
                    "s3:GetBucketLocation",
                    "s3:GetEncryptionConfiguration",
                    "s3:GetBucketVersioning",
                ],
                resources=[artifacts.bucket_arn],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadOnlyPreflightS3ObjectVersions",
                actions=["s3:GetObjectVersion"],
                resources=[f"{artifacts.bucket_arn}/{prefix}/*"],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadOnlyPreflightIam",
                actions=["iam:GetRole"],
                resources=[training_role.role_arn],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadOnlyPreflightEcr",
                actions=["ecr:DescribeImages"],
                resources=[trainer_repository.repository_arn, evaluator_repository.repository_arn],
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadOnlyPreflightServiceQuota",
                actions=["servicequotas:GetServiceQuota"],
                resources=["*"],
            )
        )

        objective_role = (
            iam.Role(
                self,
                "ObjectiveTaskRole",
                assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            )
            if internal_objective
            else None
        )
        if objective_role is not None:
            objective_role.add_to_policy(
                iam.PolicyStatement(
                    sid="ObjectiveArtifactPrefixReadWrite",
                    actions=[
                        "s3:GetObject",
                        "s3:GetObjectVersion",
                        "s3:PutObject",
                        "s3:AbortMultipartUpload",
                    ],
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
            objective_role.add_to_policy(
                iam.PolicyStatement(
                    sid="ObjectiveListVersionsPrefix",
                    actions=["s3:ListBucketVersions"],
                    resources=[artifacts.bucket_arn],
                    conditions={"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}},
                )
            )
            objective_role.add_to_policy(
                iam.PolicyStatement(
                    sid="ObjectiveReadBucketVersioning",
                    actions=["s3:GetBucketVersioning"],
                    resources=[artifacts.bucket_arn],
                )
            )
            # GetObject authorizes HeadObject; the explicit version action is
            # required because the immutable store always supplies VersionId.
            artifact_key.grant_encrypt_decrypt(objective_role)

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

        objective_service: ecs_patterns.ApplicationLoadBalancedFargateService | None = None
        if internal_objective:
            private_dns_name, private_zone_name = self._private_dns_contract()
            private_zone = route53.PrivateHostedZone(
                self,
                "ObjectivePrivateHostedZone",
                zone_name=private_zone_name,
                vpc=vpc,
                comment="Private TLS hostname for the objective worker",
            )
            # Certificate-only mode is the private, in-stack objective
            # deployment. It is always TLS; there is no HTTP fallback.
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

            if objective_role is None:  # pragma: no cover - guarded above
                raise AssertionError("internal objective role was not created")
            objective_task_definition = ecs.FargateTaskDefinition(
                self,
                "ObjectiveTaskDefinition",
                cpu=2048,
                memory_limit_mib=4096,
                task_role=objective_role,
                execution_role=objective_execution_role,
            )
            objective_task_definition.add_container(
                "Objective",
                image=self._image(backend_repository, "backend_image_digest"),
                logging=ecs.LogDrivers.aws_logs(
                    stream_prefix="objective", log_group=objective_logs
                ),
                environment={
                    "APP_MODE": "aws",
                    "SERVICE_ROLE": "objective",
                    "AWS_REGION": self.region or "us-east-1",
                    "TARGET_MODEL": target_model,
                    "OBJECTIVE_MODEL_CHECKPOINT_DIR": self._text(
                        "objective_model_checkpoint_dir", "/opt/models/functiongemma"
                    ),
                    "OBJECTIVE_MODEL_REVISION": objective_checkpoint_revision,
                    "OBJECTIVE_MODEL_SHA256": self._text("objective_model_sha256"),
                    "OBJECTIVE_BASE_MODEL_URI": objective_checkpoint_uri,
                    "OBJECTIVE_BASE_MODEL_SHA256": objective_checkpoint_sha256,
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
            objective_service = ecs_patterns.ApplicationLoadBalancedFargateService(
                self,
                "ObjectiveService",
                cluster=cluster,
                task_definition=objective_task_definition,
                desired_count=int(self._context("objective_desired_count", 1)),
                public_load_balancer=False,
                assign_public_ip=False,
                health_check_grace_period=Duration.seconds(60),
                listener_port=443,
                open_listener=False,
                protocol=elbv2.ApplicationProtocol.HTTPS,
                certificate=acm.Certificate.from_certificate_arn(
                    self, "ObjectiveCertificate", certificate_arn
                ),
            )
            objective_service.target_group.configure_health_check(path="/health", port="8080")
            objective_service.service.connections.allow_from(
                objective_service.load_balancer,
                ec2.Port.tcp(8080),
                "Allow the internal load balancer to reach the objective worker",
            )
            route53.ARecord(
                self,
                "ObjectivePrivateAlias",
                zone=private_zone,
                record_name=private_dns_name,
                target=route53.RecordTarget.from_alias(
                    route53_targets.LoadBalancerTarget(objective_service.load_balancer)
                ),
            )
            # The coordinator uses the certificate SAN, which is also bound to
            # the private alias above.  Never use the ALB-generated DNS name:
            # it is not covered by the configured certificate.
            objective_worker_url = "https://" + private_dns_name
        coordinator_security_group = ec2.SecurityGroup(
            self,
            "CoordinatorSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="Coordinator egress to private objective worker only",
        )
        if objective_service is not None:
            objective_service.load_balancer.connections.allow_from(
                coordinator_security_group,
                ec2.Port.tcp(443),
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
                # Matches config_from_environment's live controller contract.
                "OBJECTIVE_WORKER_AUTH_TOKEN": ecs.Secret.from_secrets_manager(objective_secret),
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        requested_coordinator_count = int(self._context("desired_count", 1))
        if requested_coordinator_count < 0:
            raise ValueError("desired_count must be zero or greater")
        bootstrap_without_tasks = requested_coordinator_count == 0
        coordinator = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "CoordinatorService",
            cluster=cluster,
            task_definition=task_definition,
            # The pattern rejects zero even though ECS itself supports a
            # zero-desired-count service. Build its constructs with one, then
            # override the CloudFormation service so bootstrap deploys cannot
            # start a coordinator task before immutable image digests exist.
            desired_count=max(requested_coordinator_count, 1),
            public_load_balancer=True,
            assign_public_ip=False,
            security_groups=[coordinator_security_group],
            health_check_grace_period=Duration.seconds(60),
            listener_port=443,
            open_listener=False,
            protocol=elbv2.ApplicationProtocol.HTTPS,
            certificate=acm.Certificate.from_certificate_arn(
                self, "CoordinatorCertificate", coordinator_certificate_arn
            ),
            domain_name=coordinator_dns_name,
            domain_zone=route53.HostedZone.from_hosted_zone_attributes(
                self,
                "CoordinatorPublicHostedZone",
                hosted_zone_id=coordinator_zone_id,
                zone_name=coordinator_zone_name,
            ),
        )
        if bootstrap_without_tasks:
            coordinator.service.node.default_child.add_override(
                "Properties.DesiredCount", 0
            )
        coordinator.target_group.configure_health_check(path="/health", port="8080")
        for cidr in coordinator_ingress_cidrs:
            coordinator.load_balancer.connections.allow_from(
                ec2.Peer.ipv4(cidr),
                ec2.Port.tcp(443),
                "Allow the explicitly configured demo operator CIDR",
            )
        if objective_service is not None:
            coordinator.service.connections.allow_to(
                objective_service.load_balancer,
                ec2.Port.tcp(443),
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
        CfnOutput(self, "ServiceName", value=coordinator.service.service_name)
        CfnOutput(
            self,
            "CoordinatorUrl",
            value="https://" + coordinator_dns_name,
            description=(
                "HTTPS coordinator API, reachable only from the configured IPv4 CIDR allowlist"
            ),
        )
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
            "LIVE_APPROVAL_TTL_SECONDS": self._text("approval_ttl_seconds", "86400"),
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
