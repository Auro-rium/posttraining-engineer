"""Minimal production-oriented AWS foundation for the post-training backend."""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct


class PostTrainingStack(Stack):
    """Provision storage, IAM, ECR, and an ECS/Fargate runtime boundary."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        artifacts = s3.Bucket(
            self,
            "Artifacts",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        state = dynamodb.Table(
            self,
            "RunState",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
        )
        repository = ecr.Repository(
            self,
            "BackendRepository",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.RETAIN,
        )

        vpc = ec2.Vpc(self, "RuntimeVpc", max_azs=2, nat_gateways=1)
        cluster = ecs.Cluster(self, "RuntimeCluster", vpc=vpc, container_insights=True)
        logs_group = logs.LogGroup(
            self,
            "RuntimeLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )
        training_role = iam.Role(
            self,
            "SageMakerTrainingRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        training_role.add_to_policy(
            iam.PolicyStatement(actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"], resources=[artifacts.bucket_arn, f"{artifacts.bucket_arn}/*"])
        )
        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        artifacts.grant_read_write(task_role)
        state.grant_read_write_data(task_role)
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob", "sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob"],
                resources=["*"],
            )
        )
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
            logging=ecs.LogDrivers.aws_logs(stream_prefix="post-training", log_group=logs_group),
            image=ecs.ContainerImage.from_ecr_repository(
                repository, self.node.try_get_context("image_tag") or "latest"
            ),
            environment={
                "APP_MODE": "aws",
                "SERVICE_ROLE": "coordinator",
                "AWS_REGION": self.region or "us-east-1",
                "S3_ARTIFACT_BUCKET": artifacts.bucket_name,
                "DYNAMODB_TABLE_NAME": state.table_name,
                "SAGEMAKER_TRAINING_ROLE_ARN": training_role.role_arn,
                "SAGEMAKER_TRAINING_IMAGE_URI": self.node.try_get_context("training_image_uri") or "required-at-deploy",
                "SAGEMAKER_EVALUATION_IMAGE_URI": self.node.try_get_context("evaluation_image_uri") or "required-at-deploy",
            },
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )
        service = ecs.FargateService(
            self,
            "RuntimeService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=int(self.node.try_get_context("desired_count") or 1),
            assign_public_ip=False,
            health_check_grace_period=Duration.seconds(60),
        )

        CfnOutput(self, "ArtifactBucketName", value=artifacts.bucket_name)
        CfnOutput(self, "StateTableName", value=state.table_name)
        CfnOutput(self, "RepositoryUri", value=repository.repository_uri)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=service.service_name)
