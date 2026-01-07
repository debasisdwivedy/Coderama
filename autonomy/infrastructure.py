"""
Infrastructure & Scalability Module

Provides infrastructure management:
- Infrastructure-as-code generation (Terraform/Kubernetes)
- Automatic environment provisioning
- Blue-green deployment with health checks
- Cost optimization
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"


class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class ResourceType(Enum):
    """Infrastructure resource types"""
    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    CACHE = "cache"
    QUEUE = "queue"
    CDN = "cdn"


@dataclass
class InfrastructureResource:
    """Represents an infrastructure resource"""
    id: str
    name: str
    resource_type: ResourceType
    provider: CloudProvider
    configuration: Dict[str, Any]
    cost_per_hour: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Environment:
    """Represents a deployment environment"""
    id: str
    name: str
    env_type: EnvironmentType
    resources: List[InfrastructureResource]
    configuration: Dict[str, Any]
    status: str  # active, provisioning, terminated
    created_at: datetime
    last_deployed: Optional[datetime] = None


@dataclass
class Deployment:
    """Represents a deployment"""
    id: str
    environment_id: str
    version: str
    strategy: DeploymentStrategy
    status: str  # pending, in_progress, completed, failed, rolled_back
    started_at: datetime
    completed_at: Optional[datetime] = None
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_version: Optional[str] = None


class InfrastructureGenerator:
    """
    Generates infrastructure-as-code for different providers.

    Supports Terraform, Kubernetes manifests, and Docker configurations.
    """

    TERRAFORM_TEMPLATES = {
        "aws_instance": '''
resource "aws_instance" "{name}" {{
  ami           = "{ami}"
  instance_type = "{instance_type}"

  tags = {{
    Name        = "{name}"
    Environment = "{environment}"
    {additional_tags}
  }}

  {additional_config}
}}
''',
        "aws_rds": '''
resource "aws_db_instance" "{name}" {{
  identifier        = "{name}"
  engine            = "{engine}"
  engine_version    = "{engine_version}"
  instance_class    = "{instance_class}"
  allocated_storage = {storage}

  username = var.db_username
  password = var.db_password

  skip_final_snapshot = {skip_snapshot}

  tags = {{
    Environment = "{environment}"
  }}
}}
''',
        "aws_s3": '''
resource "aws_s3_bucket" "{name}" {{
  bucket = "{bucket_name}"

  tags = {{
    Environment = "{environment}"
  }}
}}

resource "aws_s3_bucket_versioning" "{name}_versioning" {{
  bucket = aws_s3_bucket.{name}.id
  versioning_configuration {{
    status = "{versioning}"
  }}
}}
'''
    }

    KUBERNETES_TEMPLATES = {
        "deployment": '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app}
  template:
    metadata:
      labels:
        app: {app}
    spec:
      containers:
      - name: {container_name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            memory: "{memory_request}"
            cpu: "{cpu_request}"
          limits:
            memory: "{memory_limit}"
            cpu: "{cpu_limit}"
        {env_vars}
        {health_checks}
''',
        "service": '''
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {app}
  ports:
  - port: {port}
    targetPort: {target_port}
  type: {service_type}
''',
        "ingress": '''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  annotations:
    {annotations}
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: {path}
        pathType: Prefix
        backend:
          service:
            name: {service_name}
            port:
              number: {port}
'''
    }

    DOCKER_TEMPLATES = {
        "dockerfile": '''FROM {base_image}

WORKDIR /app

{copy_commands}

{install_commands}

EXPOSE {port}

{env_vars}

CMD {cmd}
''',
        "compose": '''version: '3.8'

services:
  {service_name}:
    build: {build_context}
    ports:
      - "{host_port}:{container_port}"
    environment:
      {environment}
    {volumes}
    {depends_on}
'''
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files: List[str] = []

        logger.info(f"InfrastructureGenerator initialized: {output_dir}")

    def generate_terraform(
        self,
        resources: List[InfrastructureResource],
        environment: str
    ) -> Dict[str, str]:
        """Generate Terraform configuration files"""
        files = {}

        # Main configuration
        main_tf = ["# Generated Terraform configuration\n"]
        main_tf.append(f"# Environment: {environment}\n\n")

        # Provider configuration
        providers = set(r.provider for r in resources)
        for provider in providers:
            if provider == CloudProvider.AWS:
                main_tf.append('''provider "aws" {
  region = var.aws_region
}

''')

        # Generate resource configurations
        for resource in resources:
            if resource.provider == CloudProvider.AWS:
                config = self._generate_aws_resource(resource, environment)
                main_tf.append(config)
                main_tf.append("\n")

        files["main.tf"] = "".join(main_tf)

        # Variables file
        variables_tf = '''variable "aws_region" {
  default = "us-west-2"
}

variable "db_username" {
  description = "Database username"
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  sensitive   = true
}
'''
        files["variables.tf"] = variables_tf

        # Outputs file
        outputs = self._generate_terraform_outputs(resources)
        files["outputs.tf"] = outputs

        # Save files
        tf_dir = self.output_dir / "terraform"
        tf_dir.mkdir(exist_ok=True)

        for filename, content in files.items():
            file_path = tf_dir / filename
            file_path.write_text(content)
            self.generated_files.append(str(file_path))

        logger.info(f"Generated {len(files)} Terraform files")
        return files

    def _generate_aws_resource(
        self,
        resource: InfrastructureResource,
        environment: str
    ) -> str:
        """Generate AWS resource Terraform configuration"""
        config = resource.configuration

        if resource.resource_type == ResourceType.COMPUTE:
            return self.TERRAFORM_TEMPLATES["aws_instance"].format(
                name=resource.name,
                ami=config.get("ami", "ami-0c55b159cbfafe1f0"),
                instance_type=config.get("instance_type", "t3.micro"),
                environment=environment,
                additional_tags="",
                additional_config=""
            )

        elif resource.resource_type == ResourceType.DATABASE:
            return self.TERRAFORM_TEMPLATES["aws_rds"].format(
                name=resource.name,
                engine=config.get("engine", "postgres"),
                engine_version=config.get("engine_version", "14"),
                instance_class=config.get("instance_class", "db.t3.micro"),
                storage=config.get("storage", 20),
                skip_snapshot="true" if environment != "production" else "false",
                environment=environment
            )

        elif resource.resource_type == ResourceType.STORAGE:
            return self.TERRAFORM_TEMPLATES["aws_s3"].format(
                name=resource.name,
                bucket_name=f"{resource.name}-{environment}",
                environment=environment,
                versioning="Enabled" if environment == "production" else "Suspended"
            )

        return f"# Unsupported resource type: {resource.resource_type}"

    def _generate_terraform_outputs(
        self,
        resources: List[InfrastructureResource]
    ) -> str:
        """Generate Terraform outputs file"""
        outputs = ["# Generated outputs\n\n"]

        for resource in resources:
            if resource.resource_type == ResourceType.COMPUTE:
                outputs.append(f'''output "{resource.name}_ip" {{
  value = aws_instance.{resource.name}.public_ip
}}

''')
            elif resource.resource_type == ResourceType.DATABASE:
                outputs.append(f'''output "{resource.name}_endpoint" {{
  value = aws_db_instance.{resource.name}.endpoint
}}

''')

        return "".join(outputs)

    def generate_kubernetes(
        self,
        app_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Kubernetes manifests"""
        files = {}

        # Deployment
        deployment = self.KUBERNETES_TEMPLATES["deployment"].format(
            name=f"{app_name}-deployment",
            app=app_name,
            replicas=config.get("replicas", 2),
            container_name=app_name,
            image=config.get("image", f"{app_name}:latest"),
            port=config.get("port", 8080),
            memory_request=config.get("memory_request", "128Mi"),
            cpu_request=config.get("cpu_request", "100m"),
            memory_limit=config.get("memory_limit", "256Mi"),
            cpu_limit=config.get("cpu_limit", "200m"),
            env_vars=self._format_k8s_env_vars(config.get("env_vars", {})),
            health_checks=self._format_k8s_health_checks(config.get("health_check", {}))
        )
        files["deployment.yaml"] = deployment

        # Service
        service = self.KUBERNETES_TEMPLATES["service"].format(
            name=f"{app_name}-service",
            app=app_name,
            port=config.get("service_port", 80),
            target_port=config.get("port", 8080),
            service_type=config.get("service_type", "ClusterIP")
        )
        files["service.yaml"] = service

        # Ingress (if enabled)
        if config.get("ingress", {}).get("enabled", False):
            ingress = self.KUBERNETES_TEMPLATES["ingress"].format(
                name=f"{app_name}-ingress",
                annotations=self._format_k8s_annotations(
                    config.get("ingress", {}).get("annotations", {})
                ),
                host=config.get("ingress", {}).get("host", f"{app_name}.example.com"),
                path=config.get("ingress", {}).get("path", "/"),
                service_name=f"{app_name}-service",
                port=config.get("service_port", 80)
            )
            files["ingress.yaml"] = ingress

        # Save files
        k8s_dir = self.output_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)

        for filename, content in files.items():
            file_path = k8s_dir / filename
            file_path.write_text(content)
            self.generated_files.append(str(file_path))

        logger.info(f"Generated {len(files)} Kubernetes manifests")
        return files

    def _format_k8s_env_vars(self, env_vars: Dict[str, str]) -> str:
        """Format environment variables for Kubernetes"""
        if not env_vars:
            return ""

        lines = ["env:"]
        for key, value in env_vars.items():
            lines.append(f"        - name: {key}")
            lines.append(f'          value: "{value}"')

        return "\n        ".join(lines)

    def _format_k8s_health_checks(self, health_check: Dict[str, Any]) -> str:
        """Format health checks for Kubernetes"""
        if not health_check:
            return ""

        path = health_check.get("path", "/health")
        port = health_check.get("port", 8080)

        return f'''livenessProbe:
          httpGet:
            path: {path}
            port: {port}
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {path}
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5'''

    def _format_k8s_annotations(self, annotations: Dict[str, str]) -> str:
        """Format annotations for Kubernetes"""
        if not annotations:
            return "kubernetes.io/ingress.class: nginx"

        lines = []
        for key, value in annotations.items():
            lines.append(f"{key}: {value}")

        return "\n    ".join(lines)

    def generate_docker(
        self,
        app_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate Docker configuration files"""
        files = {}

        # Dockerfile
        dockerfile = self.DOCKER_TEMPLATES["dockerfile"].format(
            base_image=config.get("base_image", "python:3.11-slim"),
            copy_commands=self._format_docker_copy(config.get("copy", [])),
            install_commands=self._format_docker_install(config.get("install", [])),
            port=config.get("port", 8080),
            env_vars=self._format_docker_env(config.get("env_vars", {})),
            cmd=json.dumps(config.get("cmd", ["python", "main.py"]))
        )
        files["Dockerfile"] = dockerfile

        # Docker Compose
        compose = self.DOCKER_TEMPLATES["compose"].format(
            service_name=app_name,
            build_context=config.get("build_context", "."),
            host_port=config.get("host_port", 8080),
            container_port=config.get("port", 8080),
            environment=self._format_compose_env(config.get("env_vars", {})),
            volumes=self._format_compose_volumes(config.get("volumes", [])),
            depends_on=self._format_compose_depends(config.get("depends_on", []))
        )
        files["docker-compose.yaml"] = compose

        # Save files
        docker_dir = self.output_dir / "docker"
        docker_dir.mkdir(exist_ok=True)

        for filename, content in files.items():
            file_path = docker_dir / filename
            file_path.write_text(content)
            self.generated_files.append(str(file_path))

        logger.info(f"Generated {len(files)} Docker files")
        return files

    def _format_docker_copy(self, copy_commands: List[Dict[str, str]]) -> str:
        """Format COPY commands for Dockerfile"""
        if not copy_commands:
            return "COPY . ."

        lines = []
        for cmd in copy_commands:
            lines.append(f"COPY {cmd['src']} {cmd['dest']}")

        return "\n".join(lines)

    def _format_docker_install(self, install_commands: List[str]) -> str:
        """Format install commands for Dockerfile"""
        if not install_commands:
            return "RUN pip install -r requirements.txt"

        lines = [f"RUN {cmd}" for cmd in install_commands]
        return "\n".join(lines)

    def _format_docker_env(self, env_vars: Dict[str, str]) -> str:
        """Format environment variables for Dockerfile"""
        if not env_vars:
            return ""

        lines = []
        for key, value in env_vars.items():
            lines.append(f"ENV {key}={value}")

        return "\n".join(lines)

    def _format_compose_env(self, env_vars: Dict[str, str]) -> str:
        """Format environment for docker-compose"""
        if not env_vars:
            return "- ENV=development"

        lines = []
        for key, value in env_vars.items():
            lines.append(f"- {key}={value}")

        return "\n      ".join(lines)

    def _format_compose_volumes(self, volumes: List[str]) -> str:
        """Format volumes for docker-compose"""
        if not volumes:
            return ""

        lines = ["volumes:"]
        for vol in volumes:
            lines.append(f"      - {vol}")

        return "\n    ".join(lines)

    def _format_compose_depends(self, depends_on: List[str]) -> str:
        """Format depends_on for docker-compose"""
        if not depends_on:
            return ""

        lines = ["depends_on:"]
        for dep in depends_on:
            lines.append(f"      - {dep}")

        return "\n    ".join(lines)


class EnvironmentProvisioner:
    """
    Automatically provisions development, staging, and production environments.
    """

    def __init__(self, infrastructure_generator: InfrastructureGenerator):
        self.generator = infrastructure_generator
        self.environments: Dict[str, Environment] = {}

        logger.info("EnvironmentProvisioner initialized")

    def create_environment(
        self,
        name: str,
        env_type: EnvironmentType,
        config: Dict[str, Any]
    ) -> Environment:
        """Create a new environment"""
        env_id = f"env_{uuid.uuid4().hex[:12]}"

        # Generate resources based on environment type
        resources = self._generate_resources(env_type, config)

        environment = Environment(
            id=env_id,
            name=name,
            env_type=env_type,
            resources=resources,
            configuration=config,
            status="provisioning",
            created_at=datetime.now()
        )

        self.environments[env_id] = environment

        # Generate infrastructure code
        self._generate_infrastructure(environment)

        environment.status = "active"
        logger.info(f"Created environment: {name} ({env_type.value})")

        return environment

    def _generate_resources(
        self,
        env_type: EnvironmentType,
        config: Dict[str, Any]
    ) -> List[InfrastructureResource]:
        """Generate resources based on environment type"""
        resources = []

        # Base resources for all environments
        compute_size = {
            EnvironmentType.DEVELOPMENT: "t3.micro",
            EnvironmentType.STAGING: "t3.small",
            EnvironmentType.PRODUCTION: "t3.medium",
            EnvironmentType.TESTING: "t3.micro"
        }

        # Compute resource
        resources.append(InfrastructureResource(
            id=f"res_{uuid.uuid4().hex[:8]}",
            name=f"app-server-{env_type.value}",
            resource_type=ResourceType.COMPUTE,
            provider=CloudProvider.AWS,
            configuration={
                "instance_type": compute_size.get(env_type, "t3.micro"),
                "ami": config.get("ami", "ami-0c55b159cbfafe1f0")
            },
            cost_per_hour=0.01 if env_type != EnvironmentType.PRODUCTION else 0.05,
            tags={"environment": env_type.value}
        ))

        # Database resource
        db_size = {
            EnvironmentType.DEVELOPMENT: "db.t3.micro",
            EnvironmentType.STAGING: "db.t3.small",
            EnvironmentType.PRODUCTION: "db.t3.medium",
            EnvironmentType.TESTING: "db.t3.micro"
        }

        resources.append(InfrastructureResource(
            id=f"res_{uuid.uuid4().hex[:8]}",
            name=f"database-{env_type.value}",
            resource_type=ResourceType.DATABASE,
            provider=CloudProvider.AWS,
            configuration={
                "instance_class": db_size.get(env_type, "db.t3.micro"),
                "engine": config.get("db_engine", "postgres"),
                "storage": 20 if env_type != EnvironmentType.PRODUCTION else 100
            },
            cost_per_hour=0.02 if env_type != EnvironmentType.PRODUCTION else 0.10,
            tags={"environment": env_type.value}
        ))

        # Storage resource
        resources.append(InfrastructureResource(
            id=f"res_{uuid.uuid4().hex[:8]}",
            name=f"storage-{env_type.value}",
            resource_type=ResourceType.STORAGE,
            provider=CloudProvider.AWS,
            configuration={
                "versioning": env_type == EnvironmentType.PRODUCTION
            },
            cost_per_hour=0.001,
            tags={"environment": env_type.value}
        ))

        return resources

    def _generate_infrastructure(self, environment: Environment) -> None:
        """Generate infrastructure code for environment"""
        self.generator.generate_terraform(
            environment.resources,
            environment.env_type.value
        )

    def get_environment(self, env_id: str) -> Optional[Environment]:
        """Get an environment by ID"""
        return self.environments.get(env_id)

    def list_environments(self) -> List[Environment]:
        """List all environments"""
        return list(self.environments.values())

    def terminate_environment(self, env_id: str) -> bool:
        """Terminate an environment"""
        if env_id in self.environments:
            self.environments[env_id].status = "terminated"
            logger.info(f"Terminated environment: {env_id}")
            return True
        return False


class DeploymentManager:
    """
    Manages deployments with blue-green and canary strategies.
    """

    def __init__(self):
        self.deployments: Dict[str, Deployment] = {}
        self.active_versions: Dict[str, str] = {}  # env_id -> version

        logger.info("DeploymentManager initialized")

    def create_deployment(
        self,
        environment_id: str,
        version: str,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
        health_check_url: Optional[str] = None
    ) -> Deployment:
        """Create a new deployment"""
        deployment_id = f"deploy_{uuid.uuid4().hex[:12]}"

        # Get current version for rollback
        current_version = self.active_versions.get(environment_id)

        deployment = Deployment(
            id=deployment_id,
            environment_id=environment_id,
            version=version,
            strategy=strategy,
            status="pending",
            started_at=datetime.now(),
            rollback_version=current_version
        )

        self.deployments[deployment_id] = deployment
        logger.info(f"Created deployment: {deployment_id} (strategy: {strategy.value})")

        return deployment

    async def execute_deployment(
        self,
        deployment: Deployment,
        health_check: Optional[Callable[[], bool]] = None
    ) -> Deployment:
        """Execute a deployment"""
        deployment.status = "in_progress"

        try:
            if deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._blue_green_deploy(deployment, health_check)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                success = await self._canary_deploy(deployment, health_check)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                success = await self._rolling_deploy(deployment, health_check)
            else:
                success = await self._recreate_deploy(deployment, health_check)

            if success:
                deployment.status = "completed"
                deployment.completed_at = datetime.now()
                self.active_versions[deployment.environment_id] = deployment.version
            else:
                deployment.status = "failed"
                # Attempt rollback
                if deployment.rollback_version:
                    await self._rollback(deployment)

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment.status = "failed"

        return deployment

    async def _blue_green_deploy(
        self,
        deployment: Deployment,
        health_check: Optional[Callable[[], bool]]
    ) -> bool:
        """Execute blue-green deployment"""
        logger.info(f"Starting blue-green deployment for {deployment.version}")

        # Simulate deployment steps
        steps = [
            "Provisioning green environment",
            "Deploying to green environment",
            "Running health checks on green",
            "Switching traffic to green",
            "Terminating blue environment"
        ]

        for step in steps:
            deployment.health_checks.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })

            # Simulate health check
            if health_check and "health" in step.lower():
                if not health_check():
                    deployment.health_checks[-1]["status"] = "failed"
                    return False

        return True

    async def _canary_deploy(
        self,
        deployment: Deployment,
        health_check: Optional[Callable[[], bool]]
    ) -> bool:
        """Execute canary deployment"""
        logger.info(f"Starting canary deployment for {deployment.version}")

        # Canary rollout percentages
        percentages = [5, 25, 50, 100]

        for pct in percentages:
            deployment.health_checks.append({
                "step": f"Routing {pct}% traffic to new version",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })

            if health_check and not health_check():
                deployment.health_checks[-1]["status"] = "failed"
                return False

        return True

    async def _rolling_deploy(
        self,
        deployment: Deployment,
        health_check: Optional[Callable[[], bool]]
    ) -> bool:
        """Execute rolling deployment"""
        logger.info(f"Starting rolling deployment for {deployment.version}")

        # Simulate rolling updates to instances
        instances = 3  # Simulated instance count

        for i in range(instances):
            deployment.health_checks.append({
                "step": f"Updating instance {i + 1}/{instances}",
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })

            if health_check and not health_check():
                deployment.health_checks[-1]["status"] = "failed"
                return False

        return True

    async def _recreate_deploy(
        self,
        deployment: Deployment,
        health_check: Optional[Callable[[], bool]]
    ) -> bool:
        """Execute recreate deployment (downtime)"""
        logger.info(f"Starting recreate deployment for {deployment.version}")

        steps = [
            "Stopping old version",
            "Deploying new version",
            "Starting new version"
        ]

        for step in steps:
            deployment.health_checks.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })

        return True

    async def _rollback(self, deployment: Deployment) -> None:
        """Rollback a failed deployment"""
        if not deployment.rollback_version:
            return

        logger.info(f"Rolling back to version {deployment.rollback_version}")
        deployment.status = "rolled_back"

        self.active_versions[deployment.environment_id] = deployment.rollback_version

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get a deployment by ID"""
        return self.deployments.get(deployment_id)

    def get_deployment_history(
        self,
        environment_id: str,
        limit: int = 10
    ) -> List[Deployment]:
        """Get deployment history for an environment"""
        deployments = [
            d for d in self.deployments.values()
            if d.environment_id == environment_id
        ]
        deployments.sort(key=lambda d: d.started_at, reverse=True)
        return deployments[:limit]


class CostOptimizer:
    """
    Optimizes infrastructure costs by right-sizing resources.
    """

    def __init__(self):
        self.recommendations: List[Dict[str, Any]] = []
        self.savings_history: List[Dict[str, Any]] = []

        logger.info("CostOptimizer initialized")

    def analyze_resources(
        self,
        resources: List[InfrastructureResource],
        usage_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze resource utilization and recommend optimizations"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_resources": len(resources),
            "total_hourly_cost": sum(r.cost_per_hour for r in resources),
            "recommendations": [],
            "potential_savings": 0.0
        }

        for resource in resources:
            resource_usage = usage_data.get(resource.id, {})

            # Check CPU utilization
            cpu_usage = resource_usage.get("cpu_avg", 50)
            memory_usage = resource_usage.get("memory_avg", 50)

            if resource.resource_type == ResourceType.COMPUTE:
                if cpu_usage < 20 and memory_usage < 30:
                    # Resource is underutilized
                    savings = resource.cost_per_hour * 0.3
                    analysis["recommendations"].append({
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "type": "downsize",
                        "reason": f"Low utilization (CPU: {cpu_usage}%, Memory: {memory_usage}%)",
                        "suggestion": "Consider downsizing to a smaller instance type",
                        "potential_savings_per_hour": savings
                    })
                    analysis["potential_savings"] += savings

                elif cpu_usage > 80 or memory_usage > 85:
                    # Resource may be under-provisioned
                    analysis["recommendations"].append({
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "type": "upsize",
                        "reason": f"High utilization (CPU: {cpu_usage}%, Memory: {memory_usage}%)",
                        "suggestion": "Consider upsizing to prevent performance issues",
                        "potential_savings_per_hour": 0
                    })

            elif resource.resource_type == ResourceType.DATABASE:
                if cpu_usage < 15:
                    savings = resource.cost_per_hour * 0.25
                    analysis["recommendations"].append({
                        "resource_id": resource.id,
                        "resource_name": resource.name,
                        "type": "downsize",
                        "reason": f"Database underutilized (CPU: {cpu_usage}%)",
                        "suggestion": "Consider a smaller instance class",
                        "potential_savings_per_hour": savings
                    })
                    analysis["potential_savings"] += savings

        self.recommendations.extend(analysis["recommendations"])

        # Calculate monthly savings
        analysis["potential_monthly_savings"] = analysis["potential_savings"] * 24 * 30

        return analysis

    def get_cost_report(
        self,
        resources: List[InfrastructureResource],
        period_hours: int = 720  # 30 days
    ) -> Dict[str, Any]:
        """Generate a cost report"""
        by_type = {}
        by_environment = {}

        total_cost = 0.0

        for resource in resources:
            resource_cost = resource.cost_per_hour * period_hours
            total_cost += resource_cost

            # Group by type
            type_name = resource.resource_type.value
            if type_name not in by_type:
                by_type[type_name] = 0.0
            by_type[type_name] += resource_cost

            # Group by environment
            env = resource.tags.get("environment", "unknown")
            if env not in by_environment:
                by_environment[env] = 0.0
            by_environment[env] += resource_cost

        return {
            "period_hours": period_hours,
            "total_cost": round(total_cost, 2),
            "by_resource_type": {k: round(v, 2) for k, v in by_type.items()},
            "by_environment": {k: round(v, 2) for k, v in by_environment.items()},
            "average_hourly_cost": round(total_cost / period_hours, 4),
            "resource_count": len(resources)
        }

    def suggest_reserved_instances(
        self,
        resources: List[InfrastructureResource],
        usage_history_months: int = 3
    ) -> List[Dict[str, Any]]:
        """Suggest reserved instances for cost savings"""
        suggestions = []

        for resource in resources:
            if resource.resource_type == ResourceType.COMPUTE:
                # Assume consistent usage for stable workloads
                monthly_cost = resource.cost_per_hour * 24 * 30

                # Reserved instances typically offer 30-60% savings
                reserved_1yr_savings = monthly_cost * 0.30 * 12
                reserved_3yr_savings = monthly_cost * 0.50 * 36

                suggestions.append({
                    "resource_id": resource.id,
                    "resource_name": resource.name,
                    "current_monthly_cost": round(monthly_cost, 2),
                    "reserved_1yr_monthly": round(monthly_cost * 0.70, 2),
                    "reserved_1yr_annual_savings": round(reserved_1yr_savings, 2),
                    "reserved_3yr_monthly": round(monthly_cost * 0.50, 2),
                    "reserved_3yr_total_savings": round(reserved_3yr_savings, 2)
                })

        return suggestions
