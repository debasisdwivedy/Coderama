"""
Autonomous Agent Orchestrator

Main orchestrator that coordinates all autonomy modules into a cohesive
self-healing, self-improving, production-grade system.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .self_healing import SelfHealingEngine
from .observability import ObservabilityDashboard
from .security import SecurityScanner, LicenseChecker, SecretsManager, AccessControl, ApprovalWorkflow
from .decision_engine import DecisionEngine
from .learning import ContinuousLearning
from .testing import TestGenerator, ChaosEngineer, PerformanceTester
from .context import VectorKnowledgeBase, DocumentationGenerator, CrossProjectLearning, ConflictResolver
from .infrastructure import InfrastructureGenerator, EnvironmentProvisioner, DeploymentManager, CostOptimizer
from .business_intel import BusinessIntelligence
from .integrations import JiraIntegration, SlackIntegration, GitWorkflow, ExternalAPITester

logger = logging.getLogger(__name__)


@dataclass
class AutonomyConfig:
    """Configuration for the autonomy system"""
    workspace_dir: str
    storage_dir: str
    hourly_rate: float = 100.0
    auto_heal: bool = True
    auto_deploy: bool = False
    require_approval_for_production: bool = True
    chaos_testing_enabled: bool = False
    learning_enabled: bool = True
    max_concurrent_agents: int = 5


@dataclass
class SystemState:
    """Current state of the autonomous system"""
    initialized: bool = False
    running: bool = False
    health_status: str = "unknown"
    active_deployments: int = 0
    pending_approvals: int = 0
    last_health_check: Optional[datetime] = None
    last_learning_update: Optional[datetime] = None
    errors_last_hour: int = 0


class AutonomousOrchestrator:
    """
    Main orchestrator for the autonomous self-healing deployment system.

    Coordinates all modules and provides a unified interface for:
    - Self-healing and recovery
    - Observability and monitoring
    - Security and governance
    - Decision making
    - Continuous learning
    - Testing automation
    - Infrastructure management
    - Business intelligence
    - External integrations
    """

    def __init__(self, config: AutonomyConfig):
        self.config = config
        self.state = SystemState()

        # Initialize storage paths
        self.workspace = Path(config.workspace_dir)
        self.storage = Path(config.storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)

        # Initialize all modules
        self._init_modules()

        logger.info("AutonomousOrchestrator initialized")

    def _init_modules(self) -> None:
        """Initialize all autonomy modules"""
        # Core modules
        self.self_healing = SelfHealingEngine(
            str(self.workspace),
            auto_heal=self.config.auto_heal
        )

        self.observability = ObservabilityDashboard(
            str(self.storage / "observability")
        )

        # Security
        self.security_scanner = SecurityScanner(str(self.workspace))
        self.license_checker = LicenseChecker(str(self.workspace))
        self.secrets_manager = SecretsManager(str(self.storage / "secrets"))
        self.access_control = AccessControl()
        self.approval_workflow = ApprovalWorkflow(
            self.access_control,
            required_approvals=2 if self.config.require_approval_for_production else 1
        )

        # Decision making
        self.decision_engine = DecisionEngine()

        # Learning
        self.learning = ContinuousLearning(
            str(self.storage / "learning")
        ) if self.config.learning_enabled else None

        # Testing
        self.test_generator = TestGenerator(str(self.workspace))
        self.chaos_engineer = ChaosEngineer(
            str(self.workspace),
            safe_mode=not self.config.chaos_testing_enabled
        )
        self.performance_tester = PerformanceTester(str(self.workspace))

        # Context and knowledge
        self.knowledge_base = VectorKnowledgeBase(
            str(self.storage / "knowledge")
        )
        self.doc_generator = DocumentationGenerator(
            str(self.workspace),
            self.knowledge_base
        )
        self.cross_project = CrossProjectLearning(self.knowledge_base)
        self.conflict_resolver = ConflictResolver()

        # Infrastructure
        self.infra_generator = InfrastructureGenerator(
            str(self.storage / "infrastructure")
        )
        self.env_provisioner = EnvironmentProvisioner(self.infra_generator)
        self.deployment_manager = DeploymentManager()
        self.cost_optimizer = CostOptimizer()

        # Business intelligence
        self.business_intel = BusinessIntelligence(self.config.hourly_rate)

        # Integrations
        self.jira = JiraIntegration()
        self.slack = SlackIntegration()
        self.git = GitWorkflow(str(self.workspace))
        self.api_tester = ExternalAPITester()

        self.state.initialized = True

    async def start(self) -> None:
        """Start the autonomous system"""
        if self.state.running:
            logger.warning("System already running")
            return

        logger.info("Starting autonomous system...")

        # Create initial checkpoint
        self.self_healing.create_deployment_checkpoint(
            "System startup checkpoint"
        )

        # Run initial health check
        await self._health_check()

        # Start background tasks
        self._start_background_tasks()

        self.state.running = True
        logger.info("Autonomous system started")

    async def stop(self) -> None:
        """Stop the autonomous system"""
        if not self.state.running:
            return

        logger.info("Stopping autonomous system...")

        # Create final checkpoint
        self.self_healing.create_deployment_checkpoint(
            "System shutdown checkpoint"
        )

        self.state.running = False
        logger.info("Autonomous system stopped")

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks"""
        # In a real implementation, these would be async tasks
        pass

    async def _health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        # Check self-healing
        healing_status = self.self_healing.get_health_status()
        health["components"]["self_healing"] = {
            "status": "healthy" if healing_status.get("healing_statistics", {}).get("success_rate", 0) > 0.7 else "degraded",
            "details": healing_status
        }

        # Check observability
        health["components"]["observability"] = {
            "status": "healthy",
            "details": self.observability.get_system_overview()
        }

        # Check security
        security_scan = self.security_scanner.scan_directory()
        critical_findings = security_scan.get("by_severity", {}).get("critical", 0)
        health["components"]["security"] = {
            "status": "healthy" if critical_findings == 0 else "critical" if critical_findings > 5 else "warning",
            "details": {"critical_findings": critical_findings}
        }

        # Overall status
        statuses = [c["status"] for c in health["components"].values()]
        if "critical" in statuses:
            health["overall"] = "critical"
        elif "degraded" in statuses or "warning" in statuses:
            health["overall"] = "degraded"
        else:
            health["overall"] = "healthy"

        self.state.health_status = health["overall"]
        self.state.last_health_check = datetime.now()

        return health

    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle an error with self-healing"""
        # Log to audit trail
        self.observability.audit_trail.log(
            actor=component,
            action="error",
            resource="system",
            resource_type="error",
            details={"error": str(error), "context": context},
            outcome="failure"
        )

        # Attempt self-healing
        healing_result = await self.self_healing.handle_failure(
            error,
            component,
            context
        )

        # Notify via Slack if critical
        if not healing_result.get("healed"):
            await self.slack.send_notification(
                channel="#alerts",
                title=f"Error in {component}",
                message=str(error),
                notification_type="error"
            )

        return healing_result

    async def make_decision(
        self,
        decision_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an autonomous decision"""
        # Start reasoning chain
        chain_id = self.observability.reasoning_tracker.start_chain(
            "decision_engine",
            f"Making {decision_type} decision"
        )

        decision = None

        if decision_type == "architecture":
            decision = self.decision_engine.make_architecture_decision(context)
        elif decision_type == "database":
            decision = self.decision_engine.make_database_decision(context)
        else:
            # Generic decision
            decision = {"error": f"Unknown decision type: {decision_type}"}

        # Log decision
        self.observability.reasoning_tracker.add_step(
            "decision_engine",
            reasoning_type="decision",
            description=f"Made {decision_type} decision",
            inputs=context,
            outputs=decision,
            confidence=decision.get("score", 0.5) if decision else 0
        )

        self.observability.reasoning_tracker.end_chain("decision_engine")

        # Log to audit trail
        self.observability.audit_trail.log(
            actor="decision_engine",
            action=f"decision_{decision_type}",
            resource=decision.get("recommendation", "unknown"),
            resource_type="decision",
            details=decision,
            outcome="success",
            reasoning=decision.get("reasoning")
        )

        return decision

    async def deploy(
        self,
        environment: str,
        version: str,
        require_approval: bool = True
    ) -> Dict[str, Any]:
        """Deploy to an environment"""
        # Create deployment checkpoint
        self.self_healing.create_deployment_checkpoint(
            f"Pre-deployment: {version} to {environment}"
        )

        # Security scan
        scan_result = self.security_scanner.scan_directory()
        if scan_result.get("by_severity", {}).get("critical", 0) > 0:
            return {
                "status": "blocked",
                "reason": "Critical security vulnerabilities found",
                "details": scan_result
            }

        # Request approval if needed
        if require_approval and environment == "production":
            approval = await self.slack.request_approval(
                channel="#deployments",
                title=f"Deploy {version} to {environment}",
                description=f"Requesting approval to deploy version {version} to {environment}",
                approvers=["@devops-team"]
            )

            return {
                "status": "pending_approval",
                "approval_id": approval.id,
                "message": "Awaiting approval"
            }

        # Execute deployment
        # (In production, this would actually deploy)
        await self.slack.send_deployment_notification(
            channel="#deployments",
            environment=environment,
            version=version,
            status="success",
            details={"deployed_by": "autonomous-agent"}
        )

        # Record in business intelligence
        self.business_intel.roi_calculator.record_task(
            "deployment",
            0.5,  # 30 minutes in hours
            quality_score=1.0
        )

        return {
            "status": "success",
            "environment": environment,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }

    async def run_tests(
        self,
        test_type: str = "all"
    ) -> Dict[str, Any]:
        """Run automated tests"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": test_type,
            "results": {}
        }

        if test_type in ["all", "unit"]:
            # Generate and run unit tests
            pass

        if test_type in ["all", "integration"]:
            # Run integration tests
            api_results = await self.api_tester.run_test_suite()
            results["results"]["api"] = api_results

        if test_type in ["all", "chaos"] and self.config.chaos_testing_enabled:
            # Run chaos tests
            templates = self.chaos_engineer.get_experiment_templates()
            results["results"]["chaos_templates"] = len(templates)

        if test_type in ["all", "performance"]:
            # Run performance tests (simplified)
            results["results"]["performance"] = {
                "baselines": len(self.performance_tester.baselines)
            }

        # Record in business intelligence
        self.business_intel.roi_calculator.record_task(
            "testing",
            1.0,
            quality_score=0.9
        )

        return results

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_state": {
                "initialized": self.state.initialized,
                "running": self.state.running,
                "health": self.state.health_status,
                "last_health_check": self.state.last_health_check.isoformat() if self.state.last_health_check else None
            },
            "observability": self.observability.get_system_overview(),
            "self_healing": self.self_healing.get_health_status(),
            "security": {
                "scan_history": self.security_scanner.scan_history[-5:],
                "license_status": self.license_checker.check_all_dependencies()
            },
            "business_intelligence": self.business_intel.get_executive_summary(),
            "learning": self.learning.get_learning_summary() if self.learning else None,
            "infrastructure": {
                "environments": [
                    {
                        "id": e.id,
                        "name": e.name,
                        "type": e.env_type.value,
                        "status": e.status
                    }
                    for e in self.env_provisioner.list_environments()
                ]
            },
            "integrations": {
                "jira": self.jira.get_sync_status(),
                "slack": {"notifications_sent": len(self.slack.notifications)},
                "git": {"operations": len(self.git.operations)},
                "api_health": self.api_tester.get_health_status()
            }
        }

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get system-wide recommendations"""
        recommendations = []

        # Security recommendations
        scan = self.security_scanner.scan_directory()
        if scan.get("total_findings", 0) > 0:
            recommendations.append({
                "category": "security",
                "priority": "high",
                "title": "Address security findings",
                "description": f"Found {scan['total_findings']} security issues that need attention"
            })

        # Cost recommendations
        # (Would analyze actual resource usage)
        recommendations.append({
            "category": "cost",
            "priority": "medium",
            "title": "Review resource utilization",
            "description": "Consider right-sizing resources based on usage patterns"
        })

        # Learning recommendations
        if self.learning:
            insights = self.learning.get_recommendations({})
            if insights.get("patterns"):
                recommendations.append({
                    "category": "best_practices",
                    "priority": "low",
                    "title": "Apply learned patterns",
                    "description": f"Found {len(insights['patterns'])} successful patterns to apply"
                })

        return recommendations

    async def conduct_retrospective(
        self,
        project_name: str
    ) -> Dict[str, Any]:
        """Conduct a project retrospective"""
        if not self.learning:
            return {"error": "Learning module not enabled"}

        # Gather metrics
        metrics = {
            "roi": self.business_intel.roi_calculator.get_summary(),
            "quality": self.business_intel.quality_metrics.get_current_status(),
            "security": self.security_scanner.scan_directory()
        }

        # Conduct post-mortem
        report = self.learning.conduct_project_retrospective(project_name, metrics)

        return {
            "report_id": report.id,
            "summary": report.summary,
            "lessons_learned": report.lessons_learned,
            "recommendations": report.recommendations,
            "action_items": report.action_items
        }

    def export_state(self) -> str:
        """Export system state for backup/restore"""
        state = {
            "exported_at": datetime.now().isoformat(),
            "config": {
                "workspace_dir": str(self.workspace),
                "storage_dir": str(self.storage),
                "auto_heal": self.config.auto_heal
            },
            "dashboard": self.get_dashboard_data(),
            "recommendations": self.get_recommendations()
        }

        return json.dumps(state, indent=2, default=str)


# Factory function
def create_autonomous_system(
    workspace_dir: str,
    storage_dir: Optional[str] = None,
    **kwargs
) -> AutonomousOrchestrator:
    """Create and configure an autonomous system"""
    if storage_dir is None:
        storage_dir = str(Path(workspace_dir) / ".coderama" / "autonomy")

    config = AutonomyConfig(
        workspace_dir=workspace_dir,
        storage_dir=storage_dir,
        **kwargs
    )

    return AutonomousOrchestrator(config)
