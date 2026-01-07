"""
Coderama Autonomy Module

Enterprise-grade autonomous self-healing deployment system with:
- Self-Healing & Recovery
- Observability & Monitoring
- Security & Governance
- Advanced Decision Making
- Continuous Learning Loop
- Testing Evolution
- Intelligent Context Sharing
- Infrastructure & Scalability
- Business Intelligence
- Integration Ecosystem
"""

from .self_healing import (
    SelfHealingEngine,
    CircuitBreaker,
    RollbackManager,
    SelfDebugAgent,
    DependencyResolver,
)
from .observability import (
    ObservabilityDashboard,
    AnomalyDetector,
    AuditTrail,
    AgentMetrics,
    ReasoningChainTracker,
)
from .security import (
    SecurityScanner,
    LicenseChecker,
    SecretsManager,
    AccessControl,
    ApprovalWorkflow,
)
from .decision_engine import (
    DecisionEngine,
    CostBenefitAnalyzer,
    PerformancePredictor,
    RiskAssessment,
    TradeoffNegotiator,
)
from .learning import (
    ContinuousLearning,
    PatternRecognition,
    PostMortemAnalyzer,
    SprintEstimator,
    ABTestEngine,
)
from .testing import (
    TestGenerator,
    ChaosEngineer,
    PerformanceTester,
    VisualRegressionTester,
)
from .context import (
    VectorKnowledgeBase,
    DocumentationGenerator,
    CrossProjectLearning,
    ConflictResolver,
)
from .infrastructure import (
    InfrastructureGenerator,
    EnvironmentProvisioner,
    DeploymentManager,
    CostOptimizer,
)
from .business_intel import (
    BusinessIntelligence,
    ROICalculator,
    VelocityTracker,
    QualityMetrics,
    PredictiveAnalytics,
)
from .integrations import (
    JiraIntegration,
    SlackIntegration,
    GitWorkflow,
    ExternalAPITester,
)

__version__ = "1.0.0"
__all__ = [
    # Self-Healing
    "SelfHealingEngine",
    "CircuitBreaker",
    "RollbackManager",
    "SelfDebugAgent",
    "DependencyResolver",
    # Observability
    "ObservabilityDashboard",
    "AnomalyDetector",
    "AuditTrail",
    "AgentMetrics",
    "ReasoningChainTracker",
    # Security
    "SecurityScanner",
    "LicenseChecker",
    "SecretsManager",
    "AccessControl",
    "ApprovalWorkflow",
    # Decision Engine
    "DecisionEngine",
    "CostBenefitAnalyzer",
    "PerformancePredictor",
    "RiskAssessment",
    "TradeoffNegotiator",
    # Learning
    "ContinuousLearning",
    "PatternRecognition",
    "PostMortemAnalyzer",
    "SprintEstimator",
    "ABTestEngine",
    # Testing
    "TestGenerator",
    "ChaosEngineer",
    "PerformanceTester",
    "VisualRegressionTester",
    # Context
    "VectorKnowledgeBase",
    "DocumentationGenerator",
    "CrossProjectLearning",
    "ConflictResolver",
    # Infrastructure
    "InfrastructureGenerator",
    "EnvironmentProvisioner",
    "DeploymentManager",
    "CostOptimizer",
    # Business Intelligence
    "BusinessIntelligence",
    "ROICalculator",
    "VelocityTracker",
    "QualityMetrics",
    "PredictiveAnalytics",
    # Integrations
    "JiraIntegration",
    "SlackIntegration",
    "GitWorkflow",
    "ExternalAPITester",
]
