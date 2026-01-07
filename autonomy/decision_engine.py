"""
Advanced Decision Making Engine

Provides intelligent decision-making capabilities:
- Cost-benefit analyzer for architectural approaches
- Performance predictor for scalability
- Risk assessment engine
- Trade-off negotiator
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import math

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Common architecture patterns"""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    MODULAR_MONOLITH = "modular_monolith"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"


class DatabaseType(Enum):
    """Database types"""
    SQL_RELATIONAL = "sql_relational"
    NOSQL_DOCUMENT = "nosql_document"
    NOSQL_KEY_VALUE = "nosql_key_value"
    NOSQL_GRAPH = "nosql_graph"
    TIME_SERIES = "time_series"
    VECTOR = "vector"


class RiskCategory(Enum):
    """Risk categories"""
    SECURITY = "security"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    COMPLIANCE = "compliance"
    COST = "cost"


class Priority(Enum):
    """Decision priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class CostBenefitResult:
    """Result of a cost-benefit analysis"""
    option_id: str
    option_name: str
    total_cost: float
    total_benefit: float
    net_benefit: float
    roi: float
    payback_period_months: Optional[float]
    costs_breakdown: Dict[str, float]
    benefits_breakdown: Dict[str, float]
    risks: List[Dict[str, Any]]
    recommendation_score: float
    confidence: float


@dataclass
class RiskAssessmentResult:
    """Result of a risk assessment"""
    id: str
    timestamp: datetime
    component: str
    risk_category: RiskCategory
    severity: str  # low, medium, high, critical
    probability: float  # 0-1
    impact: float  # 0-1
    risk_score: float
    description: str
    mitigations: List[str]
    affected_components: List[str]
    monitoring_recommendations: List[str]


@dataclass
class PerformancePrediction:
    """Performance prediction result"""
    metric: str
    current_value: float
    predicted_value: float
    prediction_horizon: str
    confidence_interval: Tuple[float, float]
    limiting_factors: List[str]
    recommendations: List[str]


@dataclass
class TradeoffAnalysis:
    """Trade-off analysis between competing priorities"""
    id: str
    timestamp: datetime
    priorities: Dict[str, float]  # priority -> weight
    options: List[Dict[str, Any]]
    recommended_option: str
    reasoning: str
    compromises: List[str]
    stakeholder_impacts: Dict[str, str]


class CostBenefitAnalyzer:
    """
    Analyzes cost-benefit trade-offs for different architectural approaches.

    Considers development time, operational costs, scalability, and maintenance.
    """

    # Cost factors for different approaches
    ARCHITECTURE_COSTS = {
        ArchitecturePattern.MONOLITH: {
            "initial_development": 0.7,
            "deployment_complexity": 0.3,
            "operational_overhead": 0.4,
            "scaling_cost": 0.8,
            "team_coordination": 0.3
        },
        ArchitecturePattern.MICROSERVICES: {
            "initial_development": 1.5,
            "deployment_complexity": 0.9,
            "operational_overhead": 0.8,
            "scaling_cost": 0.4,
            "team_coordination": 0.7
        },
        ArchitecturePattern.SERVERLESS: {
            "initial_development": 0.8,
            "deployment_complexity": 0.5,
            "operational_overhead": 0.2,
            "scaling_cost": 0.1,
            "team_coordination": 0.4
        },
        ArchitecturePattern.MODULAR_MONOLITH: {
            "initial_development": 0.9,
            "deployment_complexity": 0.4,
            "operational_overhead": 0.5,
            "scaling_cost": 0.6,
            "team_coordination": 0.4
        }
    }

    # Benefit factors
    ARCHITECTURE_BENEFITS = {
        ArchitecturePattern.MONOLITH: {
            "time_to_market": 0.9,
            "simplicity": 0.9,
            "debugging_ease": 0.8,
            "testing_ease": 0.7,
            "team_independence": 0.3
        },
        ArchitecturePattern.MICROSERVICES: {
            "time_to_market": 0.5,
            "simplicity": 0.3,
            "debugging_ease": 0.4,
            "testing_ease": 0.5,
            "team_independence": 0.9
        },
        ArchitecturePattern.SERVERLESS: {
            "time_to_market": 0.8,
            "simplicity": 0.6,
            "debugging_ease": 0.5,
            "testing_ease": 0.6,
            "team_independence": 0.7
        },
        ArchitecturePattern.MODULAR_MONOLITH: {
            "time_to_market": 0.7,
            "simplicity": 0.7,
            "debugging_ease": 0.7,
            "testing_ease": 0.8,
            "team_independence": 0.6
        }
    }

    DATABASE_COSTS = {
        DatabaseType.SQL_RELATIONAL: {
            "setup_complexity": 0.5,
            "scaling_difficulty": 0.7,
            "operational_cost": 0.5,
            "learning_curve": 0.3
        },
        DatabaseType.NOSQL_DOCUMENT: {
            "setup_complexity": 0.4,
            "scaling_difficulty": 0.4,
            "operational_cost": 0.4,
            "learning_curve": 0.5
        },
        DatabaseType.NOSQL_KEY_VALUE: {
            "setup_complexity": 0.3,
            "scaling_difficulty": 0.2,
            "operational_cost": 0.3,
            "learning_curve": 0.4
        }
    }

    def __init__(self, base_dev_cost_per_day: float = 1000):
        self.base_cost = base_dev_cost_per_day
        self.analysis_history: List[CostBenefitResult] = []

        logger.info("CostBenefitAnalyzer initialized")

    def analyze_architecture(
        self,
        options: List[ArchitecturePattern],
        project_context: Dict[str, Any]
    ) -> List[CostBenefitResult]:
        """Analyze cost-benefit for architecture options"""
        results = []

        team_size = project_context.get("team_size", 5)
        expected_scale = project_context.get("expected_users", 10000)
        time_to_market_priority = project_context.get("time_to_market_priority", 0.5)
        maintenance_horizon_years = project_context.get("maintenance_years", 3)

        for option in options:
            costs = self.ARCHITECTURE_COSTS.get(option, {})
            benefits = self.ARCHITECTURE_BENEFITS.get(option, {})

            # Calculate total costs (normalized)
            cost_weights = {
                "initial_development": 0.3,
                "deployment_complexity": 0.15,
                "operational_overhead": 0.25,
                "scaling_cost": 0.2,
                "team_coordination": 0.1
            }

            total_cost = sum(
                costs.get(k, 0.5) * w
                for k, w in cost_weights.items()
            )

            # Adjust for scale
            if expected_scale > 100000:
                total_cost *= costs.get("scaling_cost", 0.5) + 0.5

            # Adjust for team size
            if team_size > 10:
                total_cost *= costs.get("team_coordination", 0.5) + 0.5

            # Calculate total benefits
            benefit_weights = {
                "time_to_market": time_to_market_priority,
                "simplicity": 0.2,
                "debugging_ease": 0.15,
                "testing_ease": 0.15,
                "team_independence": 0.1 if team_size <= 5 else 0.3
            }

            # Normalize weights
            total_weight = sum(benefit_weights.values())
            benefit_weights = {k: v / total_weight for k, v in benefit_weights.items()}

            total_benefit = sum(
                benefits.get(k, 0.5) * w
                for k, w in benefit_weights.items()
            )

            # Calculate ROI and recommendation
            net_benefit = total_benefit - total_cost
            roi = total_benefit / max(total_cost, 0.1)

            # Estimate payback period
            monthly_benefit_rate = total_benefit / 12
            payback = total_cost / monthly_benefit_rate if monthly_benefit_rate > 0 else None

            # Risk analysis
            risks = self._assess_architecture_risks(option, project_context)

            # Final recommendation score
            risk_penalty = sum(r["impact"] * r["probability"] for r in risks) * 0.3
            recommendation_score = (roi * 0.5 + net_benefit * 0.3 + (1 - risk_penalty) * 0.2)

            result = CostBenefitResult(
                option_id=f"arch_{option.value}_{uuid.uuid4().hex[:8]}",
                option_name=option.value,
                total_cost=round(total_cost, 3),
                total_benefit=round(total_benefit, 3),
                net_benefit=round(net_benefit, 3),
                roi=round(roi, 3),
                payback_period_months=round(payback, 1) if payback else None,
                costs_breakdown={k: round(costs.get(k, 0.5) * cost_weights.get(k, 0.1), 3)
                                for k in cost_weights},
                benefits_breakdown={k: round(benefits.get(k, 0.5) * benefit_weights.get(k, 0.1), 3)
                                   for k in benefit_weights},
                risks=risks,
                recommendation_score=round(recommendation_score, 3),
                confidence=0.75
            )

            results.append(result)
            self.analysis_history.append(result)

        # Sort by recommendation score
        results.sort(key=lambda x: x.recommendation_score, reverse=True)
        return results

    def _assess_architecture_risks(
        self,
        pattern: ArchitecturePattern,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess risks for an architecture pattern"""
        risks = []

        if pattern == ArchitecturePattern.MICROSERVICES:
            risks.append({
                "category": "complexity",
                "description": "Increased operational complexity",
                "probability": 0.7,
                "impact": 0.5,
                "mitigation": "Invest in observability and DevOps automation"
            })

            if context.get("team_size", 5) < 5:
                risks.append({
                    "category": "team_capacity",
                    "description": "Team may be too small for microservices",
                    "probability": 0.8,
                    "impact": 0.7,
                    "mitigation": "Consider modular monolith instead"
                })

        elif pattern == ArchitecturePattern.MONOLITH:
            if context.get("expected_users", 0) > 100000:
                risks.append({
                    "category": "scalability",
                    "description": "May face scaling challenges at high load",
                    "probability": 0.6,
                    "impact": 0.8,
                    "mitigation": "Plan for horizontal scaling or migration path"
                })

        elif pattern == ArchitecturePattern.SERVERLESS:
            risks.append({
                "category": "vendor_lock",
                "description": "Potential vendor lock-in",
                "probability": 0.9,
                "impact": 0.4,
                "mitigation": "Use abstraction layers and infrastructure-as-code"
            })

        return risks

    def compare_databases(
        self,
        options: List[DatabaseType],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare database options based on requirements"""
        results = []

        read_heavy = requirements.get("read_heavy", False)
        write_heavy = requirements.get("write_heavy", False)
        complex_queries = requirements.get("complex_queries", False)
        horizontal_scaling = requirements.get("horizontal_scaling", False)

        for db_type in options:
            score = 0.5  # Base score
            reasons = []

            if db_type == DatabaseType.SQL_RELATIONAL:
                if complex_queries:
                    score += 0.3
                    reasons.append("Excellent for complex queries and joins")
                if horizontal_scaling:
                    score -= 0.2
                    reasons.append("Horizontal scaling can be challenging")

            elif db_type == DatabaseType.NOSQL_DOCUMENT:
                if not complex_queries:
                    score += 0.2
                    reasons.append("Good for simple queries and document storage")
                if horizontal_scaling:
                    score += 0.2
                    reasons.append("Easy to scale horizontally")
                if read_heavy:
                    score += 0.1
                    reasons.append("Optimized for read-heavy workloads")

            elif db_type == DatabaseType.NOSQL_KEY_VALUE:
                if read_heavy:
                    score += 0.3
                    reasons.append("Excellent read performance")
                if complex_queries:
                    score -= 0.3
                    reasons.append("Limited query capabilities")

            results.append({
                "type": db_type.value,
                "score": round(score, 3),
                "reasons": reasons,
                "recommended": score >= 0.6
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class PerformancePredictor:
    """
    Predicts scalability and performance needs before implementation.

    Uses historical data and modeling to estimate future requirements.
    """

    def __init__(self):
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.models: Dict[str, Callable] = {}

        logger.info("PerformancePredictor initialized")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric"""
        if metric_name not in self.historical_data:
            self.historical_data[metric_name] = []

        self.historical_data[metric_name].append({
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "context": context or {}
        })

    def predict(
        self,
        metric_name: str,
        horizon_days: int = 30,
        growth_factor: float = 1.0
    ) -> PerformancePrediction:
        """Predict future metric values"""
        history = self.historical_data.get(metric_name, [])

        if len(history) < 3:
            # Not enough data, use simple projection
            current_value = history[-1]["value"] if history else 0
            predicted = current_value * (1 + growth_factor * horizon_days / 30)

            return PerformancePrediction(
                metric=metric_name,
                current_value=current_value,
                predicted_value=predicted,
                prediction_horizon=f"{horizon_days} days",
                confidence_interval=(predicted * 0.7, predicted * 1.5),
                limiting_factors=["Insufficient historical data"],
                recommendations=["Collect more data for accurate predictions"]
            )

        # Use simple linear regression
        values = [h["value"] for h in history]
        current = values[-1]
        mean_value = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        # Simple trend estimation
        if len(values) >= 5:
            recent_trend = (values[-1] - values[-5]) / 5
        else:
            recent_trend = (values[-1] - values[0]) / len(values)

        predicted = current + recent_trend * horizon_days

        # Confidence interval based on historical variance
        ci_margin = stdev * 2

        # Identify limiting factors
        limiting_factors = []
        if predicted > current * 2:
            limiting_factors.append("Rapid growth may strain resources")
        if stdev > mean_value * 0.5:
            limiting_factors.append("High variability in historical data")

        # Generate recommendations
        recommendations = []
        if predicted > current * 1.5:
            recommendations.append("Consider scaling resources proactively")
        if "response_time" in metric_name.lower() and predicted > 1000:
            recommendations.append("Optimize performance or add caching")

        return PerformancePrediction(
            metric=metric_name,
            current_value=round(current, 2),
            predicted_value=round(max(0, predicted), 2),
            prediction_horizon=f"{horizon_days} days",
            confidence_interval=(round(max(0, predicted - ci_margin), 2),
                               round(predicted + ci_margin, 2)),
            limiting_factors=limiting_factors,
            recommendations=recommendations
        )

    def estimate_capacity(
        self,
        current_load: float,
        target_load: float,
        current_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate resources needed for target load"""
        scale_factor = target_load / max(current_load, 1)

        # Simple linear scaling estimation
        estimated_resources = {}
        for resource, value in current_resources.items():
            if isinstance(value, (int, float)):
                # Apply sub-linear scaling for some resources
                if resource in ["cpu", "memory"]:
                    estimated_resources[resource] = value * math.pow(scale_factor, 0.8)
                else:
                    estimated_resources[resource] = value * scale_factor
            else:
                estimated_resources[resource] = value

        bottlenecks = []
        if scale_factor > 2:
            bottlenecks.append("Database connections may become a bottleneck")
        if scale_factor > 5:
            bottlenecks.append("Consider distributed architecture")

        return {
            "current_load": current_load,
            "target_load": target_load,
            "scale_factor": round(scale_factor, 2),
            "estimated_resources": estimated_resources,
            "potential_bottlenecks": bottlenecks,
            "confidence": 0.6 if scale_factor < 5 else 0.4
        }


class RiskAssessment:
    """
    Risk assessment engine for security and reliability issues.

    Identifies potential risks and provides mitigation strategies.
    """

    # Risk patterns for different categories
    RISK_PATTERNS = {
        RiskCategory.SECURITY: [
            {
                "pattern": "no_auth",
                "description": "Endpoint without authentication",
                "severity": "high",
                "mitigation": "Add authentication middleware"
            },
            {
                "pattern": "sql_injection",
                "description": "Potential SQL injection vulnerability",
                "severity": "critical",
                "mitigation": "Use parameterized queries"
            },
            {
                "pattern": "hardcoded_secrets",
                "description": "Hardcoded secrets in source code",
                "severity": "critical",
                "mitigation": "Use environment variables or secret manager"
            }
        ],
        RiskCategory.RELIABILITY: [
            {
                "pattern": "no_retry",
                "description": "No retry logic for external calls",
                "severity": "medium",
                "mitigation": "Implement retry with exponential backoff"
            },
            {
                "pattern": "no_circuit_breaker",
                "description": "No circuit breaker for external dependencies",
                "severity": "medium",
                "mitigation": "Add circuit breaker pattern"
            },
            {
                "pattern": "single_point_failure",
                "description": "Single point of failure detected",
                "severity": "high",
                "mitigation": "Add redundancy or failover"
            }
        ],
        RiskCategory.PERFORMANCE: [
            {
                "pattern": "n_plus_one",
                "description": "Potential N+1 query pattern",
                "severity": "medium",
                "mitigation": "Use eager loading or batch queries"
            },
            {
                "pattern": "no_caching",
                "description": "No caching for frequently accessed data",
                "severity": "low",
                "mitigation": "Implement caching layer"
            }
        ],
        RiskCategory.SCALABILITY: [
            {
                "pattern": "sync_processing",
                "description": "Synchronous processing of heavy tasks",
                "severity": "medium",
                "mitigation": "Use async processing or message queues"
            },
            {
                "pattern": "memory_leak",
                "description": "Potential memory leak pattern",
                "severity": "high",
                "mitigation": "Review resource cleanup and disposal"
            }
        ]
    }

    def __init__(self):
        self.assessments: List[RiskAssessmentResult] = []
        self.risk_thresholds = {
            "critical": 0.8,
            "high": 0.6,
            "medium": 0.4,
            "low": 0.2
        }

        logger.info("RiskAssessment initialized")

    def assess_component(
        self,
        component: str,
        code_patterns: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[RiskAssessmentResult]:
        """Assess risks for a component"""
        results = []

        for category, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                # Check if pattern is present
                if pattern["pattern"] in code_patterns:
                    severity = pattern["severity"]
                    probability = 0.7 if severity == "critical" else 0.5
                    impact = self.risk_thresholds.get(severity, 0.5)

                    result = RiskAssessmentResult(
                        id=f"risk_{uuid.uuid4().hex[:12]}",
                        timestamp=datetime.now(),
                        component=component,
                        risk_category=category,
                        severity=severity,
                        probability=probability,
                        impact=impact,
                        risk_score=round(probability * impact, 3),
                        description=pattern["description"],
                        mitigations=[pattern["mitigation"]],
                        affected_components=[component],
                        monitoring_recommendations=[
                            f"Monitor for {pattern['pattern']} issues"
                        ]
                    )

                    results.append(result)
                    self.assessments.append(result)

        return results

    def calculate_overall_risk(
        self,
        assessments: Optional[List[RiskAssessmentResult]] = None
    ) -> Dict[str, Any]:
        """Calculate overall risk score from assessments"""
        if assessments is None:
            assessments = self.assessments

        if not assessments:
            return {
                "overall_score": 0,
                "risk_level": "low",
                "by_category": {},
                "critical_count": 0,
                "high_count": 0
            }

        # Calculate weighted average
        total_score = sum(a.risk_score for a in assessments)
        avg_score = total_score / len(assessments)

        # Count by severity
        critical_count = len([a for a in assessments if a.severity == "critical"])
        high_count = len([a for a in assessments if a.severity == "high"])

        # Determine overall risk level
        if critical_count > 0 or avg_score > 0.7:
            risk_level = "critical"
        elif high_count > 2 or avg_score > 0.5:
            risk_level = "high"
        elif avg_score > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Group by category
        by_category = {}
        for assessment in assessments:
            cat = assessment.risk_category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "component": assessment.component,
                "severity": assessment.severity,
                "score": assessment.risk_score
            })

        return {
            "overall_score": round(avg_score, 3),
            "risk_level": risk_level,
            "by_category": by_category,
            "critical_count": critical_count,
            "high_count": high_count,
            "total_risks": len(assessments)
        }

    def get_mitigation_plan(self) -> List[Dict[str, Any]]:
        """Generate a prioritized mitigation plan"""
        # Sort by risk score descending
        sorted_risks = sorted(
            self.assessments,
            key=lambda x: x.risk_score,
            reverse=True
        )

        plan = []
        for i, risk in enumerate(sorted_risks[:10], 1):  # Top 10 risks
            plan.append({
                "priority": i,
                "component": risk.component,
                "risk": risk.description,
                "severity": risk.severity,
                "score": risk.risk_score,
                "actions": risk.mitigations,
                "monitoring": risk.monitoring_recommendations
            })

        return plan


class TradeoffNegotiator:
    """
    Negotiates trade-offs between competing priorities.

    Balances speed, quality, cost, and other factors.
    """

    def __init__(self):
        self.analyses: List[TradeoffAnalysis] = []

        logger.info("TradeoffNegotiator initialized")

    def analyze_tradeoffs(
        self,
        priorities: Dict[str, float],
        options: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> TradeoffAnalysis:
        """Analyze trade-offs and recommend an option"""
        # Normalize priority weights
        total_weight = sum(priorities.values())
        normalized = {k: v / total_weight for k, v in priorities.items()}

        # Score each option
        scored_options = []
        for option in options:
            score = 0
            for priority, weight in normalized.items():
                option_value = option.get("scores", {}).get(priority, 0.5)
                score += option_value * weight

            # Apply constraints
            if constraints:
                for constraint, limit in constraints.items():
                    if option.get(constraint, 0) > limit:
                        score *= 0.5  # Penalty for violating constraints

            scored_options.append({
                **option,
                "final_score": round(score, 3)
            })

        # Sort by score
        scored_options.sort(key=lambda x: x["final_score"], reverse=True)

        # Identify compromises
        compromises = []
        recommended = scored_options[0]

        for priority, weight in normalized.items():
            if weight > 0.2:  # Significant priority
                if recommended.get("scores", {}).get(priority, 0) < 0.5:
                    compromises.append(
                        f"{priority} is somewhat compromised in this option"
                    )

        # Stakeholder impact analysis
        stakeholder_impacts = {
            "developers": "neutral",
            "operations": "neutral",
            "business": "neutral"
        }

        if recommended.get("scores", {}).get("speed", 0) > 0.7:
            stakeholder_impacts["business"] = "positive"
        if recommended.get("scores", {}).get("quality", 0) > 0.7:
            stakeholder_impacts["developers"] = "positive"
        if recommended.get("scores", {}).get("maintainability", 0) > 0.7:
            stakeholder_impacts["operations"] = "positive"

        analysis = TradeoffAnalysis(
            id=f"tradeoff_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            priorities=normalized,
            options=scored_options,
            recommended_option=recommended.get("name", "Option 1"),
            reasoning=f"Best balance of weighted priorities with score {recommended['final_score']:.2f}",
            compromises=compromises,
            stakeholder_impacts=stakeholder_impacts
        )

        self.analyses.append(analysis)
        return analysis

    def suggest_priority_adjustment(
        self,
        current_priorities: Dict[str, float],
        project_phase: str
    ) -> Dict[str, float]:
        """Suggest priority adjustments based on project phase"""
        adjustments = {
            "planning": {
                "quality": 0.3,
                "speed": 0.2,
                "cost": 0.2,
                "maintainability": 0.3
            },
            "development": {
                "quality": 0.25,
                "speed": 0.35,
                "cost": 0.15,
                "maintainability": 0.25
            },
            "testing": {
                "quality": 0.4,
                "speed": 0.2,
                "cost": 0.1,
                "maintainability": 0.3
            },
            "deployment": {
                "quality": 0.2,
                "speed": 0.3,
                "cost": 0.2,
                "maintainability": 0.3
            },
            "maintenance": {
                "quality": 0.25,
                "speed": 0.15,
                "cost": 0.25,
                "maintainability": 0.35
            }
        }

        suggested = adjustments.get(project_phase, current_priorities)

        return {
            "current": current_priorities,
            "suggested": suggested,
            "phase": project_phase,
            "changes": {
                k: round(suggested.get(k, 0) - current_priorities.get(k, 0), 2)
                for k in set(current_priorities) | set(suggested)
            }
        }


class DecisionEngine:
    """
    Main decision engine that orchestrates all decision-making components.
    """

    def __init__(self):
        self.cost_benefit = CostBenefitAnalyzer()
        self.performance = PerformancePredictor()
        self.risk = RiskAssessment()
        self.tradeoff = TradeoffNegotiator()

        self.decision_history: List[Dict[str, Any]] = []

        logger.info("DecisionEngine initialized")

    def make_architecture_decision(
        self,
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an architecture decision based on project context"""
        # Analyze all architecture options
        options = [
            ArchitecturePattern.MONOLITH,
            ArchitecturePattern.MICROSERVICES,
            ArchitecturePattern.SERVERLESS,
            ArchitecturePattern.MODULAR_MONOLITH
        ]

        cost_benefit_results = self.cost_benefit.analyze_architecture(
            options, project_context
        )

        # Get the best option
        best = cost_benefit_results[0]

        # Assess risks
        risks = self.risk.assess_component(
            best.option_name,
            [best.option_name],
            project_context
        )

        decision = {
            "decision_id": f"decision_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "type": "architecture",
            "recommendation": best.option_name,
            "score": best.recommendation_score,
            "roi": best.roi,
            "net_benefit": best.net_benefit,
            "risks": [
                {
                    "severity": r.severity,
                    "description": r.description,
                    "mitigation": r.mitigations
                }
                for r in risks
            ],
            "alternatives": [
                {
                    "option": r.option_name,
                    "score": r.recommendation_score
                }
                for r in cost_benefit_results[1:3]
            ],
            "reasoning": f"Selected {best.option_name} with ROI of {best.roi:.2f} "
                        f"and net benefit of {best.net_benefit:.2f}. "
                        f"This option best fits the project constraints."
        }

        self.decision_history.append(decision)
        return decision

    def make_database_decision(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a database technology decision"""
        options = [
            DatabaseType.SQL_RELATIONAL,
            DatabaseType.NOSQL_DOCUMENT,
            DatabaseType.NOSQL_KEY_VALUE
        ]

        results = self.cost_benefit.compare_databases(options, requirements)
        best = results[0]

        decision = {
            "decision_id": f"decision_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "type": "database",
            "recommendation": best["type"],
            "score": best["score"],
            "reasons": best["reasons"],
            "alternatives": results[1:],
            "reasoning": f"Selected {best['type']} based on requirements analysis"
        }

        self.decision_history.append(decision)
        return decision

    def get_decision_explanation(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed explanation for a decision"""
        for decision in self.decision_history:
            if decision["decision_id"] == decision_id:
                return {
                    **decision,
                    "explanation": {
                        "factors_considered": [
                            "Cost analysis",
                            "Benefit analysis",
                            "Risk assessment",
                            "Project context"
                        ],
                        "methodology": "Weighted multi-criteria decision analysis",
                        "confidence_level": "High" if decision.get("score", 0) > 0.7 else "Medium"
                    }
                }
        return None
