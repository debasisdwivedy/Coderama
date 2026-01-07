"""
Observability & Monitoring Module

Provides comprehensive monitoring capabilities:
- Real-time dashboards showing agent reasoning chains
- Automatic anomaly detection in development patterns
- Performance metrics for each agent
- Audit trails with explainable AI reasoning
"""

import asyncio
import json
import logging
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import time

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_HIGH = "error_rate_high"
    THROUGHPUT_DROP = "throughput_drop"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PATTERN_DEVIATION = "pattern_deviation"
    QUALITY_DEGRADATION = "quality_degradation"


class ReasoningType(Enum):
    """Types of agent reasoning steps"""
    ANALYSIS = "analysis"
    DECISION = "decision"
    ACTION = "action"
    EVALUATION = "evaluation"
    CORRECTION = "correction"


@dataclass
class ReasoningStep:
    """Represents a single step in an agent's reasoning chain"""
    id: str
    timestamp: datetime
    agent_name: str
    reasoning_type: ReasoningType
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    duration_ms: float
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEntry:
    """Represents an audit log entry"""
    id: str
    timestamp: datetime
    actor: str  # Agent or user
    action: str
    resource: str
    resource_type: str
    details: Dict[str, Any]
    outcome: str  # success, failure, pending
    reasoning: Optional[str] = None
    parent_entry_id: Optional[str] = None


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: str  # low, medium, high, critical
    component: str
    description: str
    metrics: Dict[str, Any]
    baseline: Dict[str, Any]
    deviation: float
    suggested_action: Optional[str] = None
    acknowledged: bool = False


@dataclass
class AgentMetricSnapshot:
    """Snapshot of agent performance metrics"""
    agent_name: str
    timestamp: datetime
    code_quality_score: float
    test_coverage: float
    velocity: float  # Tasks per hour
    error_rate: float
    avg_response_time_ms: float
    successful_tasks: int
    failed_tasks: int
    lines_of_code_generated: int
    bugs_introduced: int
    bugs_fixed: int


class MetricCollector:
    """Thread-safe metric collection"""

    def __init__(self, max_history: int = 10000):
        self._metrics: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.max_history = max_history

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + value
            self._record(name, MetricType.COUNTER, value, tags)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
            self._record(name, MetricType.GAUGE, value, tags)

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        with self._lock:
            self._record(name, MetricType.HISTOGRAM, value, tags)

    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        with self._lock:
            self._record(name, MetricType.TIMER, duration_ms, tags)

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric"""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}:{tag_str}"
        return name

    def _record(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]]
    ) -> None:
        """Record a metric value"""
        key = self._make_key(name, tags)
        if key not in self._metrics:
            self._metrics[key] = deque(maxlen=self.max_history)

        self._metrics[key].append({
            "timestamp": datetime.now().isoformat(),
            "type": metric_type.value,
            "value": value,
            "tags": tags or {}
        })

    def get_metrics(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get metric history"""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self._metrics:
                return []

            metrics = list(self._metrics[key])
            if since:
                metrics = [
                    m for m in metrics
                    if datetime.fromisoformat(m["timestamp"]) >= since
                ]
            return metrics

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value"""
        with self._lock:
            key = self._make_key(name, tags)
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value"""
        with self._lock:
            key = self._make_key(name, tags)
            return self._gauges.get(key, 0.0)

    def get_statistics(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get statistics for a metric over a time window"""
        since = datetime.now() - timedelta(minutes=window_minutes)
        metrics = self.get_metrics(name, tags, since)

        if not metrics:
            return {}

        values = [m["value"] for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": sorted(values)[int(len(values) * 0.95)] if values else 0,
            "p99": sorted(values)[int(len(values) * 0.99)] if values else 0
        }


class ReasoningChainTracker:
    """
    Tracks agent reasoning chains for explainability.

    Records the decision-making process of each agent for
    debugging and compliance auditing.
    """

    def __init__(self, max_chains: int = 1000):
        self.chains: Dict[str, List[ReasoningStep]] = {}
        self.active_chains: Dict[str, str] = {}  # agent -> chain_id
        self.max_chains = max_chains
        self._lock = threading.Lock()

        logger.info("ReasoningChainTracker initialized")

    def start_chain(self, agent_name: str, description: str = "") -> str:
        """Start a new reasoning chain for an agent"""
        chain_id = f"chain_{agent_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        with self._lock:
            self.chains[chain_id] = []
            self.active_chains[agent_name] = chain_id

            # Cleanup old chains
            if len(self.chains) > self.max_chains:
                oldest = sorted(self.chains.keys())[0]
                del self.chains[oldest]

        logger.debug(f"Started reasoning chain {chain_id} for {agent_name}")
        return chain_id

    def add_step(
        self,
        agent_name: str,
        reasoning_type: ReasoningType,
        description: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        confidence: float = 1.0,
        duration_ms: float = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningStep:
        """Add a step to the current reasoning chain"""
        step_id = f"step_{uuid.uuid4().hex[:12]}"

        step = ReasoningStep(
            id=step_id,
            timestamp=datetime.now(),
            agent_name=agent_name,
            reasoning_type=reasoning_type,
            description=description,
            inputs=inputs,
            outputs=outputs,
            confidence=confidence,
            duration_ms=duration_ms,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        with self._lock:
            chain_id = self.active_chains.get(agent_name)
            if chain_id and chain_id in self.chains:
                self.chains[chain_id].append(step)

        return step

    def end_chain(self, agent_name: str) -> Optional[List[ReasoningStep]]:
        """End and return the current reasoning chain"""
        with self._lock:
            chain_id = self.active_chains.pop(agent_name, None)
            if chain_id:
                return self.chains.get(chain_id, [])
        return None

    def get_chain(self, chain_id: str) -> List[ReasoningStep]:
        """Get a specific reasoning chain"""
        with self._lock:
            return list(self.chains.get(chain_id, []))

    def explain_decision(self, chain_id: str, decision_step_id: str) -> Dict[str, Any]:
        """Generate an explanation for a specific decision"""
        chain = self.get_chain(chain_id)

        target_step = None
        leading_steps = []

        for step in chain:
            if step.id == decision_step_id:
                target_step = step
                break
            leading_steps.append(step)

        if not target_step:
            return {"error": "Decision step not found"}

        # Find parent chain
        parent_chain = []
        current_id = target_step.parent_id
        while current_id:
            for step in chain:
                if step.id == current_id:
                    parent_chain.insert(0, step)
                    current_id = step.parent_id
                    break
            else:
                break

        return {
            "decision": {
                "id": target_step.id,
                "type": target_step.reasoning_type.value,
                "description": target_step.description,
                "confidence": target_step.confidence,
                "inputs": target_step.inputs,
                "outputs": target_step.outputs
            },
            "reasoning_path": [
                {
                    "step": s.id,
                    "type": s.reasoning_type.value,
                    "description": s.description,
                    "confidence": s.confidence
                }
                for s in parent_chain + [target_step]
            ],
            "context_considered": [
                {
                    "step": s.id,
                    "type": s.reasoning_type.value,
                    "description": s.description
                }
                for s in leading_steps[-5:]  # Last 5 steps before decision
            ]
        }

    def get_decision_summary(self, agent_name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of decisions made by an agent"""
        decisions = []

        with self._lock:
            for chain_id, steps in self.chains.items():
                for step in steps:
                    if step.agent_name == agent_name and step.reasoning_type == ReasoningType.DECISION:
                        if since is None or step.timestamp >= since:
                            decisions.append({
                                "chain_id": chain_id,
                                "step_id": step.id,
                                "timestamp": step.timestamp.isoformat(),
                                "description": step.description,
                                "confidence": step.confidence
                            })

        return {
            "agent": agent_name,
            "total_decisions": len(decisions),
            "average_confidence": statistics.mean([d["confidence"] for d in decisions]) if decisions else 0,
            "decisions": decisions[-20:]  # Last 20 decisions
        }


class AuditTrail:
    """
    Comprehensive audit trail for compliance and debugging.

    Records all actions taken by agents and users with full context.
    """

    def __init__(self, storage_path: Optional[str] = None, max_entries: int = 50000):
        self.entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self.storage_path = Path(storage_path) if storage_path else None
        self._lock = threading.Lock()

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_entries()

        logger.info(f"AuditTrail initialized with storage: {storage_path}")

    def _load_entries(self) -> None:
        """Load existing audit entries from storage"""
        if not self.storage_path:
            return

        audit_file = self.storage_path / "audit.jsonl"
        if audit_file.exists():
            try:
                with open(audit_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        self.entries.append(AuditEntry(
                            id=data["id"],
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            actor=data["actor"],
                            action=data["action"],
                            resource=data["resource"],
                            resource_type=data["resource_type"],
                            details=data["details"],
                            outcome=data["outcome"],
                            reasoning=data.get("reasoning"),
                            parent_entry_id=data.get("parent_entry_id")
                        ))
            except Exception as e:
                logger.warning(f"Failed to load audit entries: {e}")

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist an entry to storage"""
        if not self.storage_path:
            return

        audit_file = self.storage_path / "audit.jsonl"
        try:
            with open(audit_file, "a") as f:
                f.write(json.dumps({
                    "id": entry.id,
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "action": entry.action,
                    "resource": entry.resource,
                    "resource_type": entry.resource_type,
                    "details": entry.details,
                    "outcome": entry.outcome,
                    "reasoning": entry.reasoning,
                    "parent_entry_id": entry.parent_entry_id
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")

    def log(
        self,
        actor: str,
        action: str,
        resource: str,
        resource_type: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        reasoning: Optional[str] = None,
        parent_entry_id: Optional[str] = None
    ) -> AuditEntry:
        """Log an audit entry"""
        entry = AuditEntry(
            id=f"audit_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            actor=actor,
            action=action,
            resource=resource,
            resource_type=resource_type,
            details=details or {},
            outcome=outcome,
            reasoning=reasoning,
            parent_entry_id=parent_entry_id
        )

        with self._lock:
            self.entries.append(entry)
            self._persist_entry(entry)

        return entry

    def query(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        outcome: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Query audit entries with filters"""
        with self._lock:
            results = []
            for entry in reversed(self.entries):
                if actor and entry.actor != actor:
                    continue
                if action and entry.action != action:
                    continue
                if resource_type and entry.resource_type != resource_type:
                    continue
                if outcome and entry.outcome != outcome:
                    continue
                if since and entry.timestamp < since:
                    continue
                if until and entry.timestamp > until:
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break

            return results

    def get_actor_activity(self, actor: str, hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for an actor"""
        since = datetime.now() - timedelta(hours=hours)
        entries = self.query(actor=actor, since=since, limit=1000)

        actions = {}
        resources = {}
        outcomes = {"success": 0, "failure": 0, "pending": 0}

        for entry in entries:
            actions[entry.action] = actions.get(entry.action, 0) + 1
            resources[entry.resource_type] = resources.get(entry.resource_type, 0) + 1
            outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1

        return {
            "actor": actor,
            "period_hours": hours,
            "total_actions": len(entries),
            "actions_breakdown": actions,
            "resources_touched": resources,
            "outcomes": outcomes,
            "success_rate": outcomes["success"] / max(len(entries), 1)
        }

    def generate_compliance_report(
        self,
        since: datetime,
        until: datetime
    ) -> Dict[str, Any]:
        """Generate a compliance report for a time period"""
        entries = self.query(since=since, until=until, limit=10000)

        # Group by actor
        by_actor = {}
        for entry in entries:
            if entry.actor not in by_actor:
                by_actor[entry.actor] = []
            by_actor[entry.actor].append(entry)

        # Identify high-risk actions
        high_risk_actions = ["delete", "modify_config", "access_secrets", "deploy", "rollback"]
        high_risk_entries = [e for e in entries if any(hra in e.action.lower() for hra in high_risk_actions)]

        # Identify failures
        failures = [e for e in entries if e.outcome == "failure"]

        return {
            "period": {
                "start": since.isoformat(),
                "end": until.isoformat()
            },
            "total_entries": len(entries),
            "unique_actors": len(by_actor),
            "actor_summary": {
                actor: {
                    "total_actions": len(acts),
                    "success_rate": sum(1 for a in acts if a.outcome == "success") / max(len(acts), 1)
                }
                for actor, acts in by_actor.items()
            },
            "high_risk_actions": len(high_risk_entries),
            "high_risk_details": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "action": e.action,
                    "outcome": e.outcome,
                    "reasoning": e.reasoning
                }
                for e in high_risk_entries[:50]
            ],
            "failures": len(failures),
            "failure_details": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "action": e.action,
                    "details": e.details
                }
                for e in failures[:50]
            ]
        }


class AnomalyDetector:
    """
    Detects anomalies in development patterns and agent behavior.

    Uses statistical analysis to identify deviations from normal patterns.
    """

    def __init__(
        self,
        metrics: MetricCollector,
        sensitivity: float = 2.0,  # Standard deviations for anomaly threshold
        min_samples: int = 30
    ):
        self.metrics = metrics
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.anomalies: deque[Anomaly] = deque(maxlen=1000)
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._callbacks: List[Callable[[Anomaly], None]] = []

        logger.info(f"AnomalyDetector initialized with sensitivity={sensitivity}")

    def register_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """Register a callback for anomaly notifications"""
        self._callbacks.append(callback)

    def _calculate_baseline(self, metric_name: str, window_hours: int = 24) -> Dict[str, float]:
        """Calculate baseline statistics for a metric"""
        since = datetime.now() - timedelta(hours=window_hours)
        data = self.metrics.get_metrics(metric_name, since=since)

        if len(data) < self.min_samples:
            return {}

        values = [d["value"] for d in data]
        return {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "samples": len(values)
        }

    def update_baseline(self, metric_name: str) -> Dict[str, float]:
        """Update the baseline for a metric"""
        baseline = self._calculate_baseline(metric_name)
        if baseline:
            self.baselines[metric_name] = baseline
        return baseline

    def check_metric(
        self,
        metric_name: str,
        current_value: float,
        component: str = "unknown"
    ) -> Optional[Anomaly]:
        """Check if a metric value is anomalous"""
        baseline = self.baselines.get(metric_name)

        if not baseline or baseline.get("samples", 0) < self.min_samples:
            baseline = self.update_baseline(metric_name)

        if not baseline or baseline.get("stdev", 0) == 0:
            return None

        # Calculate z-score
        z_score = abs(current_value - baseline["mean"]) / baseline["stdev"]

        if z_score > self.sensitivity:
            # Determine severity
            if z_score > self.sensitivity * 3:
                severity = "critical"
            elif z_score > self.sensitivity * 2:
                severity = "high"
            elif z_score > self.sensitivity * 1.5:
                severity = "medium"
            else:
                severity = "low"

            # Determine anomaly type
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                anomaly_type = AnomalyType.LATENCY_SPIKE
            elif "error" in metric_name.lower():
                anomaly_type = AnomalyType.ERROR_RATE_HIGH
            elif "throughput" in metric_name.lower() or "rate" in metric_name.lower():
                anomaly_type = AnomalyType.THROUGHPUT_DROP
            else:
                anomaly_type = AnomalyType.PATTERN_DEVIATION

            anomaly = Anomaly(
                id=f"anomaly_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                anomaly_type=anomaly_type,
                severity=severity,
                component=component,
                description=f"{metric_name} deviated significantly from baseline",
                metrics={"current": current_value, "z_score": z_score},
                baseline=baseline,
                deviation=z_score,
                suggested_action=self._suggest_action(anomaly_type, severity)
            )

            self.anomalies.append(anomaly)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Anomaly callback failed: {e}")

            return anomaly

        return None

    def _suggest_action(self, anomaly_type: AnomalyType, severity: str) -> str:
        """Suggest an action based on anomaly type"""
        suggestions = {
            AnomalyType.LATENCY_SPIKE: "Check for resource contention or network issues. Consider scaling.",
            AnomalyType.ERROR_RATE_HIGH: "Review recent changes. Consider rollback if critical.",
            AnomalyType.THROUGHPUT_DROP: "Check system resources and external dependencies.",
            AnomalyType.RESOURCE_EXHAUSTION: "Scale resources or optimize memory usage.",
            AnomalyType.PATTERN_DEVIATION: "Investigate recent changes in behavior.",
            AnomalyType.QUALITY_DEGRADATION: "Review code quality metrics and test coverage."
        }
        return suggestions.get(anomaly_type, "Investigate the anomaly.")

    def get_recent_anomalies(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Anomaly]:
        """Get recent anomalies"""
        anomalies = list(self.anomalies)
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        return anomalies[-limit:]

    def acknowledge_anomaly(self, anomaly_id: str) -> bool:
        """Acknowledge an anomaly"""
        for anomaly in self.anomalies:
            if anomaly.id == anomaly_id:
                anomaly.acknowledged = True
                return True
        return False


class AgentMetrics:
    """
    Collects and tracks performance metrics for each agent.

    Provides insights into agent efficiency, code quality, and productivity.
    """

    def __init__(self, metrics: MetricCollector):
        self.metrics = metrics
        self.snapshots: Dict[str, deque[AgentMetricSnapshot]] = {}
        self.max_snapshots = 1000

        logger.info("AgentMetrics initialized")

    def record_task_completion(
        self,
        agent_name: str,
        success: bool,
        duration_ms: float,
        lines_of_code: int = 0,
        test_coverage: float = 0
    ) -> None:
        """Record a task completion event"""
        tags = {"agent": agent_name}

        self.metrics.increment(
            "agent.tasks.total",
            tags=tags
        )

        if success:
            self.metrics.increment("agent.tasks.success", tags=tags)
        else:
            self.metrics.increment("agent.tasks.failure", tags=tags)

        self.metrics.timer("agent.task.duration", duration_ms, tags=tags)
        self.metrics.histogram("agent.code.lines", lines_of_code, tags=tags)
        self.metrics.gauge("agent.test.coverage", test_coverage, tags=tags)

    def record_code_quality(
        self,
        agent_name: str,
        quality_score: float,
        bugs_found: int = 0,
        bugs_fixed: int = 0
    ) -> None:
        """Record code quality metrics"""
        tags = {"agent": agent_name}

        self.metrics.gauge("agent.code.quality", quality_score, tags=tags)
        self.metrics.increment("agent.bugs.found", bugs_found, tags=tags)
        self.metrics.increment("agent.bugs.fixed", bugs_fixed, tags=tags)

    def get_agent_snapshot(self, agent_name: str) -> AgentMetricSnapshot:
        """Get current metrics snapshot for an agent"""
        tags = {"agent": agent_name}
        stats = self.metrics.get_statistics("agent.task.duration", tags=tags, window_minutes=60)

        total_tasks = self.metrics.get_counter("agent.tasks.total", tags=tags)
        success_tasks = self.metrics.get_counter("agent.tasks.success", tags=tags)
        failure_tasks = self.metrics.get_counter("agent.tasks.failure", tags=tags)

        lines_stats = self.metrics.get_statistics("agent.code.lines", tags=tags, window_minutes=60)

        snapshot = AgentMetricSnapshot(
            agent_name=agent_name,
            timestamp=datetime.now(),
            code_quality_score=self.metrics.get_gauge("agent.code.quality", tags=tags),
            test_coverage=self.metrics.get_gauge("agent.test.coverage", tags=tags),
            velocity=success_tasks / max(1, 1),  # Tasks per hour approximation
            error_rate=failure_tasks / max(total_tasks, 1),
            avg_response_time_ms=stats.get("mean", 0),
            successful_tasks=success_tasks,
            failed_tasks=failure_tasks,
            lines_of_code_generated=int(lines_stats.get("count", 0) * lines_stats.get("mean", 0)),
            bugs_introduced=self.metrics.get_counter("agent.bugs.found", tags=tags),
            bugs_fixed=self.metrics.get_counter("agent.bugs.fixed", tags=tags)
        )

        # Store snapshot
        if agent_name not in self.snapshots:
            self.snapshots[agent_name] = deque(maxlen=self.max_snapshots)
        self.snapshots[agent_name].append(snapshot)

        return snapshot

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get agent leaderboard by performance"""
        agents = set()
        for key in self.metrics._counters.keys():
            if key.startswith("agent.tasks.total:agent="):
                agent = key.split("=")[1]
                agents.add(agent)

        leaderboard = []
        for agent in agents:
            snapshot = self.get_agent_snapshot(agent)
            score = (
                snapshot.code_quality_score * 0.3 +
                (1 - snapshot.error_rate) * 0.3 +
                snapshot.test_coverage * 0.2 +
                min(snapshot.velocity / 10, 1) * 0.2
            )
            leaderboard.append({
                "agent": agent,
                "score": score,
                "metrics": {
                    "quality": snapshot.code_quality_score,
                    "success_rate": 1 - snapshot.error_rate,
                    "coverage": snapshot.test_coverage,
                    "velocity": snapshot.velocity
                }
            })

        return sorted(leaderboard, key=lambda x: x["score"], reverse=True)


class ObservabilityDashboard:
    """
    Main dashboard for real-time observability.

    Aggregates all monitoring components into a unified view.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.metrics = MetricCollector()
        self.reasoning_tracker = ReasoningChainTracker()
        self.audit_trail = AuditTrail(storage_path)
        self.anomaly_detector = AnomalyDetector(self.metrics)
        self.agent_metrics = AgentMetrics(self.metrics)

        # Dashboard state
        self._start_time = datetime.now()

        logger.info("ObservabilityDashboard initialized")

    def get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview"""
        uptime = datetime.now() - self._start_time

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "agents": self.agent_metrics.get_leaderboard(),
            "recent_anomalies": [
                {
                    "id": a.id,
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "component": a.component,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self.anomaly_detector.get_recent_anomalies(limit=10)
            ],
            "audit_summary": {
                "total_entries": len(self.audit_trail.entries),
                "recent_failures": len(self.audit_trail.query(outcome="failure", limit=100))
            }
        }

    def get_agent_dashboard(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed dashboard for a specific agent"""
        snapshot = self.agent_metrics.get_agent_snapshot(agent_name)
        decisions = self.reasoning_tracker.get_decision_summary(agent_name)
        activity = self.audit_trail.get_actor_activity(agent_name)

        return {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "quality_score": snapshot.code_quality_score,
                "test_coverage": snapshot.test_coverage,
                "velocity": snapshot.velocity,
                "error_rate": snapshot.error_rate,
                "avg_response_time_ms": snapshot.avg_response_time_ms
            },
            "productivity": {
                "successful_tasks": snapshot.successful_tasks,
                "failed_tasks": snapshot.failed_tasks,
                "lines_generated": snapshot.lines_of_code_generated,
                "bugs_fixed": snapshot.bugs_fixed
            },
            "decision_making": decisions,
            "activity": activity
        }

    def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard"""
        since = datetime.now() - timedelta(minutes=window_minutes)

        return {
            "window_minutes": window_minutes,
            "timestamp": datetime.now().isoformat(),
            "task_completion_rate": self.metrics.get_statistics(
                "agent.tasks.total", window_minutes=window_minutes
            ),
            "response_times": self.metrics.get_statistics(
                "agent.task.duration", window_minutes=window_minutes
            ),
            "error_rate": self.metrics.get_statistics(
                "agent.tasks.failure", window_minutes=window_minutes
            ),
            "active_reasoning_chains": len(self.reasoning_tracker.active_chains),
            "recent_audit_entries": len(self.audit_trail.query(since=since))
        }

    def export_data(
        self,
        format: str = "json",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> str:
        """Export observability data"""
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        if until is None:
            until = datetime.now()

        data = {
            "export_time": datetime.now().isoformat(),
            "period": {
                "start": since.isoformat(),
                "end": until.isoformat()
            },
            "system_overview": self.get_system_overview(),
            "audit_report": self.audit_trail.generate_compliance_report(since, until),
            "anomalies": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.anomaly_type.value,
                    "severity": a.severity,
                    "component": a.component,
                    "description": a.description,
                    "deviation": a.deviation
                }
                for a in self.anomaly_detector.anomalies
                if since <= a.timestamp <= until
            ]
        }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)
