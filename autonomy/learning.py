"""
Continuous Learning Loop Module

Provides learning capabilities:
- Feedback system that learns from past sprints
- Pattern recognition for successful approaches
- Post-mortem analysis
- A/B testing different implementations
"""

import json
import logging
import statistics
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import re

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns recognized"""
    ARCHITECTURAL = "architectural"
    CODE_STRUCTURE = "code_structure"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    PERFORMANCE = "performance"
    SECURITY = "security"


class LearningEventType(Enum):
    """Types of learning events"""
    SPRINT_COMPLETED = "sprint_completed"
    BUG_FIXED = "bug_fixed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    DEPLOYMENT_SUCCESS = "deployment_success"
    DEPLOYMENT_FAILURE = "deployment_failure"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"


@dataclass
class LearningEvent:
    """Represents a learning event"""
    id: str
    timestamp: datetime
    event_type: LearningEventType
    context: Dict[str, Any]
    outcome: str  # success, failure, partial
    metrics: Dict[str, float]
    lessons: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class RecognizedPattern:
    """A recognized successful pattern"""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    occurrences: int
    success_rate: float
    contexts: List[str]
    code_examples: List[str]
    related_patterns: List[str]
    last_seen: datetime


@dataclass
class PostMortemReport:
    """Post-mortem analysis report"""
    id: str
    project_name: str
    timestamp: datetime
    duration_days: int
    summary: str
    what_went_well: List[str]
    what_went_wrong: List[str]
    lessons_learned: List[str]
    action_items: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class SprintEstimate:
    """Sprint estimation data"""
    sprint_id: str
    estimated_hours: float
    actual_hours: float
    complexity_score: float
    accuracy: float
    factors: Dict[str, float]


@dataclass
class ABTestResult:
    """A/B test result"""
    id: str
    name: str
    variant_a: Dict[str, Any]
    variant_b: Dict[str, Any]
    metrics: Dict[str, Dict[str, float]]
    winner: str
    confidence: float
    recommendation: str


class PatternRecognition:
    """
    Recognizes successful patterns from past projects.

    Learns from code patterns, architectural decisions, and development practices.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.patterns: Dict[str, RecognizedPattern] = {}
        self.pattern_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_patterns()

        logger.info("PatternRecognition initialized")

    def _load_patterns(self) -> None:
        """Load patterns from storage"""
        if not self.storage_path:
            return

        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = RecognizedPattern(
                            id=pattern_data["id"],
                            pattern_type=PatternType(pattern_data["pattern_type"]),
                            name=pattern_data["name"],
                            description=pattern_data["description"],
                            occurrences=pattern_data["occurrences"],
                            success_rate=pattern_data["success_rate"],
                            contexts=pattern_data["contexts"],
                            code_examples=pattern_data.get("code_examples", []),
                            related_patterns=pattern_data.get("related_patterns", []),
                            last_seen=datetime.fromisoformat(pattern_data["last_seen"])
                        )
                        self.patterns[pattern.id] = pattern
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Save patterns to storage"""
        if not self.storage_path:
            return

        patterns_file = self.storage_path / "patterns.json"
        data = [
            {
                "id": p.id,
                "pattern_type": p.pattern_type.value,
                "name": p.name,
                "description": p.description,
                "occurrences": p.occurrences,
                "success_rate": p.success_rate,
                "contexts": p.contexts,
                "code_examples": p.code_examples,
                "related_patterns": p.related_patterns,
                "last_seen": p.last_seen.isoformat()
            }
            for p in self.patterns.values()
        ]

        with open(patterns_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_occurrence(
        self,
        pattern_name: str,
        pattern_type: PatternType,
        context: str,
        success: bool,
        code_example: Optional[str] = None
    ) -> RecognizedPattern:
        """Record a pattern occurrence"""
        # Generate or find pattern ID
        pattern_id = f"pattern_{hashlib.md5(pattern_name.encode()).hexdigest()[:12]}"

        # Record occurrence
        self.pattern_occurrences[pattern_id].append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "success": success
        })

        # Update or create pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.occurrences += 1
            if context not in pattern.contexts:
                pattern.contexts.append(context)
            if code_example and code_example not in pattern.code_examples:
                pattern.code_examples.append(code_example[:500])
            pattern.last_seen = datetime.now()

            # Recalculate success rate
            occurrences = self.pattern_occurrences[pattern_id]
            successes = sum(1 for o in occurrences if o["success"])
            pattern.success_rate = successes / len(occurrences)
        else:
            pattern = RecognizedPattern(
                id=pattern_id,
                pattern_type=pattern_type,
                name=pattern_name,
                description=f"Pattern: {pattern_name}",
                occurrences=1,
                success_rate=1.0 if success else 0.0,
                contexts=[context],
                code_examples=[code_example[:500]] if code_example else [],
                related_patterns=[],
                last_seen=datetime.now()
            )
            self.patterns[pattern_id] = pattern

        self._save_patterns()
        return pattern

    def find_similar_patterns(
        self,
        context: str,
        pattern_type: Optional[PatternType] = None
    ) -> List[RecognizedPattern]:
        """Find patterns similar to the current context"""
        matches = []

        context_words = set(context.lower().split())

        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            # Calculate similarity based on context overlap
            pattern_contexts = " ".join(pattern.contexts).lower()
            pattern_words = set(pattern_contexts.split())

            overlap = len(context_words & pattern_words)
            if overlap > 2:  # Minimum overlap threshold
                matches.append((pattern, overlap, pattern.success_rate))

        # Sort by success rate and then overlap
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return [m[0] for m in matches[:10]]

    def get_successful_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_occurrences: int = 3,
        min_success_rate: float = 0.7
    ) -> List[RecognizedPattern]:
        """Get patterns with high success rates"""
        successful = []

        for pattern in self.patterns.values():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            if pattern.occurrences >= min_occurrences and pattern.success_rate >= min_success_rate:
                successful.append(pattern)

        successful.sort(key=lambda x: x.success_rate, reverse=True)
        return successful

    def suggest_pattern(
        self,
        context: str,
        pattern_type: PatternType
    ) -> Optional[Dict[str, Any]]:
        """Suggest a pattern for the given context"""
        similar = self.find_similar_patterns(context, pattern_type)

        if not similar:
            return None

        best = similar[0]

        return {
            "pattern_id": best.id,
            "name": best.name,
            "description": best.description,
            "success_rate": best.success_rate,
            "occurrences": best.occurrences,
            "examples": best.code_examples[:2],
            "confidence": min(best.success_rate, best.occurrences / 10)
        }


class SprintEstimator:
    """
    Learns from past sprints to improve time estimates.

    Uses historical data to predict sprint duration and complexity.
    """

    def __init__(self):
        self.estimates: List[SprintEstimate] = []
        self.complexity_factors: Dict[str, float] = {
            "new_technology": 1.3,
            "integration": 1.2,
            "refactoring": 0.8,
            "bug_fix": 0.6,
            "feature": 1.0,
            "ui_changes": 1.1,
            "database_changes": 1.25,
            "api_changes": 1.15
        }

        logger.info("SprintEstimator initialized")

    def record_sprint(
        self,
        sprint_id: str,
        estimated_hours: float,
        actual_hours: float,
        complexity_factors: List[str]
    ) -> SprintEstimate:
        """Record a completed sprint for learning"""
        complexity_score = self._calculate_complexity(complexity_factors)
        accuracy = 1 - abs(estimated_hours - actual_hours) / max(estimated_hours, 1)

        estimate = SprintEstimate(
            sprint_id=sprint_id,
            estimated_hours=estimated_hours,
            actual_hours=actual_hours,
            complexity_score=complexity_score,
            accuracy=accuracy,
            factors={f: self.complexity_factors.get(f, 1.0) for f in complexity_factors}
        )

        self.estimates.append(estimate)

        # Update complexity factors based on actual vs estimated
        self._update_factors(estimate)

        return estimate

    def _calculate_complexity(self, factors: List[str]) -> float:
        """Calculate complexity score from factors"""
        if not factors:
            return 1.0

        total = 1.0
        for factor in factors:
            total *= self.complexity_factors.get(factor, 1.0)

        return total

    def _update_factors(self, estimate: SprintEstimate) -> None:
        """Update complexity factors based on actual results"""
        if len(self.estimates) < 5:
            return

        # Analyze recent estimates for each factor
        for factor, weight in estimate.factors.items():
            # Find estimates with this factor
            with_factor = [e for e in self.estimates[-20:] if factor in e.factors]

            if len(with_factor) >= 3:
                # Calculate average under/overestimation
                ratios = [e.actual_hours / max(e.estimated_hours, 1) for e in with_factor]
                avg_ratio = statistics.mean(ratios)

                # Adjust factor slightly
                adjustment = (avg_ratio - 1) * 0.1
                self.complexity_factors[factor] = max(0.5, min(2.0,
                    self.complexity_factors[factor] * (1 + adjustment)
                ))

    def estimate_sprint(
        self,
        base_hours: float,
        complexity_factors: List[str],
        team_velocity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate an estimate for a new sprint"""
        complexity = self._calculate_complexity(complexity_factors)
        adjusted_hours = base_hours * complexity

        # Apply historical accuracy correction
        if len(self.estimates) >= 5:
            recent = self.estimates[-10:]
            avg_accuracy = statistics.mean(e.accuracy for e in recent)

            if avg_accuracy < 0.8:
                # We tend to underestimate, add buffer
                adjusted_hours *= 1.1
            elif avg_accuracy > 0.9:
                # We tend to overestimate, reduce slightly
                adjusted_hours *= 0.95

        # Apply team velocity if available
        if team_velocity:
            adjusted_hours *= (1.0 / team_velocity)

        # Calculate confidence interval
        if len(self.estimates) >= 3:
            variations = [e.actual_hours / max(e.estimated_hours, 1) for e in self.estimates[-10:]]
            stdev = statistics.stdev(variations) if len(variations) > 1 else 0.2
            lower = adjusted_hours * (1 - stdev)
            upper = adjusted_hours * (1 + stdev)
        else:
            lower = adjusted_hours * 0.7
            upper = adjusted_hours * 1.5

        return {
            "base_hours": base_hours,
            "complexity_score": round(complexity, 2),
            "adjusted_hours": round(adjusted_hours, 1),
            "confidence_interval": (round(lower, 1), round(upper, 1)),
            "factors_applied": {f: round(self.complexity_factors.get(f, 1.0), 2)
                               for f in complexity_factors},
            "confidence": self._calculate_confidence()
        }

    def _calculate_confidence(self) -> float:
        """Calculate confidence in estimates based on historical accuracy"""
        if len(self.estimates) < 5:
            return 0.5

        recent = self.estimates[-10:]
        avg_accuracy = statistics.mean(e.accuracy for e in recent)
        return min(0.95, avg_accuracy)

    def get_estimation_insights(self) -> Dict[str, Any]:
        """Get insights about estimation accuracy"""
        if not self.estimates:
            return {"message": "No historical data available"}

        total_estimated = sum(e.estimated_hours for e in self.estimates)
        total_actual = sum(e.actual_hours for e in self.estimates)
        accuracies = [e.accuracy for e in self.estimates]

        return {
            "total_sprints": len(self.estimates),
            "total_estimated_hours": total_estimated,
            "total_actual_hours": total_actual,
            "overall_ratio": round(total_actual / max(total_estimated, 1), 2),
            "average_accuracy": round(statistics.mean(accuracies), 2),
            "accuracy_trend": self._calculate_trend(accuracies),
            "most_impactful_factors": self._get_impactful_factors(),
            "recommendations": self._generate_recommendations()
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in values"""
        if len(values) < 5:
            return "insufficient_data"

        recent = statistics.mean(values[-5:])
        earlier = statistics.mean(values[:5])

        if recent > earlier + 0.05:
            return "improving"
        elif recent < earlier - 0.05:
            return "declining"
        return "stable"

    def _get_impactful_factors(self) -> List[Dict[str, Any]]:
        """Get the most impactful complexity factors"""
        sorted_factors = sorted(
            self.complexity_factors.items(),
            key=lambda x: abs(x[1] - 1.0),
            reverse=True
        )
        return [
            {"factor": f, "multiplier": round(m, 2)}
            for f, m in sorted_factors[:5]
        ]

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on patterns"""
        recommendations = []

        if len(self.estimates) >= 5:
            recent = self.estimates[-5:]
            avg_accuracy = statistics.mean(e.accuracy for e in recent)

            if avg_accuracy < 0.7:
                recommendations.append("Consider breaking down sprints into smaller tasks")
                recommendations.append("Add buffer time for unexpected issues")

            # Check for consistently underestimated factors
            for factor, weight in self.complexity_factors.items():
                if weight > 1.4:
                    recommendations.append(
                        f"'{factor}' tasks typically take longer than expected"
                    )

        return recommendations


class PostMortemAnalyzer:
    """
    Conducts post-mortem analysis after each project.

    Updates internal knowledge base with lessons learned.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.reports: List[PostMortemReport] = []
        self.knowledge_base: Dict[str, List[str]] = defaultdict(list)
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_knowledge()

        logger.info("PostMortemAnalyzer initialized")

    def _load_knowledge(self) -> None:
        """Load knowledge base from storage"""
        if not self.storage_path:
            return

        kb_file = self.storage_path / "knowledge_base.json"
        if kb_file.exists():
            try:
                with open(kb_file, "r") as f:
                    self.knowledge_base = defaultdict(list, json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")

    def _save_knowledge(self) -> None:
        """Save knowledge base to storage"""
        if not self.storage_path:
            return

        kb_file = self.storage_path / "knowledge_base.json"
        with open(kb_file, "w") as f:
            json.dump(dict(self.knowledge_base), f, indent=2)

    def conduct_post_mortem(
        self,
        project_name: str,
        metrics: Dict[str, Any],
        events: List[LearningEvent],
        team_feedback: Optional[List[str]] = None
    ) -> PostMortemReport:
        """Conduct a post-mortem analysis"""
        # Analyze events
        successes = [e for e in events if e.outcome == "success"]
        failures = [e for e in events if e.outcome == "failure"]

        # Identify what went well
        what_went_well = []
        if len(successes) > len(failures):
            what_went_well.append("Majority of tasks completed successfully")

        success_tags = Counter(tag for e in successes for tag in e.tags)
        for tag, count in success_tags.most_common(3):
            what_went_well.append(f"{tag} tasks had high success rate ({count} successes)")

        # Identify what went wrong
        what_went_wrong = []
        failure_tags = Counter(tag for e in failures for tag in e.tags)
        for tag, count in failure_tags.most_common(3):
            what_went_wrong.append(f"{tag} tasks had issues ({count} failures)")

        if metrics.get("deadline_missed"):
            what_went_wrong.append("Project deadline was not met")

        # Extract lessons
        lessons = []
        all_lessons = [lesson for e in events for lesson in e.lessons]
        lesson_counts = Counter(all_lessons)
        for lesson, count in lesson_counts.most_common(5):
            lessons.append(lesson)

        # Generate action items
        action_items = []
        for lesson in lessons[:3]:
            action_items.append({
                "action": f"Address: {lesson}",
                "priority": "high" if lesson in [l for e in failures for l in e.lessons] else "medium",
                "owner": "team"
            })

        # Calculate duration
        if events:
            start = min(e.timestamp for e in events)
            end = max(e.timestamp for e in events)
            duration = (end - start).days
        else:
            duration = 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            what_went_well, what_went_wrong, lessons
        )

        report = PostMortemReport(
            id=f"pm_{uuid.uuid4().hex[:12]}",
            project_name=project_name,
            timestamp=datetime.now(),
            duration_days=duration,
            summary=self._generate_summary(metrics, len(successes), len(failures)),
            what_went_well=what_went_well,
            what_went_wrong=what_went_wrong,
            lessons_learned=lessons,
            action_items=action_items,
            metrics=metrics,
            recommendations=recommendations
        )

        self.reports.append(report)

        # Update knowledge base
        self._update_knowledge(report)

        return report

    def _generate_summary(
        self,
        metrics: Dict[str, Any],
        successes: int,
        failures: int
    ) -> str:
        """Generate project summary"""
        total = successes + failures
        success_rate = successes / max(total, 1)

        summary_parts = [f"Project completed with {success_rate:.0%} success rate."]

        if metrics.get("on_time"):
            summary_parts.append("Delivered on time.")
        else:
            summary_parts.append("Experienced timeline overrun.")

        if metrics.get("quality_score", 0) >= 0.8:
            summary_parts.append("High code quality achieved.")

        return " ".join(summary_parts)

    def _generate_recommendations(
        self,
        went_well: List[str],
        went_wrong: List[str],
        lessons: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Continue doing what worked
        for item in went_well[:2]:
            recommendations.append(f"Continue: {item}")

        # Address issues
        for item in went_wrong[:2]:
            recommendations.append(f"Improve: {item}")

        # Apply lessons
        for lesson in lessons[:2]:
            recommendations.append(f"Apply lesson: {lesson}")

        return recommendations

    def _update_knowledge(self, report: PostMortemReport) -> None:
        """Update knowledge base with lessons from post-mortem"""
        # Index lessons by keywords
        for lesson in report.lessons_learned:
            words = lesson.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    self.knowledge_base[word].append(lesson)

        # Index recommendations
        for rec in report.recommendations:
            words = rec.lower().split()
            for word in words:
                if len(word) > 4:
                    self.knowledge_base[word].append(rec)

        self._save_knowledge()

    def query_knowledge(self, query: str) -> List[str]:
        """Query the knowledge base"""
        words = query.lower().split()
        results = []

        for word in words:
            if word in self.knowledge_base:
                results.extend(self.knowledge_base[word])

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        return unique_results[:10]

    def get_insights_for_project_type(
        self,
        project_type: str
    ) -> Dict[str, Any]:
        """Get insights for a specific project type"""
        relevant_reports = [
            r for r in self.reports
            if project_type.lower() in r.project_name.lower()
        ]

        if not relevant_reports:
            return {"message": "No relevant historical data"}

        all_lessons = [l for r in relevant_reports for l in r.lessons_learned]
        all_recommendations = [r for rep in relevant_reports for r in rep.recommendations]

        return {
            "project_type": project_type,
            "historical_projects": len(relevant_reports),
            "common_lessons": Counter(all_lessons).most_common(5),
            "common_recommendations": Counter(all_recommendations).most_common(5),
            "average_duration_days": statistics.mean(r.duration_days for r in relevant_reports)
        }


class ABTestEngine:
    """
    A/B testing for different implementation approaches.

    Tests different code patterns, architectures, or configurations.
    """

    def __init__(self):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.completed_tests: List[ABTestResult] = []

        logger.info("ABTestEngine initialized")

    def create_test(
        self,
        name: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        metrics_to_track: List[str],
        min_samples: int = 30
    ) -> str:
        """Create a new A/B test"""
        test_id = f"ab_{uuid.uuid4().hex[:12]}"

        self.active_tests[test_id] = {
            "id": test_id,
            "name": name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metrics_to_track": metrics_to_track,
            "min_samples": min_samples,
            "results_a": [],
            "results_b": [],
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        logger.info(f"Created A/B test: {name} ({test_id})")
        return test_id

    def record_result(
        self,
        test_id: str,
        variant: str,
        metrics: Dict[str, float]
    ) -> None:
        """Record a result for a test variant"""
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]

        if variant == "a":
            test["results_a"].append(metrics)
        else:
            test["results_b"].append(metrics)

        # Check if we have enough samples
        if (len(test["results_a"]) >= test["min_samples"] and
            len(test["results_b"]) >= test["min_samples"]):
            self._complete_test(test_id)

    def _complete_test(self, test_id: str) -> ABTestResult:
        """Complete a test and determine winner"""
        test = self.active_tests[test_id]

        # Calculate metrics for each variant
        metrics_a = self._aggregate_metrics(test["results_a"])
        metrics_b = self._aggregate_metrics(test["results_b"])

        # Determine winner based on primary metric
        primary_metric = test["metrics_to_track"][0]
        value_a = metrics_a.get(primary_metric, {}).get("mean", 0)
        value_b = metrics_b.get(primary_metric, {}).get("mean", 0)

        # Calculate statistical significance
        confidence = self._calculate_significance(
            test["results_a"],
            test["results_b"],
            primary_metric
        )

        if confidence > 0.95:
            winner = "A" if value_a > value_b else "B"
            recommendation = f"Adopt variant {winner} with high confidence"
        elif confidence > 0.8:
            winner = "A" if value_a > value_b else "B"
            recommendation = f"Consider variant {winner}, but collect more data"
        else:
            winner = "inconclusive"
            recommendation = "No significant difference, continue testing"

        result = ABTestResult(
            id=test_id,
            name=test["name"],
            variant_a=test["variant_a"],
            variant_b=test["variant_b"],
            metrics={
                "A": metrics_a,
                "B": metrics_b
            },
            winner=winner,
            confidence=confidence,
            recommendation=recommendation
        )

        self.completed_tests.append(result)
        test["status"] = "completed"

        logger.info(f"Completed A/B test: {test['name']}, winner: {winner}")
        return result

    def _aggregate_metrics(
        self,
        results: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics from results"""
        if not results:
            return {}

        aggregated = {}
        metrics = results[0].keys()

        for metric in metrics:
            values = [r.get(metric, 0) for r in results]
            aggregated[metric] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "samples": len(values)
            }

        return aggregated

    def _calculate_significance(
        self,
        results_a: List[Dict[str, float]],
        results_b: List[Dict[str, float]],
        metric: str
    ) -> float:
        """Calculate statistical significance using a simple t-test approximation"""
        values_a = [r.get(metric, 0) for r in results_a]
        values_b = [r.get(metric, 0) for r in results_b]

        if len(values_a) < 2 or len(values_b) < 2:
            return 0.0

        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        std_a = statistics.stdev(values_a)
        std_b = statistics.stdev(values_b)

        # Pooled standard error
        se = ((std_a ** 2 / len(values_a)) + (std_b ** 2 / len(values_b))) ** 0.5

        if se == 0:
            return 0.0

        # t-statistic
        t = abs(mean_a - mean_b) / se

        # Approximate p-value (simplified)
        # For large samples, t > 2 suggests p < 0.05, t > 2.5 suggests p < 0.01
        if t > 3:
            return 0.99
        elif t > 2.5:
            return 0.95
        elif t > 2:
            return 0.90
        elif t > 1.5:
            return 0.80
        else:
            return 0.5 + (t / 3) * 0.3

    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a test"""
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            return {
                "id": test_id,
                "name": test["name"],
                "status": test["status"],
                "samples_a": len(test["results_a"]),
                "samples_b": len(test["results_b"]),
                "min_samples": test["min_samples"],
                "progress": min(
                    len(test["results_a"]) / test["min_samples"],
                    len(test["results_b"]) / test["min_samples"]
                )
            }

        for result in self.completed_tests:
            if result.id == test_id:
                return {
                    "id": test_id,
                    "name": result.name,
                    "status": "completed",
                    "winner": result.winner,
                    "confidence": result.confidence,
                    "recommendation": result.recommendation
                }

        return None


class ContinuousLearning:
    """
    Main orchestrator for the continuous learning loop.

    Coordinates pattern recognition, estimation, post-mortems, and A/B testing.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None

        self.patterns = PatternRecognition(
            str(self.storage_path / "patterns") if self.storage_path else None
        )
        self.estimator = SprintEstimator()
        self.post_mortem = PostMortemAnalyzer(
            str(self.storage_path / "post_mortem") if self.storage_path else None
        )
        self.ab_testing = ABTestEngine()

        self.events: List[LearningEvent] = []

        logger.info("ContinuousLearning system initialized")

    def record_event(
        self,
        event_type: LearningEventType,
        context: Dict[str, Any],
        outcome: str,
        metrics: Optional[Dict[str, float]] = None,
        lessons: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> LearningEvent:
        """Record a learning event"""
        event = LearningEvent(
            id=f"event_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(),
            event_type=event_type,
            context=context,
            outcome=outcome,
            metrics=metrics or {},
            lessons=lessons or [],
            tags=tags or []
        )

        self.events.append(event)

        # Record pattern if successful
        if outcome == "success" and "pattern" in context:
            self.patterns.record_occurrence(
                context["pattern"],
                PatternType.CODE_STRUCTURE,
                str(context),
                True,
                context.get("code_example")
            )

        return event

    def get_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on all learning sources"""
        recommendations = {
            "patterns": [],
            "estimation": {},
            "lessons": [],
            "ab_test_insights": []
        }

        # Get pattern recommendations
        if "task_type" in context:
            pattern_suggestion = self.patterns.suggest_pattern(
                str(context),
                PatternType.CODE_STRUCTURE
            )
            if pattern_suggestion:
                recommendations["patterns"].append(pattern_suggestion)

        # Get estimation insights
        if "complexity_factors" in context:
            recommendations["estimation"] = self.estimator.estimate_sprint(
                context.get("base_hours", 40),
                context.get("complexity_factors", [])
            )

        # Get relevant lessons
        if "keywords" in context:
            query = " ".join(context["keywords"])
            recommendations["lessons"] = self.post_mortem.query_knowledge(query)

        # Get A/B test insights
        for result in self.ab_testing.completed_tests:
            if result.winner != "inconclusive":
                recommendations["ab_test_insights"].append({
                    "test": result.name,
                    "winner": result.winner,
                    "confidence": result.confidence
                })

        return recommendations

    def conduct_project_retrospective(
        self,
        project_name: str,
        metrics: Dict[str, Any]
    ) -> PostMortemReport:
        """Conduct a full project retrospective"""
        # Get recent events for this project
        project_events = [
            e for e in self.events
            if project_name.lower() in str(e.context).lower()
        ]

        return self.post_mortem.conduct_post_mortem(
            project_name,
            metrics,
            project_events
        )

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning data"""
        return {
            "total_events": len(self.events),
            "patterns_recognized": len(self.patterns.patterns),
            "successful_patterns": len(self.patterns.get_successful_patterns()),
            "sprints_estimated": len(self.estimator.estimates),
            "estimation_insights": self.estimator.get_estimation_insights(),
            "post_mortems_conducted": len(self.post_mortem.reports),
            "ab_tests_completed": len(self.ab_testing.completed_tests),
            "knowledge_base_size": sum(len(v) for v in self.post_mortem.knowledge_base.values())
        }
