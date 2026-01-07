"""
Business Intelligence Module

Provides business intelligence capabilities:
- ROI calculator for time/cost saved
- Velocity tracking and burndown charts
- Quality metrics dashboard
- Predictive analytics for project completion
"""

import json
import logging
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricPeriod(Enum):
    """Time periods for metrics"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class ROIMetrics:
    """ROI calculation metrics"""
    id: str
    period_start: datetime
    period_end: datetime
    traditional_hours: float
    ai_assisted_hours: float
    hourly_rate: float
    time_saved_hours: float
    cost_saved: float
    roi_percentage: float
    tasks_completed: int
    quality_score: float


@dataclass
class VelocityPoint:
    """Single velocity measurement"""
    sprint_id: str
    date: datetime
    story_points_committed: int
    story_points_completed: int
    tasks_committed: int
    tasks_completed: int
    team_size: int


@dataclass
class QualityMetric:
    """Quality metric data point"""
    id: str
    timestamp: datetime
    metric_name: str
    value: float
    target: float
    status: str  # good, warning, critical
    details: Dict[str, Any]


@dataclass
class ProjectPrediction:
    """Project completion prediction"""
    id: str
    timestamp: datetime
    predicted_completion: datetime
    confidence: float
    remaining_work: float
    current_velocity: float
    risk_factors: List[str]
    scenarios: Dict[str, datetime]


class ROICalculator:
    """
    Calculates ROI showing time and cost saved vs traditional development.
    """

    # Industry benchmarks (hours per task type)
    TRADITIONAL_BENCHMARKS = {
        "feature_small": 8,
        "feature_medium": 24,
        "feature_large": 80,
        "bug_fix_simple": 2,
        "bug_fix_complex": 8,
        "refactoring": 16,
        "documentation": 4,
        "testing": 8,
        "code_review": 2,
        "deployment": 4
    }

    def __init__(self, hourly_rate: float = 100.0):
        self.hourly_rate = hourly_rate
        self.calculations: List[ROIMetrics] = []
        self.task_history: List[Dict[str, Any]] = []

        logger.info(f"ROICalculator initialized (hourly_rate=${hourly_rate})")

    def record_task(
        self,
        task_type: str,
        actual_hours: float,
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record a completed task"""
        traditional_hours = self.TRADITIONAL_BENCHMARKS.get(task_type, 8)

        task = {
            "id": f"task_{uuid.uuid4().hex[:12]}",
            "type": task_type,
            "traditional_hours": traditional_hours,
            "actual_hours": actual_hours,
            "time_saved": traditional_hours - actual_hours,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.task_history.append(task)
        return task

    def calculate_roi(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> ROIMetrics:
        """Calculate ROI for a period"""
        if period_start is None:
            period_start = datetime.now() - timedelta(days=30)
        if period_end is None:
            period_end = datetime.now()

        # Filter tasks in period
        tasks = [
            t for t in self.task_history
            if period_start <= datetime.fromisoformat(t["timestamp"]) <= period_end
        ]

        if not tasks:
            return ROIMetrics(
                id=f"roi_{uuid.uuid4().hex[:8]}",
                period_start=period_start,
                period_end=period_end,
                traditional_hours=0,
                ai_assisted_hours=0,
                hourly_rate=self.hourly_rate,
                time_saved_hours=0,
                cost_saved=0,
                roi_percentage=0,
                tasks_completed=0,
                quality_score=0
            )

        traditional_hours = sum(t["traditional_hours"] for t in tasks)
        actual_hours = sum(t["actual_hours"] for t in tasks)
        time_saved = sum(t["time_saved"] for t in tasks)
        avg_quality = statistics.mean(t["quality_score"] for t in tasks)

        cost_saved = time_saved * self.hourly_rate
        roi_percentage = (time_saved / max(actual_hours, 1)) * 100

        metrics = ROIMetrics(
            id=f"roi_{uuid.uuid4().hex[:8]}",
            period_start=period_start,
            period_end=period_end,
            traditional_hours=traditional_hours,
            ai_assisted_hours=actual_hours,
            hourly_rate=self.hourly_rate,
            time_saved_hours=time_saved,
            cost_saved=cost_saved,
            roi_percentage=roi_percentage,
            tasks_completed=len(tasks),
            quality_score=avg_quality
        )

        self.calculations.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get overall ROI summary"""
        if not self.task_history:
            return {"message": "No tasks recorded"}

        total_traditional = sum(t["traditional_hours"] for t in self.task_history)
        total_actual = sum(t["actual_hours"] for t in self.task_history)
        total_saved = sum(t["time_saved"] for t in self.task_history)

        # Group by task type
        by_type = defaultdict(lambda: {"traditional": 0, "actual": 0, "count": 0})
        for task in self.task_history:
            by_type[task["type"]]["traditional"] += task["traditional_hours"]
            by_type[task["type"]]["actual"] += task["actual_hours"]
            by_type[task["type"]]["count"] += 1

        return {
            "total_tasks": len(self.task_history),
            "total_traditional_hours": round(total_traditional, 1),
            "total_ai_assisted_hours": round(total_actual, 1),
            "total_time_saved_hours": round(total_saved, 1),
            "total_cost_saved": round(total_saved * self.hourly_rate, 2),
            "average_time_reduction": round((total_saved / max(total_traditional, 1)) * 100, 1),
            "by_task_type": {
                k: {
                    "count": v["count"],
                    "efficiency_gain": round(
                        (v["traditional"] - v["actual"]) / max(v["traditional"], 1) * 100, 1
                    )
                }
                for k, v in by_type.items()
            }
        }

    def generate_report(
        self,
        format: str = "markdown"
    ) -> str:
        """Generate an ROI report"""
        summary = self.get_summary()

        if format == "markdown":
            report = f"""# ROI Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | {summary.get('total_tasks', 0)} |
| Traditional Hours | {summary.get('total_traditional_hours', 0)} |
| AI-Assisted Hours | {summary.get('total_ai_assisted_hours', 0)} |
| Time Saved | {summary.get('total_time_saved_hours', 0)} hours |
| Cost Saved | ${summary.get('total_cost_saved', 0):,.2f} |
| Average Time Reduction | {summary.get('average_time_reduction', 0)}% |

## By Task Type

"""
            for task_type, data in summary.get('by_task_type', {}).items():
                report += f"- **{task_type}**: {data['count']} tasks, {data['efficiency_gain']}% efficiency gain\n"

            return report

        return json.dumps(summary, indent=2)


class VelocityTracker:
    """
    Tracks team velocity and generates burndown charts.
    """

    def __init__(self):
        self.velocity_points: List[VelocityPoint] = []
        self.sprints: Dict[str, Dict[str, Any]] = {}

        logger.info("VelocityTracker initialized")

    def start_sprint(
        self,
        sprint_id: str,
        story_points: int,
        tasks: int,
        team_size: int,
        duration_days: int = 14
    ) -> Dict[str, Any]:
        """Start a new sprint"""
        sprint = {
            "id": sprint_id,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "story_points_committed": story_points,
            "story_points_completed": 0,
            "tasks_committed": tasks,
            "tasks_completed": 0,
            "team_size": team_size,
            "status": "active"
        }

        self.sprints[sprint_id] = sprint
        logger.info(f"Started sprint: {sprint_id}")

        return sprint

    def record_progress(
        self,
        sprint_id: str,
        story_points_completed: int,
        tasks_completed: int
    ) -> VelocityPoint:
        """Record daily progress"""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint {sprint_id} not found")

        sprint["story_points_completed"] = story_points_completed
        sprint["tasks_completed"] = tasks_completed

        point = VelocityPoint(
            sprint_id=sprint_id,
            date=datetime.now(),
            story_points_committed=sprint["story_points_committed"],
            story_points_completed=story_points_completed,
            tasks_committed=sprint["tasks_committed"],
            tasks_completed=tasks_completed,
            team_size=sprint["team_size"]
        )

        self.velocity_points.append(point)
        return point

    def complete_sprint(self, sprint_id: str) -> Dict[str, Any]:
        """Complete a sprint"""
        if sprint_id not in self.sprints:
            raise ValueError(f"Sprint {sprint_id} not found")

        sprint = self.sprints[sprint_id]
        sprint["status"] = "completed"
        sprint["actual_end_date"] = datetime.now().isoformat()

        # Calculate velocity
        sprint["velocity"] = sprint["story_points_completed"] / max(sprint["team_size"], 1)

        return sprint

    def get_average_velocity(self, last_n_sprints: int = 5) -> float:
        """Get average velocity from recent sprints"""
        completed = [
            s for s in self.sprints.values()
            if s.get("status") == "completed"
        ]

        if not completed:
            return 0.0

        recent = sorted(
            completed,
            key=lambda x: x.get("actual_end_date", ""),
            reverse=True
        )[:last_n_sprints]

        velocities = [s.get("velocity", 0) for s in recent]
        return statistics.mean(velocities) if velocities else 0.0

    def get_burndown_data(self, sprint_id: str) -> Dict[str, Any]:
        """Get burndown chart data for a sprint"""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            return {"error": "Sprint not found"}

        points = [
            p for p in self.velocity_points
            if p.sprint_id == sprint_id
        ]

        if not points:
            return {
                "sprint_id": sprint_id,
                "data_points": [],
                "ideal_burndown": []
            }

        # Sort by date
        points.sort(key=lambda x: x.date)

        # Calculate ideal burndown
        total_points = sprint["story_points_committed"]
        start_date = datetime.fromisoformat(sprint["start_date"])
        end_date = datetime.fromisoformat(sprint["end_date"])
        total_days = (end_date - start_date).days

        ideal = []
        for i in range(total_days + 1):
            date = start_date + timedelta(days=i)
            remaining = total_points * (1 - i / total_days)
            ideal.append({
                "date": date.strftime("%Y-%m-%d"),
                "remaining": round(remaining, 1)
            })

        # Actual burndown
        actual = []
        for point in points:
            remaining = sprint["story_points_committed"] - point.story_points_completed
            actual.append({
                "date": point.date.strftime("%Y-%m-%d"),
                "remaining": remaining
            })

        return {
            "sprint_id": sprint_id,
            "total_points": total_points,
            "ideal_burndown": ideal,
            "actual_burndown": actual,
            "on_track": len(actual) > 0 and actual[-1]["remaining"] <= ideal[-1]["remaining"]
        }

    def get_velocity_trend(self) -> Dict[str, Any]:
        """Get velocity trend over sprints"""
        completed = [
            s for s in self.sprints.values()
            if s.get("status") == "completed"
        ]

        if len(completed) < 2:
            return {"trend": "insufficient_data"}

        velocities = [s.get("velocity", 0) for s in sorted(
            completed,
            key=lambda x: x.get("actual_end_date", "")
        )]

        # Calculate trend
        if velocities[-1] > velocities[0] * 1.1:
            trend = "improving"
        elif velocities[-1] < velocities[0] * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "velocities": velocities,
            "average": round(statistics.mean(velocities), 2),
            "latest": velocities[-1] if velocities else 0
        }


class QualityMetrics:
    """
    Tracks code quality metrics and technical debt.
    """

    # Thresholds for quality metrics
    THRESHOLDS = {
        "test_coverage": {"warning": 60, "critical": 40, "target": 80},
        "code_complexity": {"warning": 15, "critical": 25, "target": 10},
        "technical_debt_hours": {"warning": 40, "critical": 80, "target": 20},
        "bugs_per_sprint": {"warning": 5, "critical": 10, "target": 2},
        "code_duplication": {"warning": 10, "critical": 20, "target": 5}
    }

    def __init__(self):
        self.metrics: List[QualityMetric] = []

        logger.info("QualityMetrics initialized")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        details: Optional[Dict[str, Any]] = None
    ) -> QualityMetric:
        """Record a quality metric"""
        thresholds = self.THRESHOLDS.get(metric_name, {
            "warning": value * 1.2,
            "critical": value * 1.5,
            "target": value * 0.8
        })

        # Determine status (lower is better for most metrics except coverage)
        if metric_name in ["test_coverage"]:
            # Higher is better
            if value >= thresholds["target"]:
                status = "good"
            elif value >= thresholds["warning"]:
                status = "warning"
            else:
                status = "critical"
        else:
            # Lower is better
            if value <= thresholds["target"]:
                status = "good"
            elif value <= thresholds["warning"]:
                status = "warning"
            else:
                status = "critical"

        metric = QualityMetric(
            id=f"qm_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            target=thresholds["target"],
            status=status,
            details=details or {}
        )

        self.metrics.append(metric)
        return metric

    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of all quality metrics"""
        # Get most recent value for each metric
        latest = {}
        for metric in reversed(self.metrics):
            if metric.metric_name not in latest:
                latest[metric.metric_name] = {
                    "value": metric.value,
                    "target": metric.target,
                    "status": metric.status,
                    "timestamp": metric.timestamp.isoformat()
                }

        # Calculate overall health
        statuses = [m["status"] for m in latest.values()]
        if "critical" in statuses:
            overall = "critical"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "good"

        return {
            "overall_status": overall,
            "metrics": latest,
            "metrics_count": len(latest),
            "critical_count": statuses.count("critical"),
            "warning_count": statuses.count("warning")
        }

    def get_metric_trend(
        self,
        metric_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get trend for a specific metric"""
        since = datetime.now() - timedelta(days=days)
        values = [
            m for m in self.metrics
            if m.metric_name == metric_name and m.timestamp >= since
        ]

        if not values:
            return {"message": "No data available"}

        data_points = [{"date": m.timestamp.isoformat(), "value": m.value} for m in values]

        # Calculate trend
        if len(values) >= 2:
            if values[-1].value > values[0].value * 1.1:
                trend = "increasing"
            elif values[-1].value < values[0].value * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "metric": metric_name,
            "trend": trend,
            "current_value": values[-1].value,
            "target": values[-1].target,
            "data_points": data_points[-20:]  # Last 20 points
        }

    def calculate_technical_debt(self) -> Dict[str, Any]:
        """Calculate technical debt score"""
        current = self.get_current_status()
        metrics = current.get("metrics", {})

        debt_factors = {
            "code_complexity": 2,
            "code_duplication": 3,
            "test_coverage": 4,  # Inverse impact
            "technical_debt_hours": 1,
            "bugs_per_sprint": 2
        }

        debt_score = 0
        components = []

        for metric_name, factor in debt_factors.items():
            if metric_name in metrics:
                metric = metrics[metric_name]
                value = metric["value"]
                target = metric["target"]

                if metric_name == "test_coverage":
                    # Lower coverage = more debt
                    deviation = max(0, target - value) / target
                else:
                    # Higher value = more debt
                    deviation = max(0, value - target) / max(target, 1)

                component_debt = deviation * factor * 10
                debt_score += component_debt

                components.append({
                    "metric": metric_name,
                    "contribution": round(component_debt, 2),
                    "status": metric["status"]
                })

        return {
            "total_debt_score": round(debt_score, 2),
            "components": sorted(components, key=lambda x: x["contribution"], reverse=True),
            "health_grade": self._get_grade(debt_score),
            "recommendations": self._get_recommendations(components)
        }

    def _get_grade(self, debt_score: float) -> str:
        """Convert debt score to letter grade"""
        if debt_score < 10:
            return "A"
        elif debt_score < 25:
            return "B"
        elif debt_score < 50:
            return "C"
        elif debt_score < 75:
            return "D"
        else:
            return "F"

    def _get_recommendations(
        self,
        components: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on debt components"""
        recommendations = []

        for component in components[:3]:  # Top 3 contributors
            metric = component["metric"]

            if metric == "test_coverage" and component["status"] != "good":
                recommendations.append("Increase test coverage to reduce risk")
            elif metric == "code_complexity" and component["status"] != "good":
                recommendations.append("Refactor complex functions to improve maintainability")
            elif metric == "code_duplication" and component["status"] != "good":
                recommendations.append("Extract duplicated code into shared utilities")
            elif metric == "bugs_per_sprint" and component["status"] != "good":
                recommendations.append("Invest in code review and testing to reduce bugs")

        return recommendations


class PredictiveAnalytics:
    """
    Predicts project completion dates based on historical data.
    """

    def __init__(
        self,
        velocity_tracker: VelocityTracker,
        roi_calculator: ROICalculator
    ):
        self.velocity = velocity_tracker
        self.roi = roi_calculator
        self.predictions: List[ProjectPrediction] = []

        logger.info("PredictiveAnalytics initialized")

    def predict_completion(
        self,
        remaining_story_points: int,
        team_size: int,
        risk_factors: Optional[List[str]] = None
    ) -> ProjectPrediction:
        """Predict project completion date"""
        # Get average velocity
        avg_velocity = self.velocity.get_average_velocity()

        if avg_velocity == 0:
            avg_velocity = 5  # Default velocity

        # Calculate days needed
        team_velocity = avg_velocity * team_size
        sprints_needed = remaining_story_points / max(team_velocity, 1)
        days_needed = sprints_needed * 14  # 2-week sprints

        # Apply risk factors
        risk_multiplier = 1.0
        identified_risks = risk_factors or []

        if "new_technology" in identified_risks:
            risk_multiplier *= 1.2
        if "team_changes" in identified_risks:
            risk_multiplier *= 1.15
        if "unclear_requirements" in identified_risks:
            risk_multiplier *= 1.3
        if "external_dependencies" in identified_risks:
            risk_multiplier *= 1.1

        adjusted_days = days_needed * risk_multiplier

        # Calculate confidence
        velocity_trend = self.velocity.get_velocity_trend()
        if velocity_trend.get("trend") == "improving":
            confidence = 0.8
        elif velocity_trend.get("trend") == "declining":
            confidence = 0.5
        else:
            confidence = 0.7

        # Reduce confidence for higher risk
        confidence *= 1 / risk_multiplier

        # Generate scenarios
        predicted_date = datetime.now() + timedelta(days=adjusted_days)
        optimistic = datetime.now() + timedelta(days=days_needed * 0.8)
        pessimistic = datetime.now() + timedelta(days=days_needed * 1.5)

        prediction = ProjectPrediction(
            id=f"pred_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            predicted_completion=predicted_date,
            confidence=round(confidence, 2),
            remaining_work=remaining_story_points,
            current_velocity=avg_velocity,
            risk_factors=identified_risks,
            scenarios={
                "optimistic": optimistic,
                "realistic": predicted_date,
                "pessimistic": pessimistic
            }
        )

        self.predictions.append(prediction)
        return prediction

    def get_sprint_forecast(
        self,
        num_sprints: int = 5
    ) -> List[Dict[str, Any]]:
        """Forecast performance for upcoming sprints"""
        avg_velocity = self.velocity.get_average_velocity()
        trend = self.velocity.get_velocity_trend()

        # Apply trend adjustment
        if trend.get("trend") == "improving":
            trend_factor = 1.05
        elif trend.get("trend") == "declining":
            trend_factor = 0.95
        else:
            trend_factor = 1.0

        forecasts = []
        current_velocity = avg_velocity

        for i in range(num_sprints):
            sprint_num = i + 1
            projected_velocity = current_velocity * (trend_factor ** sprint_num)

            forecasts.append({
                "sprint": sprint_num,
                "projected_velocity": round(projected_velocity, 1),
                "projected_points": round(projected_velocity * 14, 0),  # 2-week sprint
                "confidence": round(0.9 - (0.1 * i), 2)  # Decreasing confidence
            })

        return forecasts

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze potential bottlenecks"""
        bottlenecks = []

        # Analyze task types from ROI data
        roi_summary = self.roi.get_summary()
        by_type = roi_summary.get("by_task_type", {})

        for task_type, data in by_type.items():
            if data.get("efficiency_gain", 0) < 20:
                bottlenecks.append({
                    "type": "task_efficiency",
                    "component": task_type,
                    "description": f"{task_type} tasks have low efficiency gain",
                    "impact": "medium",
                    "suggestion": f"Review {task_type} process for optimization"
                })

        # Analyze velocity trend
        velocity_trend = self.velocity.get_velocity_trend()
        if velocity_trend.get("trend") == "declining":
            bottlenecks.append({
                "type": "velocity",
                "component": "team",
                "description": "Team velocity is declining",
                "impact": "high",
                "suggestion": "Investigate blockers and team capacity"
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "bottlenecks_found": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "overall_health": "good" if len(bottlenecks) == 0 else (
                "warning" if len(bottlenecks) < 3 else "critical"
            )
        }


class BusinessIntelligence:
    """
    Main orchestrator for all business intelligence features.
    """

    def __init__(self, hourly_rate: float = 100.0):
        self.roi_calculator = ROICalculator(hourly_rate)
        self.velocity_tracker = VelocityTracker()
        self.quality_metrics = QualityMetrics()
        self.predictive = PredictiveAnalytics(
            self.velocity_tracker,
            self.roi_calculator
        )

        logger.info("BusinessIntelligence system initialized")

    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary of all metrics"""
        roi = self.roi_calculator.get_summary()
        velocity = self.velocity_tracker.get_velocity_trend()
        quality = self.quality_metrics.get_current_status()
        bottlenecks = self.predictive.analyze_bottlenecks()

        return {
            "generated_at": datetime.now().isoformat(),
            "roi": {
                "total_cost_saved": roi.get("total_cost_saved", 0),
                "time_reduction": f"{roi.get('average_time_reduction', 0)}%",
                "tasks_completed": roi.get("total_tasks", 0)
            },
            "velocity": {
                "trend": velocity.get("trend", "unknown"),
                "average": velocity.get("average", 0)
            },
            "quality": {
                "overall_status": quality.get("overall_status", "unknown"),
                "critical_issues": quality.get("critical_count", 0)
            },
            "risks": {
                "bottleneck_count": bottlenecks.get("bottlenecks_found", 0),
                "health": bottlenecks.get("overall_health", "unknown")
            },
            "recommendations": self._generate_recommendations(
                roi, velocity, quality, bottlenecks
            )
        }

    def _generate_recommendations(
        self,
        roi: Dict[str, Any],
        velocity: Dict[str, Any],
        quality: Dict[str, Any],
        bottlenecks: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # ROI recommendations
        if roi.get("average_time_reduction", 0) < 30:
            recommendations.append(
                "Consider automating more repetitive tasks to improve efficiency"
            )

        # Velocity recommendations
        if velocity.get("trend") == "declining":
            recommendations.append(
                "Review sprint planning and remove blockers to restore velocity"
            )

        # Quality recommendations
        if quality.get("critical_count", 0) > 0:
            recommendations.append(
                "Address critical quality issues to prevent technical debt"
            )

        # Bottleneck recommendations
        for bottleneck in bottlenecks.get("bottlenecks", [])[:2]:
            recommendations.append(bottleneck.get("suggestion", ""))

        return recommendations[:5]

    def export_dashboard_data(
        self,
        format: str = "json"
    ) -> str:
        """Export dashboard data"""
        data = {
            "summary": self.get_executive_summary(),
            "roi_details": self.roi_calculator.get_summary(),
            "velocity_data": self.velocity_tracker.get_velocity_trend(),
            "quality_status": self.quality_metrics.get_current_status(),
            "technical_debt": self.quality_metrics.calculate_technical_debt(),
            "forecasts": self.predictive.get_sprint_forecast()
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)

        return json.dumps(data, default=str)
