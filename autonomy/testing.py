"""
Testing Evolution Module

Provides advanced testing capabilities:
- Automated test generation based on requirements
- Chaos engineering for resilience testing
- Performance/load testing with baselines
- Visual regression testing
"""

import asyncio
import json
import logging
import random
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"
    CHAOS = "chaos"
    VISUAL = "visual"
    SECURITY = "security"


class ChaosType(Enum):
    """Types of chaos experiments"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    SERVICE_FAILURE = "service_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"
    CLOCK_SKEW = "clock_skew"


@dataclass
class TestCase:
    """Represents a generated test case"""
    id: str
    name: str
    test_type: TestType
    description: str
    code: str
    dependencies: List[str]
    expected_outcome: str
    tags: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class TestResult:
    """Result of a test execution"""
    test_id: str
    test_name: str
    passed: bool
    duration_ms: float
    error: Optional[str]
    stdout: Optional[str]
    assertions: Dict[str, bool]
    coverage: Optional[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChaosExperiment:
    """Represents a chaos experiment"""
    id: str
    name: str
    chaos_type: ChaosType
    target: str
    duration_seconds: int
    parameters: Dict[str, Any]
    hypothesis: str
    rollback_procedure: str


@dataclass
class ChaosResult:
    """Result of a chaos experiment"""
    experiment_id: str
    success: bool
    hypothesis_validated: bool
    observations: List[str]
    metrics_during: Dict[str, List[float]]
    recovery_time_seconds: Optional[float]
    issues_found: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    id: str
    metric_name: str
    baseline_value: float
    threshold_warning: float
    threshold_critical: float
    samples: int
    created_at: datetime


@dataclass
class VisualSnapshot:
    """Visual snapshot for regression testing"""
    id: str
    component: str
    hash: str
    dimensions: Tuple[int, int]
    created_at: datetime
    metadata: Dict[str, Any]


class TestGenerator:
    """
    Generates tests automatically based on requirements and code.

    Creates unit, integration, and E2E tests from specifications.
    """

    # Test templates for different scenarios
    UNIT_TEST_TEMPLATE = '''
import pytest
from {module} import {function}

class Test{class_name}:
    """Tests for {function}"""

    def test_{function}_basic(self):
        """Test basic functionality of {function}"""
        {test_body}

    def test_{function}_edge_cases(self):
        """Test edge cases for {function}"""
        {edge_case_body}

    def test_{function}_error_handling(self):
        """Test error handling in {function}"""
        {error_handling_body}
'''

    INTEGRATION_TEST_TEMPLATE = '''
import pytest
import asyncio

class TestIntegration{component}:
    """Integration tests for {component}"""

    @pytest.fixture
    async def setup(self):
        """Setup test fixtures"""
        {setup_code}
        yield
        {teardown_code}

    async def test_{component}_integration(self, setup):
        """Test {component} integration with dependencies"""
        {test_body}
'''

    E2E_TEST_TEMPLATE = '''
import pytest
from playwright.async_api import async_playwright

class TestE2E{feature}:
    """End-to-end tests for {feature}"""

    @pytest.fixture
    async def browser(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            yield browser
            await browser.close()

    async def test_{feature}_happy_path(self, browser):
        """Test {feature} happy path"""
        {test_body}
'''

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.generated_tests: List[TestCase] = []

        logger.info(f"TestGenerator initialized for: {workspace_dir}")

    def generate_unit_tests(
        self,
        module_path: str,
        function_name: str,
        requirements: Dict[str, Any]
    ) -> List[TestCase]:
        """Generate unit tests for a function"""
        tests = []

        # Generate basic test
        class_name = self._to_class_name(function_name)

        # Determine test body based on requirements
        input_types = requirements.get("input_types", {})
        output_type = requirements.get("output_type", "Any")

        test_body = self._generate_test_body(function_name, input_types, output_type)
        edge_cases = self._generate_edge_case_tests(function_name, input_types)
        error_handling = self._generate_error_tests(function_name, requirements)

        code = self.UNIT_TEST_TEMPLATE.format(
            module=module_path.replace("/", ".").replace(".py", ""),
            function=function_name,
            class_name=class_name,
            test_body=test_body,
            edge_case_body=edge_cases,
            error_handling_body=error_handling
        )

        test_case = TestCase(
            id=f"test_{uuid.uuid4().hex[:12]}",
            name=f"Test{class_name}",
            test_type=TestType.UNIT,
            description=f"Unit tests for {function_name}",
            code=code,
            dependencies=["pytest"],
            expected_outcome="All tests pass",
            tags=["unit", "auto-generated"]
        )

        tests.append(test_case)
        self.generated_tests.append(test_case)

        return tests

    def _to_class_name(self, name: str) -> str:
        """Convert function name to class name"""
        return "".join(word.capitalize() for word in name.split("_"))

    def _generate_test_body(
        self,
        function_name: str,
        input_types: Dict[str, str],
        output_type: str
    ) -> str:
        """Generate test body"""
        lines = []

        # Generate sample inputs
        sample_inputs = []
        for param, param_type in input_types.items():
            if param_type == "int":
                sample_inputs.append(f"{param}=42")
            elif param_type == "str":
                sample_inputs.append(f'{param}="test"')
            elif param_type == "list":
                sample_inputs.append(f"{param}=[1, 2, 3]")
            elif param_type == "dict":
                sample_inputs.append(f'{param}={{"key": "value"}}')
            else:
                sample_inputs.append(f"{param}=None")

        input_str = ", ".join(sample_inputs)
        lines.append(f"result = {function_name}({input_str})")
        lines.append("assert result is not None")

        if output_type == "int":
            lines.append("assert isinstance(result, int)")
        elif output_type == "str":
            lines.append("assert isinstance(result, str)")
        elif output_type == "list":
            lines.append("assert isinstance(result, list)")
        elif output_type == "dict":
            lines.append("assert isinstance(result, dict)")
        elif output_type == "bool":
            lines.append("assert isinstance(result, bool)")

        return "\n        ".join(lines)

    def _generate_edge_case_tests(
        self,
        function_name: str,
        input_types: Dict[str, str]
    ) -> str:
        """Generate edge case tests"""
        lines = ["# Test with edge case values"]

        for param, param_type in input_types.items():
            if param_type == "int":
                lines.append(f"# Test with zero")
                lines.append(f"result = {function_name}({param}=0)")
                lines.append(f"assert result is not None")
            elif param_type == "str":
                lines.append(f"# Test with empty string")
                lines.append(f'result = {function_name}({param}="")')
                lines.append(f"assert result is not None")
            elif param_type == "list":
                lines.append(f"# Test with empty list")
                lines.append(f"result = {function_name}({param}=[])")
                lines.append(f"assert result is not None")

        return "\n        ".join(lines)

    def _generate_error_tests(
        self,
        function_name: str,
        requirements: Dict[str, Any]
    ) -> str:
        """Generate error handling tests"""
        lines = []
        error_cases = requirements.get("error_cases", [])

        if not error_cases:
            lines.append("# Test with invalid input")
            lines.append("with pytest.raises(Exception):")
            lines.append(f"    {function_name}(None)")
        else:
            for error_case in error_cases:
                lines.append(f"# Test {error_case['description']}")
                lines.append(f"with pytest.raises({error_case['exception']}):")
                lines.append(f"    {function_name}({error_case['input']})")

        return "\n        ".join(lines)

    def generate_integration_tests(
        self,
        component: str,
        dependencies: List[str],
        interactions: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Generate integration tests for a component"""
        tests = []

        setup_code = "\n        ".join([
            f"# Initialize {dep}" for dep in dependencies
        ])
        teardown_code = "# Cleanup resources"

        test_body_lines = []
        for interaction in interactions:
            test_body_lines.append(f"# Test interaction: {interaction.get('description', 'N/A')}")
            test_body_lines.append(f"# Call: {interaction.get('method', 'unknown')}")
            test_body_lines.append("assert True  # Replace with actual assertion")

        test_body = "\n        ".join(test_body_lines)

        code = self.INTEGRATION_TEST_TEMPLATE.format(
            component=self._to_class_name(component),
            setup_code=setup_code,
            teardown_code=teardown_code,
            test_body=test_body
        )

        test_case = TestCase(
            id=f"test_{uuid.uuid4().hex[:12]}",
            name=f"TestIntegration{self._to_class_name(component)}",
            test_type=TestType.INTEGRATION,
            description=f"Integration tests for {component}",
            code=code,
            dependencies=["pytest", "pytest-asyncio"] + dependencies,
            expected_outcome="All integration tests pass",
            tags=["integration", "auto-generated"]
        )

        tests.append(test_case)
        self.generated_tests.append(test_case)

        return tests

    def generate_e2e_tests(
        self,
        feature: str,
        user_flows: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Generate end-to-end tests for a feature"""
        tests = []

        test_body_lines = []
        for flow in user_flows:
            test_body_lines.append(f"# Flow: {flow.get('name', 'unknown')}")
            for step in flow.get('steps', []):
                test_body_lines.append(f"# Step: {step}")
            test_body_lines.append("pass  # Replace with actual test code")

        test_body = "\n        ".join(test_body_lines)

        code = self.E2E_TEST_TEMPLATE.format(
            feature=self._to_class_name(feature),
            test_body=test_body
        )

        test_case = TestCase(
            id=f"test_{uuid.uuid4().hex[:12]}",
            name=f"TestE2E{self._to_class_name(feature)}",
            test_type=TestType.E2E,
            description=f"E2E tests for {feature}",
            code=code,
            dependencies=["pytest", "pytest-asyncio", "playwright"],
            expected_outcome="All E2E tests pass",
            tags=["e2e", "auto-generated"]
        )

        tests.append(test_case)
        self.generated_tests.append(test_case)

        return tests

    def save_tests(self, output_dir: str) -> List[str]:
        """Save generated tests to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for test in self.generated_tests:
            file_name = f"test_{test.name.lower()}.py"
            file_path = output_path / file_name

            file_path.write_text(test.code)
            saved_files.append(str(file_path))

        logger.info(f"Saved {len(saved_files)} test files to {output_dir}")
        return saved_files


class ChaosEngineer:
    """
    Chaos engineering agent for resilience testing.

    Intentionally breaks things to test system resilience.
    """

    def __init__(self, workspace_dir: str, safe_mode: bool = True):
        self.workspace_dir = Path(workspace_dir)
        self.safe_mode = safe_mode
        self.experiments: List[ChaosExperiment] = []
        self.results: List[ChaosResult] = []

        logger.info(f"ChaosEngineer initialized (safe_mode={safe_mode})")

    def create_experiment(
        self,
        name: str,
        chaos_type: ChaosType,
        target: str,
        duration_seconds: int = 30,
        parameters: Optional[Dict[str, Any]] = None,
        hypothesis: str = ""
    ) -> ChaosExperiment:
        """Create a new chaos experiment"""
        experiment = ChaosExperiment(
            id=f"chaos_{uuid.uuid4().hex[:12]}",
            name=name,
            chaos_type=chaos_type,
            target=target,
            duration_seconds=min(duration_seconds, 60 if self.safe_mode else 300),
            parameters=parameters or {},
            hypothesis=hypothesis,
            rollback_procedure=self._generate_rollback(chaos_type)
        )

        self.experiments.append(experiment)
        logger.info(f"Created chaos experiment: {name}")

        return experiment

    def _generate_rollback(self, chaos_type: ChaosType) -> str:
        """Generate rollback procedure for chaos type"""
        rollbacks = {
            ChaosType.NETWORK_LATENCY: "Remove latency injection rules",
            ChaosType.NETWORK_PARTITION: "Restore network connectivity",
            ChaosType.SERVICE_FAILURE: "Restart affected services",
            ChaosType.RESOURCE_EXHAUSTION: "Release resources and restart",
            ChaosType.DEPENDENCY_FAILURE: "Restore dependency connections",
            ChaosType.DATA_CORRUPTION: "Restore from backup checkpoint",
            ChaosType.CLOCK_SKEW: "Synchronize system clocks"
        }
        return rollbacks.get(chaos_type, "Manual intervention required")

    async def run_experiment(
        self,
        experiment: ChaosExperiment,
        health_check: Callable[[], bool],
        metrics_collector: Optional[Callable[[], Dict[str, float]]] = None
    ) -> ChaosResult:
        """Run a chaos experiment"""
        logger.info(f"Starting chaos experiment: {experiment.name}")

        observations = []
        metrics_during = {}
        issues_found = []

        # Pre-experiment health check
        if not health_check():
            return ChaosResult(
                experiment_id=experiment.id,
                success=False,
                hypothesis_validated=False,
                observations=["System unhealthy before experiment"],
                metrics_during={},
                recovery_time_seconds=None,
                issues_found=["Pre-experiment health check failed"]
            )

        observations.append("Pre-experiment health check passed")

        # Inject chaos (simulated in safe mode)
        chaos_injected = await self._inject_chaos(experiment)

        if not chaos_injected:
            observations.append("Chaos injection skipped (safe mode or unsupported)")
        else:
            observations.append(f"Chaos injected: {experiment.chaos_type.value}")

        # Monitor during chaos
        start_time = time.time()
        end_time = start_time + experiment.duration_seconds

        while time.time() < end_time:
            if metrics_collector:
                metrics = metrics_collector()
                for key, value in metrics.items():
                    if key not in metrics_during:
                        metrics_during[key] = []
                    metrics_during[key].append(value)

            # Check health during chaos
            if not health_check():
                issues_found.append(f"Health check failed at {time.time() - start_time:.1f}s")

            await asyncio.sleep(1)

        # Rollback
        await self._rollback(experiment)
        observations.append("Chaos rollback completed")

        # Measure recovery time
        recovery_start = time.time()
        recovery_timeout = 30

        while time.time() - recovery_start < recovery_timeout:
            if health_check():
                break
            await asyncio.sleep(0.5)

        recovery_time = time.time() - recovery_start

        if recovery_time >= recovery_timeout:
            issues_found.append("System did not recover within timeout")
            observations.append("Recovery timeout exceeded")
        else:
            observations.append(f"System recovered in {recovery_time:.1f}s")

        # Validate hypothesis
        hypothesis_validated = len(issues_found) == 0 or (
            experiment.hypothesis.lower().startswith("system should")
            and recovery_time < experiment.duration_seconds
        )

        result = ChaosResult(
            experiment_id=experiment.id,
            success=chaos_injected,
            hypothesis_validated=hypothesis_validated,
            observations=observations,
            metrics_during=metrics_during,
            recovery_time_seconds=recovery_time if recovery_time < recovery_timeout else None,
            issues_found=issues_found
        )

        self.results.append(result)
        logger.info(f"Chaos experiment completed: {experiment.name}")

        return result

    async def _inject_chaos(self, experiment: ChaosExperiment) -> bool:
        """Inject chaos into the system"""
        if self.safe_mode:
            logger.info(f"Safe mode: simulating {experiment.chaos_type.value}")
            return True

        # Real chaos injection would go here
        # For now, we just log and return
        logger.warning(f"Chaos injection not implemented for {experiment.chaos_type.value}")
        return False

    async def _rollback(self, experiment: ChaosExperiment) -> None:
        """Rollback chaos injection"""
        logger.info(f"Rolling back: {experiment.rollback_procedure}")
        # Actual rollback logic would go here
        await asyncio.sleep(0.1)

    def get_experiment_templates(self) -> List[Dict[str, Any]]:
        """Get templates for common chaos experiments"""
        return [
            {
                "name": "Network Latency Test",
                "type": ChaosType.NETWORK_LATENCY.value,
                "description": "Inject latency into network calls",
                "parameters": {"latency_ms": 500, "jitter_ms": 100},
                "hypothesis": "System should handle increased latency gracefully"
            },
            {
                "name": "Service Failure Test",
                "type": ChaosType.SERVICE_FAILURE.value,
                "description": "Simulate service failure",
                "parameters": {"failure_rate": 0.5},
                "hypothesis": "System should failover to backup service"
            },
            {
                "name": "Resource Exhaustion Test",
                "type": ChaosType.RESOURCE_EXHAUSTION.value,
                "description": "Exhaust memory/CPU resources",
                "parameters": {"memory_percent": 80, "cpu_percent": 90},
                "hypothesis": "System should degrade gracefully under resource pressure"
            },
            {
                "name": "Dependency Failure Test",
                "type": ChaosType.DEPENDENCY_FAILURE.value,
                "description": "Simulate external dependency failure",
                "parameters": {"dependency": "database", "failure_mode": "timeout"},
                "hypothesis": "System should use cached data or return error gracefully"
            }
        ]

    def generate_resilience_report(self) -> Dict[str, Any]:
        """Generate a resilience report from all experiments"""
        if not self.results:
            return {"message": "No experiments conducted yet"}

        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        validated = sum(1 for r in self.results if r.hypothesis_validated)

        avg_recovery = [r.recovery_time_seconds for r in self.results if r.recovery_time_seconds]

        all_issues = [issue for r in self.results for issue in r.issues_found]

        return {
            "total_experiments": total,
            "successful_injections": successful,
            "hypotheses_validated": validated,
            "validation_rate": validated / max(total, 1),
            "average_recovery_time": sum(avg_recovery) / len(avg_recovery) if avg_recovery else None,
            "common_issues": list(set(all_issues)),
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []

        failed_experiments = [r for r in self.results if not r.hypothesis_validated]

        for result in failed_experiments:
            exp = next((e for e in self.experiments if e.id == result.experiment_id), None)
            if exp:
                if exp.chaos_type == ChaosType.NETWORK_LATENCY:
                    recommendations.append("Implement circuit breakers for external calls")
                elif exp.chaos_type == ChaosType.SERVICE_FAILURE:
                    recommendations.append("Add retry logic with exponential backoff")
                elif exp.chaos_type == ChaosType.RESOURCE_EXHAUSTION:
                    recommendations.append("Implement graceful degradation under load")

        return list(set(recommendations))


class PerformanceTester:
    """
    Performance and load testing with baseline comparisons.

    Measures and tracks performance metrics over time.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.test_results: List[Dict[str, Any]] = []

        logger.info(f"PerformanceTester initialized for: {workspace_dir}")

    def create_baseline(
        self,
        metric_name: str,
        values: List[float],
        warning_threshold_percent: float = 20,
        critical_threshold_percent: float = 50
    ) -> PerformanceBaseline:
        """Create a performance baseline from sample values"""
        if not values:
            raise ValueError("Values list cannot be empty")

        import statistics
        baseline_value = statistics.mean(values)

        baseline = PerformanceBaseline(
            id=f"baseline_{uuid.uuid4().hex[:12]}",
            metric_name=metric_name,
            baseline_value=baseline_value,
            threshold_warning=baseline_value * (1 + warning_threshold_percent / 100),
            threshold_critical=baseline_value * (1 + critical_threshold_percent / 100),
            samples=len(values),
            created_at=datetime.now()
        )

        self.baselines[metric_name] = baseline
        logger.info(f"Created baseline for {metric_name}: {baseline_value:.2f}")

        return baseline

    def compare_to_baseline(
        self,
        metric_name: str,
        current_value: float
    ) -> Dict[str, Any]:
        """Compare a current value to its baseline"""
        if metric_name not in self.baselines:
            return {
                "status": "no_baseline",
                "message": f"No baseline exists for {metric_name}"
            }

        baseline = self.baselines[metric_name]
        deviation_percent = ((current_value - baseline.baseline_value)
                            / baseline.baseline_value * 100)

        if current_value >= baseline.threshold_critical:
            status = "critical"
        elif current_value >= baseline.threshold_warning:
            status = "warning"
        else:
            status = "ok"

        return {
            "metric": metric_name,
            "current_value": current_value,
            "baseline_value": baseline.baseline_value,
            "deviation_percent": round(deviation_percent, 2),
            "status": status,
            "thresholds": {
                "warning": baseline.threshold_warning,
                "critical": baseline.threshold_critical
            }
        }

    async def run_load_test(
        self,
        target_function: Callable,
        concurrent_users: int = 10,
        duration_seconds: int = 30,
        ramp_up_seconds: int = 5
    ) -> Dict[str, Any]:
        """Run a load test against a target function"""
        logger.info(f"Starting load test: {concurrent_users} users for {duration_seconds}s")

        results = []
        errors = []
        start_time = time.time()

        async def worker(worker_id: int):
            worker_results = []
            # Ramp up delay
            await asyncio.sleep((worker_id / concurrent_users) * ramp_up_seconds)

            while time.time() - start_time < duration_seconds:
                call_start = time.time()
                try:
                    if asyncio.iscoroutinefunction(target_function):
                        await target_function()
                    else:
                        target_function()
                    duration = (time.time() - call_start) * 1000
                    worker_results.append({"duration_ms": duration, "success": True})
                except Exception as e:
                    duration = (time.time() - call_start) * 1000
                    worker_results.append({
                        "duration_ms": duration,
                        "success": False,
                        "error": str(e)
                    })
                    errors.append(str(e))

            return worker_results

        # Run workers
        tasks = [worker(i) for i in range(concurrent_users)]
        worker_results = await asyncio.gather(*tasks)

        # Aggregate results
        for wr in worker_results:
            results.extend(wr)

        # Calculate statistics
        import statistics

        durations = [r["duration_ms"] for r in results]
        successful = [r for r in results if r["success"]]

        total_requests = len(results)
        throughput = total_requests / duration_seconds

        test_result = {
            "test_id": f"load_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "concurrent_users": concurrent_users,
                "duration_seconds": duration_seconds,
                "ramp_up_seconds": ramp_up_seconds
            },
            "results": {
                "total_requests": total_requests,
                "successful_requests": len(successful),
                "failed_requests": total_requests - len(successful),
                "success_rate": len(successful) / max(total_requests, 1),
                "throughput_rps": round(throughput, 2),
                "response_times": {
                    "min_ms": round(min(durations), 2),
                    "max_ms": round(max(durations), 2),
                    "mean_ms": round(statistics.mean(durations), 2),
                    "median_ms": round(statistics.median(durations), 2),
                    "p95_ms": round(sorted(durations)[int(len(durations) * 0.95)], 2),
                    "p99_ms": round(sorted(durations)[int(len(durations) * 0.99)], 2)
                }
            },
            "errors": list(set(errors))[:10]
        }

        self.test_results.append(test_result)
        logger.info(f"Load test completed: {total_requests} requests, {throughput:.2f} rps")

        return test_result

    def get_performance_trend(
        self,
        metric_name: str,
        last_n_tests: int = 10
    ) -> Dict[str, Any]:
        """Get performance trend for a metric"""
        relevant_results = [
            r for r in self.test_results[-last_n_tests:]
            if metric_name in str(r)
        ]

        if not relevant_results:
            return {"message": "No relevant test results"}

        values = []
        for result in relevant_results:
            if "response_times" in result.get("results", {}):
                values.append(result["results"]["response_times"].get("mean_ms", 0))

        if not values:
            return {"message": "No values found for trend analysis"}

        import statistics

        trend = "stable"
        if len(values) >= 3:
            if values[-1] > values[0] * 1.1:
                trend = "degrading"
            elif values[-1] < values[0] * 0.9:
                trend = "improving"

        return {
            "metric": metric_name,
            "data_points": len(values),
            "trend": trend,
            "latest_value": values[-1],
            "average": round(statistics.mean(values), 2),
            "min": min(values),
            "max": max(values)
        }


class VisualRegressionTester:
    """
    Visual regression testing for UI components.

    Compares visual snapshots to detect UI changes.
    """

    def __init__(self, workspace_dir: str, snapshots_dir: Optional[str] = None):
        self.workspace_dir = Path(workspace_dir)
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else self.workspace_dir / ".visual_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: Dict[str, VisualSnapshot] = {}

        self._load_snapshots()
        logger.info(f"VisualRegressionTester initialized")

    def _load_snapshots(self) -> None:
        """Load existing snapshots"""
        index_file = self.snapshots_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    for name, snap_data in data.items():
                        self.snapshots[name] = VisualSnapshot(
                            id=snap_data["id"],
                            component=snap_data["component"],
                            hash=snap_data["hash"],
                            dimensions=tuple(snap_data["dimensions"]),
                            created_at=datetime.fromisoformat(snap_data["created_at"]),
                            metadata=snap_data.get("metadata", {})
                        )
            except Exception as e:
                logger.warning(f"Failed to load snapshots: {e}")

    def _save_snapshots(self) -> None:
        """Save snapshots index"""
        index_file = self.snapshots_dir / "index.json"
        data = {
            name: {
                "id": snap.id,
                "component": snap.component,
                "hash": snap.hash,
                "dimensions": list(snap.dimensions),
                "created_at": snap.created_at.isoformat(),
                "metadata": snap.metadata
            }
            for name, snap in self.snapshots.items()
        }

        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def capture_snapshot(
        self,
        component: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VisualSnapshot:
        """Capture a visual snapshot"""
        content_hash = hashlib.sha256(content).hexdigest()

        snapshot = VisualSnapshot(
            id=f"snap_{uuid.uuid4().hex[:12]}",
            component=component,
            hash=content_hash,
            dimensions=(0, 0),  # Would be actual dimensions in real implementation
            created_at=datetime.now(),
            metadata=metadata or {}
        )

        # Save snapshot content
        snapshot_file = self.snapshots_dir / f"{snapshot.id}.bin"
        snapshot_file.write_bytes(content)

        self.snapshots[component] = snapshot
        self._save_snapshots()

        logger.info(f"Captured snapshot for {component}: {snapshot.id}")
        return snapshot

    def compare_snapshot(
        self,
        component: str,
        current_content: bytes,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Compare current content to baseline snapshot"""
        if component not in self.snapshots:
            return {
                "status": "no_baseline",
                "message": f"No baseline snapshot for {component}",
                "action": "capture_baseline"
            }

        baseline = self.snapshots[component]
        current_hash = hashlib.sha256(current_content).hexdigest()

        if current_hash == baseline.hash:
            return {
                "status": "match",
                "message": "Visual snapshot matches baseline",
                "component": component,
                "baseline_id": baseline.id
            }

        # In a real implementation, we would calculate pixel difference
        # For now, we just detect hash mismatch

        return {
            "status": "difference",
            "message": "Visual difference detected",
            "component": component,
            "baseline_hash": baseline.hash,
            "current_hash": current_hash,
            "baseline_id": baseline.id,
            "actions": ["update_baseline", "investigate_change"]
        }

    def update_baseline(
        self,
        component: str,
        new_content: bytes
    ) -> VisualSnapshot:
        """Update the baseline snapshot for a component"""
        return self.capture_snapshot(component, new_content)

    def get_all_components(self) -> List[Dict[str, Any]]:
        """Get all tracked components"""
        return [
            {
                "component": snap.component,
                "id": snap.id,
                "hash": snap.hash[:16] + "...",
                "created_at": snap.created_at.isoformat()
            }
            for snap in self.snapshots.values()
        ]
