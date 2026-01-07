"""
Self-Healing & Recovery Module

Provides autonomous recovery capabilities:
- Automatic rollback on failed deployments with root cause analysis
- Self-debugging agents that can fix their own code errors
- Automatic dependency conflict resolution
- Circuit breakers to isolate failing components
"""

import asyncio
import json
import logging
import hashlib
import subprocess
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import re
import shutil

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure threshold exceeded, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureCategory(Enum):
    """Categories of failures for root cause analysis"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    DEPENDENCY_ERROR = "dependency_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    """Represents a failure event for analysis"""
    timestamp: datetime
    category: FailureCategory
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    component: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    auto_fixed: bool = False


@dataclass
class Checkpoint:
    """Represents a deployment checkpoint for rollback"""
    id: str
    timestamp: datetime
    description: str
    files: Dict[str, str]  # file_path -> content hash
    file_contents: Dict[str, str]  # file_path -> actual content
    dependencies: Dict[str, str]  # package -> version
    config: Dict[str, Any]
    health_status: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for isolating failing components.

    Prevents cascade failures by temporarily stopping calls to a failing service.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._metrics: List[Dict[str, Any]] = []

        logger.info(f"CircuitBreaker '{name}' initialized with threshold={failure_threshold}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, auto-transitioning if recovery timeout passed"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    logger.info(f"CircuitBreaker '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    def can_execute(self) -> bool:
        """Check if the circuit allows execution"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False

    def record_success(self) -> None:
        """Record a successful call"""
        self._success_count += 1
        self._metrics.append({
            "timestamp": datetime.now().isoformat(),
            "type": "success",
            "state": self._state.value
        })

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                logger.info(f"CircuitBreaker '{self.name}' recovered, transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        self._metrics.append({
            "timestamp": datetime.now().isoformat(),
            "type": "failure",
            "state": self._state.value,
            "error": str(error) if error else None
        })

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"CircuitBreaker '{self.name}' failed in HALF_OPEN, reopening")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            logger.warning(f"CircuitBreaker '{self.name}' threshold exceeded, opening circuit")
            self._state = CircuitState.OPEN

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        if not self.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit '{self.name}' is OPEN. Retry after {self.recovery_timeout}s"
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "recent_events": self._metrics[-20:]  # Last 20 events
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"CircuitBreaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class RollbackManager:
    """
    Manages deployment checkpoints and automatic rollback.

    Creates snapshots of the system state and can restore previous states
    when deployments fail.
    """

    def __init__(self, workspace_dir: str, max_checkpoints: int = 10):
        self.workspace_dir = Path(workspace_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: deque[Checkpoint] = deque(maxlen=max_checkpoints)
        self.checkpoint_dir = self.workspace_dir / ".coderama" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoints()

        logger.info(f"RollbackManager initialized with workspace: {workspace_dir}")

    def _load_checkpoints(self) -> None:
        """Load existing checkpoints from disk"""
        checkpoint_index = self.checkpoint_dir / "index.json"
        if checkpoint_index.exists():
            try:
                with open(checkpoint_index, "r") as f:
                    data = json.load(f)
                    for cp_data in data.get("checkpoints", []):
                        self.checkpoints.append(Checkpoint(
                            id=cp_data["id"],
                            timestamp=datetime.fromisoformat(cp_data["timestamp"]),
                            description=cp_data["description"],
                            files=cp_data["files"],
                            file_contents=cp_data.get("file_contents", {}),
                            dependencies=cp_data.get("dependencies", {}),
                            config=cp_data.get("config", {}),
                            health_status=cp_data.get("health_status", True),
                            metadata=cp_data.get("metadata", {})
                        ))
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints from disk")
            except Exception as e:
                logger.warning(f"Failed to load checkpoints: {e}")

    def _save_checkpoints(self) -> None:
        """Persist checkpoints to disk"""
        checkpoint_index = self.checkpoint_dir / "index.json"
        data = {
            "checkpoints": [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp.isoformat(),
                    "description": cp.description,
                    "files": cp.files,
                    "file_contents": cp.file_contents,
                    "dependencies": cp.dependencies,
                    "config": cp.config,
                    "health_status": cp.health_status,
                    "metadata": cp.metadata
                }
                for cp in self.checkpoints
            ]
        }
        with open(checkpoint_index, "w") as f:
            json.dump(data, f, indent=2)

    def _hash_content(self, content: str) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _collect_files(self, patterns: Optional[List[str]] = None) -> Dict[str, str]:
        """Collect all relevant files and their contents"""
        if patterns is None:
            patterns = ["**/*.py", "**/*.json", "**/*.yaml", "**/*.yml",
                       "**/*.txt", "**/*.md", "**/*.sh", "**/*.toml"]

        files = {}
        for pattern in patterns:
            for file_path in self.workspace_dir.glob(pattern):
                if file_path.is_file() and ".coderama" not in str(file_path):
                    try:
                        relative_path = str(file_path.relative_to(self.workspace_dir))
                        content = file_path.read_text()
                        files[relative_path] = content
                    except Exception as e:
                        logger.warning(f"Failed to read {file_path}: {e}")

        return files

    def _get_dependencies(self) -> Dict[str, str]:
        """Extract installed dependencies"""
        dependencies = {}

        # Check requirements.txt
        req_file = self.workspace_dir / "requirements.txt"
        if req_file.exists():
            for line in req_file.read_text().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "==" in line:
                        pkg, version = line.split("==", 1)
                        dependencies[pkg] = version
                    elif ">=" in line:
                        pkg, version = line.split(">=", 1)
                        dependencies[pkg] = f">={version}"
                    else:
                        dependencies[line] = "any"

        # Check pyproject.toml
        pyproject = self.workspace_dir / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                    deps = data.get("project", {}).get("dependencies", [])
                    for dep in deps:
                        if "==" in dep:
                            pkg, version = dep.split("==", 1)
                            dependencies[pkg] = version
            except Exception:
                pass

        return dependencies

    def create_checkpoint(
        self,
        description: str,
        health_check: Optional[Callable[[], bool]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Create a new checkpoint of the current system state"""
        checkpoint_id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(description.encode()).hexdigest()[:8]}"

        files = self._collect_files()
        file_hashes = {path: self._hash_content(content) for path, content in files.items()}

        health_status = True
        if health_check:
            try:
                health_status = health_check()
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                health_status = False

        checkpoint = Checkpoint(
            id=checkpoint_id,
            timestamp=datetime.now(),
            description=description,
            files=file_hashes,
            file_contents=files,
            dependencies=self._get_dependencies(),
            config={},
            health_status=health_status,
            metadata=metadata or {}
        )

        self.checkpoints.append(checkpoint)
        self._save_checkpoints()

        logger.info(f"Created checkpoint '{checkpoint_id}': {description}")
        return checkpoint

    def rollback(
        self,
        checkpoint_id: Optional[str] = None,
        steps: int = 1
    ) -> Tuple[bool, str]:
        """
        Rollback to a previous checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to rollback to
            steps: Number of checkpoints to go back (if checkpoint_id not specified)

        Returns:
            Tuple of (success, message)
        """
        if not self.checkpoints:
            return False, "No checkpoints available for rollback"

        target_checkpoint: Optional[Checkpoint] = None

        if checkpoint_id:
            for cp in self.checkpoints:
                if cp.id == checkpoint_id:
                    target_checkpoint = cp
                    break
            if not target_checkpoint:
                return False, f"Checkpoint '{checkpoint_id}' not found"
        else:
            if steps > len(self.checkpoints):
                steps = len(self.checkpoints)
            target_checkpoint = list(self.checkpoints)[-steps - 1] if steps < len(self.checkpoints) else list(self.checkpoints)[0]

        # Create a rollback checkpoint first
        self.create_checkpoint(
            f"Pre-rollback snapshot before reverting to {target_checkpoint.id}",
            metadata={"rollback_target": target_checkpoint.id}
        )

        # Restore files
        restored_files = []
        failed_files = []

        for file_path, content in target_checkpoint.file_contents.items():
            try:
                full_path = self.workspace_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                restored_files.append(file_path)
            except Exception as e:
                failed_files.append((file_path, str(e)))
                logger.error(f"Failed to restore {file_path}: {e}")

        if failed_files:
            return False, f"Rollback partially failed. Restored {len(restored_files)} files, failed {len(failed_files)}: {failed_files}"

        logger.info(f"Successfully rolled back to checkpoint '{target_checkpoint.id}'")
        return True, f"Rolled back to '{target_checkpoint.id}' ({target_checkpoint.description}). Restored {len(restored_files)} files."

    def get_checkpoint_diff(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get differences between current state and a checkpoint"""
        checkpoint = None
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            return {"error": f"Checkpoint '{checkpoint_id}' not found"}

        current_files = self._collect_files()
        current_hashes = {path: self._hash_content(content) for path, content in current_files.items()}

        added = set(current_hashes.keys()) - set(checkpoint.files.keys())
        removed = set(checkpoint.files.keys()) - set(current_hashes.keys())
        modified = {
            path for path in current_hashes.keys() & checkpoint.files.keys()
            if current_hashes[path] != checkpoint.files[path]
        }

        return {
            "checkpoint_id": checkpoint_id,
            "checkpoint_timestamp": checkpoint.timestamp.isoformat(),
            "added_files": list(added),
            "removed_files": list(removed),
            "modified_files": list(modified),
            "total_changes": len(added) + len(removed) + len(modified)
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        return [
            {
                "id": cp.id,
                "timestamp": cp.timestamp.isoformat(),
                "description": cp.description,
                "file_count": len(cp.files),
                "health_status": cp.health_status,
                "metadata": cp.metadata
            }
            for cp in self.checkpoints
        ]


class SelfDebugAgent:
    """
    Self-debugging agent that can analyze and fix code errors autonomously.

    Uses pattern matching and LLM-powered analysis to identify root causes
    and generate fixes.
    """

    # Common error patterns and their fixes
    ERROR_PATTERNS = [
        {
            "pattern": r"IndentationError: (unexpected indent|expected an indented block)",
            "category": FailureCategory.SYNTAX_ERROR,
            "fix_strategy": "indentation"
        },
        {
            "pattern": r"SyntaxError: (invalid syntax|unexpected EOF)",
            "category": FailureCategory.SYNTAX_ERROR,
            "fix_strategy": "syntax"
        },
        {
            "pattern": r"NameError: name '(\w+)' is not defined",
            "category": FailureCategory.RUNTIME_ERROR,
            "fix_strategy": "undefined_name"
        },
        {
            "pattern": r"ImportError: (No module named|cannot import name) '(\w+)'",
            "category": FailureCategory.DEPENDENCY_ERROR,
            "fix_strategy": "missing_import"
        },
        {
            "pattern": r"ModuleNotFoundError: No module named '(\w+)'",
            "category": FailureCategory.DEPENDENCY_ERROR,
            "fix_strategy": "missing_module"
        },
        {
            "pattern": r"TypeError: .+ takes (\d+) positional arguments? but (\d+) (?:was|were) given",
            "category": FailureCategory.LOGIC_ERROR,
            "fix_strategy": "argument_count"
        },
        {
            "pattern": r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
            "category": FailureCategory.LOGIC_ERROR,
            "fix_strategy": "missing_attribute"
        },
        {
            "pattern": r"KeyError: '(\w+)'",
            "category": FailureCategory.LOGIC_ERROR,
            "fix_strategy": "missing_key"
        },
        {
            "pattern": r"FileNotFoundError: \[Errno 2\] No such file or directory: '(.+)'",
            "category": FailureCategory.RESOURCE_ERROR,
            "fix_strategy": "missing_file"
        },
        {
            "pattern": r"ConnectionError|ConnectionRefusedError|TimeoutError",
            "category": FailureCategory.NETWORK_ERROR,
            "fix_strategy": "network_retry"
        },
        {
            "pattern": r"MemoryError|ResourceWarning",
            "category": FailureCategory.RESOURCE_ERROR,
            "fix_strategy": "resource_optimization"
        }
    ]

    def __init__(self, workspace_dir: str, llm_client: Optional[Any] = None):
        self.workspace_dir = Path(workspace_dir)
        self.llm_client = llm_client
        self.failure_history: List[FailureEvent] = []
        self.fix_attempts: Dict[str, int] = {}  # Track fix attempts per error
        self.max_fix_attempts = 3

        logger.info(f"SelfDebugAgent initialized for workspace: {workspace_dir}")

    def analyze_error(
        self,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FailureEvent:
        """Analyze an error and categorize it"""
        category = FailureCategory.UNKNOWN
        file_path = None
        line_number = None
        suggested_fix = None

        # Match against known patterns
        for pattern_info in self.ERROR_PATTERNS:
            match = re.search(pattern_info["pattern"], error_message)
            if match:
                category = pattern_info["category"]
                break

        # Extract file path and line number from stack trace
        if stack_trace:
            file_match = re.search(r'File "(.+?)", line (\d+)', stack_trace)
            if file_match:
                file_path = file_match.group(1)
                line_number = int(file_match.group(2))

        failure = FailureEvent(
            timestamp=datetime.now(),
            category=category,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
            component="unknown",
            file_path=file_path,
            line_number=line_number,
            suggested_fix=suggested_fix
        )

        self.failure_history.append(failure)
        return failure

    def _generate_fix_for_undefined_name(self, name: str, file_path: str) -> Optional[str]:
        """Generate fix for undefined name errors"""
        # Common import mappings
        common_imports = {
            "json": "import json",
            "os": "import os",
            "sys": "import sys",
            "re": "import re",
            "datetime": "from datetime import datetime",
            "Path": "from pathlib import Path",
            "List": "from typing import List",
            "Dict": "from typing import Dict",
            "Optional": "from typing import Optional",
            "Any": "from typing import Any",
            "asyncio": "import asyncio",
            "logging": "import logging",
        }

        if name in common_imports:
            return common_imports[name]

        return None

    def _generate_fix_for_missing_module(self, module_name: str) -> Optional[str]:
        """Generate pip install command for missing modules"""
        # Module to package mapping (common cases)
        package_mapping = {
            "PIL": "Pillow",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
        }

        package = package_mapping.get(module_name, module_name)
        return f"pip install {package}"

    async def attempt_fix(
        self,
        failure: FailureEvent,
        dry_run: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Attempt to automatically fix an error.

        Returns:
            Tuple of (success, message, fix_applied)
        """
        error_key = f"{failure.category.value}:{failure.error_message[:100]}"

        # Check if we've exceeded max attempts
        if self.fix_attempts.get(error_key, 0) >= self.max_fix_attempts:
            return False, f"Max fix attempts ({self.max_fix_attempts}) exceeded for this error", None

        self.fix_attempts[error_key] = self.fix_attempts.get(error_key, 0) + 1

        fix_applied = None
        success = False
        message = ""

        # Handle different categories
        if failure.category == FailureCategory.DEPENDENCY_ERROR:
            # Extract module name
            match = re.search(r"No module named '(\w+)'", failure.error_message)
            if match:
                module_name = match.group(1)
                install_cmd = self._generate_fix_for_missing_module(module_name)

                if dry_run:
                    return True, f"Would run: {install_cmd}", install_cmd

                try:
                    result = subprocess.run(
                        install_cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        success = True
                        message = f"Successfully installed {module_name}"
                        fix_applied = install_cmd
                    else:
                        message = f"Failed to install: {result.stderr}"
                except Exception as e:
                    message = f"Installation failed: {e}"

        elif failure.category == FailureCategory.RUNTIME_ERROR:
            # Handle undefined name
            match = re.search(r"name '(\w+)' is not defined", failure.error_message)
            if match and failure.file_path:
                name = match.group(1)
                import_statement = self._generate_fix_for_undefined_name(name, failure.file_path)

                if import_statement:
                    if dry_run:
                        return True, f"Would add: {import_statement}", import_statement

                    try:
                        file_path = Path(failure.file_path)
                        if file_path.exists():
                            content = file_path.read_text()
                            # Add import at the top (after any existing imports)
                            lines = content.split("\n")
                            import_index = 0
                            for i, line in enumerate(lines):
                                if line.startswith("import ") or line.startswith("from "):
                                    import_index = i + 1

                            lines.insert(import_index, import_statement)
                            file_path.write_text("\n".join(lines))

                            success = True
                            message = f"Added import: {import_statement}"
                            fix_applied = import_statement
                    except Exception as e:
                        message = f"Failed to add import: {e}"

        elif failure.category == FailureCategory.SYNTAX_ERROR:
            # For syntax errors, we need LLM assistance
            if self.llm_client and failure.file_path:
                # This would integrate with LLM for complex fixes
                message = "Syntax error requires manual review or LLM assistance"
            else:
                message = "Cannot auto-fix syntax error without LLM client"

        else:
            message = f"No auto-fix available for {failure.category.value}"

        failure.auto_fixed = success
        failure.suggested_fix = fix_applied

        return success, message, fix_applied

    def get_root_cause_analysis(self, failure: FailureEvent) -> Dict[str, Any]:
        """Generate root cause analysis for a failure"""
        analysis = {
            "failure_id": id(failure),
            "timestamp": failure.timestamp.isoformat(),
            "category": failure.category.value,
            "error_message": failure.error_message,
            "location": {
                "file": failure.file_path,
                "line": failure.line_number
            },
            "probable_causes": [],
            "suggested_actions": [],
            "similar_failures": []
        }

        # Add category-specific analysis
        if failure.category == FailureCategory.DEPENDENCY_ERROR:
            analysis["probable_causes"].append("Missing or incompatible package dependency")
            analysis["suggested_actions"].append("Install missing package or update requirements.txt")

        elif failure.category == FailureCategory.SYNTAX_ERROR:
            analysis["probable_causes"].append("Code syntax issue, possibly from incomplete generation")
            analysis["suggested_actions"].append("Review and correct syntax at the indicated location")

        elif failure.category == FailureCategory.RUNTIME_ERROR:
            analysis["probable_causes"].append("Variable or function referenced before definition")
            analysis["suggested_actions"].append("Check variable scope and import statements")

        elif failure.category == FailureCategory.NETWORK_ERROR:
            analysis["probable_causes"].append("Network connectivity issue or service unavailable")
            analysis["suggested_actions"].append("Implement retry logic with exponential backoff")

        # Find similar failures
        for past_failure in self.failure_history[-20:]:
            if past_failure.category == failure.category and past_failure != failure:
                if past_failure.auto_fixed and past_failure.suggested_fix:
                    analysis["similar_failures"].append({
                        "timestamp": past_failure.timestamp.isoformat(),
                        "fixed": True,
                        "fix_applied": past_failure.suggested_fix
                    })

        return analysis

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failures and fixes"""
        category_counts = {}
        fixed_counts = {}

        for failure in self.failure_history:
            cat = failure.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if failure.auto_fixed:
                fixed_counts[cat] = fixed_counts.get(cat, 0) + 1

        return {
            "total_failures": len(self.failure_history),
            "by_category": category_counts,
            "auto_fixed_by_category": fixed_counts,
            "fix_success_rate": sum(fixed_counts.values()) / max(len(self.failure_history), 1)
        }


class DependencyResolver:
    """
    Automatic dependency conflict resolution and version management.

    Analyzes dependency trees, detects conflicts, and proposes resolutions.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.dependency_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"DependencyResolver initialized for: {workspace_dir}")

    def _parse_requirements(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Parse requirements file into structured format"""
        deps = {}
        if not file_path.exists():
            return deps

        for line in file_path.read_text().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse version specifiers
            for op in ["==", ">=", "<=", "~=", "!="]:
                if op in line:
                    pkg, version = line.split(op, 1)
                    deps[pkg.strip()] = {"version": version.strip(), "operator": op}
                    break
            else:
                deps[line] = {"version": "any", "operator": None}

        return deps

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies for conflicts"""
        req_file = self.workspace_dir / "requirements.txt"
        deps = self._parse_requirements(req_file)

        conflicts = []
        warnings = []

        # Check for common conflict patterns
        conflict_groups = {
            "tensorflow": ["tensorflow-gpu", "tf-nightly"],
            "pillow": ["PIL"],
            "cv2": ["opencv-python", "opencv-python-headless"],
        }

        installed = list(deps.keys())
        for main_pkg, alternatives in conflict_groups.items():
            found = [pkg for pkg in [main_pkg] + alternatives if pkg in installed]
            if len(found) > 1:
                conflicts.append({
                    "type": "version_conflict",
                    "packages": found,
                    "suggestion": f"Use only one of: {', '.join(found)}"
                })

        # Check for pinned vs unpinned
        unpinned = [pkg for pkg, info in deps.items() if info["operator"] is None]
        if unpinned:
            warnings.append({
                "type": "unpinned_dependency",
                "packages": unpinned,
                "suggestion": "Pin all dependencies for reproducibility"
            })

        return {
            "total_dependencies": len(deps),
            "dependencies": deps,
            "conflicts": conflicts,
            "warnings": warnings,
            "health_score": 1.0 - (len(conflicts) * 0.2) - (len(warnings) * 0.05)
        }

    async def resolve_conflicts(
        self,
        strategy: str = "conservative"
    ) -> Tuple[bool, str, Dict[str, str]]:
        """
        Attempt to resolve dependency conflicts.

        Args:
            strategy: 'conservative' (prefer older stable) or 'aggressive' (prefer latest)
        """
        analysis = self.analyze_dependencies()

        if not analysis["conflicts"]:
            return True, "No conflicts to resolve", {}

        resolutions = {}

        for conflict in analysis["conflicts"]:
            if conflict["type"] == "version_conflict":
                # For version conflicts, keep the first one (usually the main package)
                main_pkg = conflict["packages"][0]
                for pkg in conflict["packages"][1:]:
                    resolutions[pkg] = f"remove (conflicts with {main_pkg})"

        return True, f"Proposed {len(resolutions)} resolutions", resolutions

    def generate_lockfile(self) -> str:
        """Generate a dependency lockfile with exact versions"""
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                lockfile_path = self.workspace_dir / "requirements.lock"
                lockfile_path.write_text(result.stdout)
                return str(lockfile_path)
        except Exception as e:
            logger.error(f"Failed to generate lockfile: {e}")

        return ""


class SelfHealingEngine:
    """
    Main orchestrator for self-healing capabilities.

    Coordinates circuit breakers, rollback manager, self-debugging,
    and dependency resolution.
    """

    def __init__(
        self,
        workspace_dir: str,
        llm_client: Optional[Any] = None,
        auto_heal: bool = True
    ):
        self.workspace_dir = workspace_dir
        self.auto_heal = auto_heal

        # Initialize components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rollback_manager = RollbackManager(workspace_dir)
        self.debug_agent = SelfDebugAgent(workspace_dir, llm_client)
        self.dependency_resolver = DependencyResolver(workspace_dir)

        # Healing statistics
        self.healing_attempts = 0
        self.healing_successes = 0
        self.last_healing_event: Optional[datetime] = None

        logger.info(f"SelfHealingEngine initialized with auto_heal={auto_heal}")

    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a component"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[name]

    async def handle_failure(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a failure with automatic healing attempt.

        Returns a report of actions taken.
        """
        self.healing_attempts += 1
        self.last_healing_event = datetime.now()

        report = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error": str(error),
            "actions_taken": [],
            "healed": False
        }

        # Record in circuit breaker
        cb = self.get_circuit_breaker(component)
        cb.record_failure(error)
        report["circuit_breaker_state"] = cb.state.value

        if not self.auto_heal:
            report["actions_taken"].append("Auto-heal disabled, skipping recovery")
            return report

        # Analyze the error
        failure = self.debug_agent.analyze_error(
            str(error),
            traceback.format_exc(),
            context
        )
        report["failure_analysis"] = self.debug_agent.get_root_cause_analysis(failure)

        # Attempt automatic fix
        success, message, fix = await self.debug_agent.attempt_fix(failure)
        report["actions_taken"].append(f"Auto-fix attempt: {message}")

        if success:
            report["healed"] = True
            report["fix_applied"] = fix
            self.healing_successes += 1
            cb.reset()
        else:
            # If auto-fix failed and circuit is open, consider rollback
            if cb.state == CircuitState.OPEN:
                report["actions_taken"].append("Circuit breaker open, considering rollback")

                # Check if we have a healthy checkpoint
                checkpoints = self.rollback_manager.list_checkpoints()
                healthy = [cp for cp in checkpoints if cp["health_status"]]

                if healthy:
                    rollback_success, rollback_msg = self.rollback_manager.rollback()
                    report["actions_taken"].append(f"Rollback: {rollback_msg}")
                    if rollback_success:
                        report["healed"] = True
                        self.healing_successes += 1
                        cb.reset()

        logger.info(f"Failure handling complete: healed={report['healed']}")
        return report

    def create_deployment_checkpoint(
        self,
        description: str,
        health_check: Optional[Callable[[], bool]] = None
    ) -> Checkpoint:
        """Create a checkpoint before deployment"""
        return self.rollback_manager.create_checkpoint(description, health_check)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the system"""
        return {
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": {
                name: cb.get_metrics() for name, cb in self.circuit_breakers.items()
            },
            "healing_statistics": {
                "attempts": self.healing_attempts,
                "successes": self.healing_successes,
                "success_rate": self.healing_successes / max(self.healing_attempts, 1),
                "last_event": self.last_healing_event.isoformat() if self.last_healing_event else None
            },
            "failure_statistics": self.debug_agent.get_failure_statistics(),
            "dependency_health": self.dependency_resolver.analyze_dependencies(),
            "checkpoints_available": len(self.rollback_manager.list_checkpoints())
        }
