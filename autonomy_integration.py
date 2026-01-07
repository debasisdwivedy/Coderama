"""
Autonomy Integration Layer

Integrates the autonomy modules with the main Coderama application.
Provides self-healing wrappers, dashboard data, and hooks into the agent workflow.
"""

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global autonomy system instance
_autonomy_system = None
_initialized = False


def get_autonomy_system():
    """Get or create the global autonomy system instance."""
    global _autonomy_system, _initialized

    if _autonomy_system is None:
        from autonomy.orchestrator import create_autonomous_system

        workspace = os.getenv("WORKSPACE_DIR", "/tmp/coderama_workspace")
        storage = str(Path(workspace).parent / ".coderama_autonomy")

        _autonomy_system = create_autonomous_system(
            workspace_dir=workspace,
            storage_dir=storage,
            auto_heal=True,
            learning_enabled=True,
            chaos_testing_enabled=False,
            require_approval_for_production=True
        )
        _initialized = True
        logger.info("Autonomy system initialized")

    return _autonomy_system


def update_workspace(workspace_dir: str):
    """Update the workspace directory for the autonomy system."""
    global _autonomy_system

    if _autonomy_system is not None:
        _autonomy_system.workspace = Path(workspace_dir)
        _autonomy_system.self_healing.workspace_dir = workspace_dir
        _autonomy_system.security_scanner.workspace_dir = Path(workspace_dir)
        logger.info(f"Autonomy workspace updated to: {workspace_dir}")


class SelfHealingWrapper:
    """Wrapper that adds self-healing capabilities to agent calls."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.system = get_autonomy_system()
        self.circuit_breaker = self.system.self_healing.get_circuit_breaker(
            component_name,
            failure_threshold=3,
            recovery_timeout=30
        )

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with self-healing protection."""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {self.component_name}")
            return {
                "error": f"Service {self.component_name} is temporarily unavailable",
                "retry_after": self.circuit_breaker.recovery_timeout
            }

        try:
            # Record start in observability
            self.system.observability.metrics.increment(
                "agent.calls.total",
                tags={"component": self.component_name}
            )

            start_time = datetime.now()

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.circuit_breaker.record_success()

            self.system.observability.metrics.timer(
                "agent.call.duration",
                duration_ms,
                tags={"component": self.component_name}
            )

            # Log to audit trail
            self.system.observability.audit_trail.log(
                actor=self.component_name,
                action="agent_call",
                resource="coordinator",
                resource_type="agent",
                outcome="success"
            )

            # Record for learning
            if self.system.learning:
                from autonomy.learning import LearningEventType
                self.system.learning.record_event(
                    event_type=LearningEventType.SPRINT_COMPLETED,
                    context={"component": self.component_name},
                    outcome="success",
                    metrics={"duration_ms": duration_ms}
                )

            return result

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure(e)

            self.system.observability.metrics.increment(
                "agent.calls.failure",
                tags={"component": self.component_name}
            )

            # Attempt self-healing
            healing_result = await self.system.handle_error(
                e,
                self.component_name,
                {"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
            )

            if healing_result.get("healed"):
                logger.info(f"Self-healing succeeded for {self.component_name}")
                # Retry once after healing
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Retry failed after healing: {retry_error}")
                    raise retry_error

            raise


# Wrapper instances for each agent
_agent_wrappers: Dict[str, SelfHealingWrapper] = {}


def get_agent_wrapper(agent_name: str) -> SelfHealingWrapper:
    """Get or create a self-healing wrapper for an agent."""
    if agent_name not in _agent_wrappers:
        _agent_wrappers[agent_name] = SelfHealingWrapper(agent_name)
    return _agent_wrappers[agent_name]


async def wrapped_send_message(original_send_message, agent_name: str, task: str, tool_context):
    """Wrapped version of send_message with self-healing."""
    wrapper = get_agent_wrapper(agent_name)
    return await wrapper.execute(original_send_message, agent_name, task, tool_context)


# Dashboard data functions for Gradio UI

def get_dashboard_summary() -> Dict[str, Any]:
    """Get summary data for the autonomy dashboard."""
    try:
        system = get_autonomy_system()
        return system.get_dashboard_data()
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {"error": str(e)}


def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    try:
        system = get_autonomy_system()
        return system.self_healing.get_health_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_security_scan_results() -> Dict[str, Any]:
    """Run and get security scan results."""
    try:
        system = get_autonomy_system()
        return system.security_scanner.scan_directory()
    except Exception as e:
        return {"error": str(e)}


def get_roi_summary() -> Dict[str, Any]:
    """Get ROI summary."""
    try:
        system = get_autonomy_system()
        return system.business_intel.roi_calculator.get_summary()
    except Exception as e:
        return {"error": str(e)}


def get_recommendations() -> List[Dict[str, Any]]:
    """Get system recommendations."""
    try:
        system = get_autonomy_system()
        return system.get_recommendations()
    except Exception as e:
        return [{"error": str(e)}]


def record_task_completion(task_type: str, hours: float, quality: float = 1.0):
    """Record a task completion for ROI tracking."""
    try:
        system = get_autonomy_system()
        system.business_intel.roi_calculator.record_task(task_type, hours, quality)
    except Exception as e:
        logger.error(f"Failed to record task: {e}")


def create_checkpoint(description: str) -> bool:
    """Create a deployment checkpoint."""
    try:
        system = get_autonomy_system()
        system.self_healing.create_deployment_checkpoint(description)
        return True
    except Exception as e:
        logger.error(f"Failed to create checkpoint: {e}")
        return False


def rollback_to_checkpoint(checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """Rollback to a checkpoint."""
    try:
        system = get_autonomy_system()
        success, message = system.self_healing.rollback_manager.rollback(checkpoint_id)
        return {"success": success, "message": message}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_checkpoints() -> List[Dict[str, Any]]:
    """Get list of available checkpoints."""
    try:
        system = get_autonomy_system()
        return system.self_healing.rollback_manager.list_checkpoints()
    except Exception as e:
        return []


def get_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent audit log entries."""
    try:
        system = get_autonomy_system()
        entries = system.observability.audit_trail.query(limit=limit)
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "actor": e.actor,
                "action": e.action,
                "resource": e.resource,
                "outcome": e.outcome
            }
            for e in entries
        ]
    except Exception as e:
        return []


def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    try:
        system = get_autonomy_system()
        return {
            name: cb.get_metrics()
            for name, cb in system.self_healing.circuit_breakers.items()
        }
    except Exception as e:
        return {"error": str(e)}


def format_dashboard_html() -> str:
    """Format dashboard data as HTML for Gradio display."""
    try:
        data = get_dashboard_summary()

        if "error" in data:
            return f"<div style='color: red;'>Error: {data['error']}</div>"

        system_state = data.get("system_state", {})
        health = system_state.get("health", "unknown")
        health_color = {
            "healthy": "#2eb886",
            "degraded": "#daa038",
            "critical": "#cc0000",
            "unknown": "#6272a4"
        }.get(health, "#6272a4")

        bi = data.get("business_intelligence", {})
        roi = bi.get("roi", {})

        html = f"""
        <div style="font-family: 'Segoe UI', monospace; padding: 10px;">
            <h3 style="color: #E6E6E6;">🤖 Autonomy System Status</h3>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px;">
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">SYSTEM HEALTH</div>
                    <div style="color: {health_color}; font-size: 24px; font-weight: bold;">{health.upper()}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">COST SAVED</div>
                    <div style="color: #2eb886; font-size: 24px; font-weight: bold;">${roi.get('total_cost_saved', 0):,.0f}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">TIME REDUCTION</div>
                    <div style="color: #00D9FF; font-size: 24px; font-weight: bold;">{roi.get('time_reduction', '0%')}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">TASKS COMPLETED</div>
                    <div style="color: #E6E6E6; font-size: 24px; font-weight: bold;">{roi.get('tasks_completed', 0)}</div>
                </div>
            </div>

            <h4 style="color: #E6E6E6;">📊 Self-Healing Status</h4>
            <div style="background: #2a2a40; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                <pre style="color: #E6E6E6; font-size: 11px; margin: 0; overflow-x: auto;">{json.dumps(data.get('self_healing', {}), indent=2, default=str)[:1000]}</pre>
            </div>

            <h4 style="color: #E6E6E6;">🔐 Security Status</h4>
            <div style="background: #2a2a40; padding: 10px; border-radius: 8px;">
                <pre style="color: #E6E6E6; font-size: 11px; margin: 0; overflow-x: auto;">{json.dumps(data.get('security', {}), indent=2, default=str)[:500]}</pre>
            </div>
        </div>
        """

        return html

    except Exception as e:
        return f"<div style='color: red;'>Error loading dashboard: {str(e)}</div>"


def format_security_html() -> str:
    """Format security scan results as HTML."""
    try:
        scan = get_security_scan_results()

        if "error" in scan:
            return f"<div style='color: red;'>Error: {scan['error']}</div>"

        by_severity = scan.get("by_severity", {})
        critical = by_severity.get("critical", 0)
        high = by_severity.get("high", 0)
        medium = by_severity.get("medium", 0)
        low = by_severity.get("low", 0)

        status_color = "#2eb886" if critical == 0 and high == 0 else ("#daa038" if critical == 0 else "#cc0000")

        html = f"""
        <div style="font-family: 'Segoe UI', monospace; padding: 10px;">
            <h3 style="color: #E6E6E6;">🔐 Security Scan Results</h3>

            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 20px;">
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">STATUS</div>
                    <div style="color: {status_color}; font-size: 18px; font-weight: bold;">{"SECURE" if critical == 0 and high == 0 else "ISSUES FOUND"}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">CRITICAL</div>
                    <div style="color: #cc0000; font-size: 24px; font-weight: bold;">{critical}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">HIGH</div>
                    <div style="color: #ff6b6b; font-size: 24px; font-weight: bold;">{high}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">MEDIUM</div>
                    <div style="color: #daa038; font-size: 24px; font-weight: bold;">{medium}</div>
                </div>
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px;">
                    <div style="color: #6272a4; font-size: 12px;">LOW</div>
                    <div style="color: #6272a4; font-size: 24px; font-weight: bold;">{low}</div>
                </div>
            </div>

            <div style="color: #6272a4; font-size: 12px;">
                Files Scanned: {scan.get('files_scanned', 0)} |
                Total Findings: {scan.get('total_findings', 0)} |
                Scan ID: {scan.get('scan_id', 'N/A')}
            </div>
        </div>
        """

        return html

    except Exception as e:
        return f"<div style='color: red;'>Error running security scan: {str(e)}</div>"


def format_recommendations_html() -> str:
    """Format recommendations as HTML."""
    try:
        recs = get_recommendations()

        if not recs:
            return "<div style='color: #6272a4;'>No recommendations at this time.</div>"

        html = """
        <div style="font-family: 'Segoe UI', monospace; padding: 10px;">
            <h3 style="color: #E6E6E6;">💡 Recommendations</h3>
        """

        priority_colors = {
            "high": "#cc0000",
            "medium": "#daa038",
            "low": "#2eb886"
        }

        for rec in recs:
            if isinstance(rec, dict) and "error" not in rec:
                priority = rec.get("priority", "medium")
                color = priority_colors.get(priority, "#6272a4")
                html += f"""
                <div style="background: #2a2a40; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {color};">
                    <div style="color: {color}; font-size: 12px; text-transform: uppercase;">{rec.get('category', 'General')} - {priority}</div>
                    <div style="color: #E6E6E6; font-size: 14px; font-weight: bold; margin-top: 5px;">{rec.get('title', 'Recommendation')}</div>
                    <div style="color: #6272a4; font-size: 12px; margin-top: 5px;">{rec.get('description', '')}</div>
                </div>
                """

        html += "</div>"
        return html

    except Exception as e:
        return f"<div style='color: red;'>Error loading recommendations: {str(e)}</div>"


def format_audit_log_html() -> str:
    """Format audit log as HTML."""
    try:
        entries = get_audit_log(20)

        if not entries:
            return "<div style='color: #6272a4;'>No audit entries yet.</div>"

        html = """
        <div style="font-family: 'Segoe UI', monospace; padding: 10px;">
            <h3 style="color: #E6E6E6;">📋 Recent Audit Log</h3>
            <div style="background: #2a2a40; border-radius: 8px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                    <thead>
                        <tr style="background: #1a1a2e;">
                            <th style="padding: 8px; text-align: left; color: #6272a4;">Time</th>
                            <th style="padding: 8px; text-align: left; color: #6272a4;">Actor</th>
                            <th style="padding: 8px; text-align: left; color: #6272a4;">Action</th>
                            <th style="padding: 8px; text-align: left; color: #6272a4;">Resource</th>
                            <th style="padding: 8px; text-align: left; color: #6272a4;">Outcome</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for entry in entries[:15]:
            outcome_color = "#2eb886" if entry.get("outcome") == "success" else "#cc0000"
            timestamp = entry.get("timestamp", "")[:19]  # Trim to readable format

            html += f"""
                        <tr style="border-bottom: 1px solid #1a1a2e;">
                            <td style="padding: 8px; color: #6272a4;">{timestamp}</td>
                            <td style="padding: 8px; color: #E6E6E6;">{entry.get('actor', 'N/A')}</td>
                            <td style="padding: 8px; color: #00D9FF;">{entry.get('action', 'N/A')}</td>
                            <td style="padding: 8px; color: #E6E6E6;">{entry.get('resource', 'N/A')}</td>
                            <td style="padding: 8px; color: {outcome_color};">{entry.get('outcome', 'N/A')}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """

        return html

    except Exception as e:
        return f"<div style='color: red;'>Error loading audit log: {str(e)}</div>"
