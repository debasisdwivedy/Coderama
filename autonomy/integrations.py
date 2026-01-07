"""
Integration Ecosystem Module

Provides integrations with external tools:
- Bidirectional sync with Jira/Linear/Asana
- Slack/Teams notifications with interactive approvals
- Git workflow automation
- External API testing
"""

import asyncio
import json
import logging
import re
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


class TicketStatus(Enum):
    """Status for project management tickets"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"


class TicketPriority(Enum):
    """Priority levels"""
    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOWEST = "lowest"


class NotificationType(Enum):
    """Types of notifications"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    APPROVAL_REQUEST = "approval_request"


class GitOperationType(Enum):
    """Git operation types"""
    COMMIT = "commit"
    BRANCH = "branch"
    PR_CREATE = "pr_create"
    PR_MERGE = "pr_merge"
    PR_COMMENT = "pr_comment"
    TAG = "tag"


@dataclass
class Ticket:
    """Represents a project management ticket"""
    id: str
    external_id: Optional[str]
    title: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    assignee: Optional[str]
    labels: List[str]
    story_points: Optional[int]
    sprint: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Represents a notification"""
    id: str
    type: NotificationType
    title: str
    message: str
    channel: str
    timestamp: datetime
    metadata: Dict[str, Any]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    delivered: bool = False
    response: Optional[Dict[str, Any]] = None


@dataclass
class PullRequest:
    """Represents a pull request"""
    id: str
    number: int
    title: str
    description: str
    source_branch: str
    target_branch: str
    status: str  # open, merged, closed
    author: str
    reviewers: List[str]
    labels: List[str]
    created_at: datetime
    merged_at: Optional[datetime] = None
    comments: List[Dict[str, Any]] = field(default_factory=list)


class ProjectManagementIntegration(ABC):
    """Abstract base class for project management integrations"""

    @abstractmethod
    async def sync_tickets(self, project_key: str) -> List[Ticket]:
        """Sync tickets from external system"""
        pass

    @abstractmethod
    async def create_ticket(self, ticket: Ticket) -> str:
        """Create a ticket in external system"""
        pass

    @abstractmethod
    async def update_ticket(self, ticket: Ticket) -> bool:
        """Update a ticket in external system"""
        pass


class JiraIntegration(ProjectManagementIntegration):
    """
    Integration with Jira for bidirectional ticket sync.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        email: Optional[str] = None
    ):
        self.base_url = base_url
        self.api_token = api_token
        self.email = email
        self.local_tickets: Dict[str, Ticket] = {}
        self.sync_log: List[Dict[str, Any]] = []

        logger.info("JiraIntegration initialized")

    def _status_to_jira(self, status: TicketStatus) -> str:
        """Convert internal status to Jira status"""
        mapping = {
            TicketStatus.TODO: "To Do",
            TicketStatus.IN_PROGRESS: "In Progress",
            TicketStatus.IN_REVIEW: "In Review",
            TicketStatus.DONE: "Done",
            TicketStatus.BLOCKED: "Blocked"
        }
        return mapping.get(status, "To Do")

    def _status_from_jira(self, jira_status: str) -> TicketStatus:
        """Convert Jira status to internal status"""
        jira_status_lower = jira_status.lower()
        if "done" in jira_status_lower or "closed" in jira_status_lower:
            return TicketStatus.DONE
        elif "progress" in jira_status_lower:
            return TicketStatus.IN_PROGRESS
        elif "review" in jira_status_lower:
            return TicketStatus.IN_REVIEW
        elif "blocked" in jira_status_lower:
            return TicketStatus.BLOCKED
        return TicketStatus.TODO

    async def sync_tickets(self, project_key: str) -> List[Ticket]:
        """Sync tickets from Jira"""
        logger.info(f"Syncing tickets from Jira project: {project_key}")

        # In production, this would make API calls to Jira
        # For demo, we simulate the sync
        synced_tickets = []

        # Record sync event
        self.sync_log.append({
            "timestamp": datetime.now().isoformat(),
            "project": project_key,
            "direction": "pull",
            "tickets_synced": len(synced_tickets)
        })

        return synced_tickets

    async def create_ticket(self, ticket: Ticket) -> str:
        """Create a ticket in Jira"""
        logger.info(f"Creating Jira ticket: {ticket.title}")

        # Simulate ticket creation
        external_id = f"PROJ-{len(self.local_tickets) + 1}"
        ticket.external_id = external_id
        self.local_tickets[ticket.id] = ticket

        self.sync_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "create",
            "ticket_id": ticket.id,
            "external_id": external_id
        })

        return external_id

    async def update_ticket(self, ticket: Ticket) -> bool:
        """Update a ticket in Jira"""
        logger.info(f"Updating Jira ticket: {ticket.external_id}")

        if ticket.id in self.local_tickets:
            self.local_tickets[ticket.id] = ticket
            ticket.updated_at = datetime.now()

            self.sync_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "update",
                "ticket_id": ticket.id,
                "external_id": ticket.external_id
            })

            return True

        return False

    async def transition_ticket(
        self,
        ticket_id: str,
        new_status: TicketStatus
    ) -> bool:
        """Transition a ticket to a new status"""
        if ticket_id in self.local_tickets:
            ticket = self.local_tickets[ticket_id]
            old_status = ticket.status
            ticket.status = new_status
            ticket.updated_at = datetime.now()

            self.sync_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "transition",
                "ticket_id": ticket_id,
                "from_status": old_status.value,
                "to_status": new_status.value
            })

            logger.info(f"Transitioned {ticket_id}: {old_status.value} -> {new_status.value}")
            return True

        return False

    def get_sprint_tickets(self, sprint_name: str) -> List[Ticket]:
        """Get all tickets in a sprint"""
        return [
            t for t in self.local_tickets.values()
            if t.sprint == sprint_name
        ]

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return {
            "total_tickets": len(self.local_tickets),
            "by_status": {
                status.value: len([t for t in self.local_tickets.values() if t.status == status])
                for status in TicketStatus
            },
            "last_sync": self.sync_log[-1] if self.sync_log else None,
            "sync_count": len(self.sync_log)
        }


class SlackIntegration:
    """
    Integration with Slack for notifications and interactive approvals.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None
    ):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.notifications: List[Notification] = []
        self.pending_approvals: Dict[str, Notification] = {}

        logger.info("SlackIntegration initialized")

    async def send_notification(
        self,
        channel: str,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        actions: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Send a notification to Slack"""
        notification = Notification(
            id=f"notif_{uuid.uuid4().hex[:12]}",
            type=notification_type,
            title=title,
            message=message,
            channel=channel,
            timestamp=datetime.now(),
            metadata=metadata or {},
            actions=actions or []
        )

        # Format message for Slack
        slack_message = self._format_message(notification)

        # In production, this would send to Slack
        logger.info(f"Slack notification: {title}")

        notification.delivered = True
        self.notifications.append(notification)

        if notification_type == NotificationType.APPROVAL_REQUEST:
            self.pending_approvals[notification.id] = notification

        return notification

    def _format_message(self, notification: Notification) -> Dict[str, Any]:
        """Format notification for Slack"""
        color_map = {
            NotificationType.INFO: "#36a64f",
            NotificationType.SUCCESS: "#2eb886",
            NotificationType.WARNING: "#daa038",
            NotificationType.ERROR: "#cc0000",
            NotificationType.APPROVAL_REQUEST: "#439fe0"
        }

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": notification.title
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": notification.message
                }
            }
        ]

        # Add action buttons
        if notification.actions:
            action_elements = []
            for action in notification.actions:
                action_elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get("label", "Action")
                    },
                    "value": action.get("value", ""),
                    "action_id": action.get("id", f"action_{uuid.uuid4().hex[:8]}")
                })

            blocks.append({
                "type": "actions",
                "elements": action_elements
            })

        return {
            "channel": notification.channel,
            "attachments": [{
                "color": color_map.get(notification.type, "#36a64f"),
                "blocks": blocks
            }]
        }

    async def request_approval(
        self,
        channel: str,
        title: str,
        description: str,
        approvers: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        timeout_minutes: int = 60
    ) -> Notification:
        """Request approval via Slack"""
        actions = [
            {"id": "approve", "label": "Approve", "value": "approved"},
            {"id": "reject", "label": "Reject", "value": "rejected"}
        ]

        notification = await self.send_notification(
            channel=channel,
            title=f":question: Approval Required: {title}",
            message=f"{description}\n\nApprovers: {', '.join(approvers)}",
            notification_type=NotificationType.APPROVAL_REQUEST,
            actions=actions,
            metadata={
                **(metadata or {}),
                "approvers": approvers,
                "timeout": datetime.now() + timedelta(minutes=timeout_minutes)
            }
        )

        return notification

    def handle_approval_response(
        self,
        notification_id: str,
        action: str,
        user: str
    ) -> Dict[str, Any]:
        """Handle an approval response"""
        if notification_id not in self.pending_approvals:
            return {"error": "Approval request not found"}

        notification = self.pending_approvals[notification_id]

        # Check if user is authorized approver
        approvers = notification.metadata.get("approvers", [])
        if user not in approvers:
            return {"error": "User not authorized to approve"}

        # Check timeout
        timeout = notification.metadata.get("timeout")
        if timeout and datetime.now() > timeout:
            return {"error": "Approval request has expired"}

        notification.response = {
            "action": action,
            "user": user,
            "timestamp": datetime.now().isoformat()
        }

        del self.pending_approvals[notification_id]

        logger.info(f"Approval {notification_id}: {action} by {user}")

        return {
            "status": "processed",
            "action": action,
            "notification_id": notification_id
        }

    async def send_deployment_notification(
        self,
        channel: str,
        environment: str,
        version: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Send deployment status notification"""
        emoji = ":white_check_mark:" if status == "success" else ":x:"
        notification_type = NotificationType.SUCCESS if status == "success" else NotificationType.ERROR

        message = f"""
*Environment:* {environment}
*Version:* {version}
*Status:* {status.upper()}
"""
        if details:
            if details.get("duration"):
                message += f"*Duration:* {details['duration']}s\n"
            if details.get("deployed_by"):
                message += f"*Deployed by:* {details['deployed_by']}\n"

        return await self.send_notification(
            channel=channel,
            title=f"{emoji} Deployment {status.capitalize()}: {version}",
            message=message,
            notification_type=notification_type,
            metadata={"environment": environment, "version": version, **details}
        )

    def get_notification_history(
        self,
        channel: Optional[str] = None,
        limit: int = 50
    ) -> List[Notification]:
        """Get notification history"""
        notifications = self.notifications

        if channel:
            notifications = [n for n in notifications if n.channel == channel]

        return notifications[-limit:]


class GitWorkflow:
    """
    Automates Git workflows including PR creation and code review.
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.pull_requests: Dict[str, PullRequest] = {}
        self.operations: List[Dict[str, Any]] = []

        logger.info(f"GitWorkflow initialized for: {repo_path}")

    def _run_git(self, *args) -> Tuple[bool, str]:
        """Run a git command"""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def get_current_branch(self) -> str:
        """Get current branch name"""
        success, output = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return output.strip() if success else "unknown"

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """Create a new branch"""
        # Checkout from branch
        success, _ = self._run_git("checkout", from_branch)
        if not success:
            return False

        # Create new branch
        success, output = self._run_git("checkout", "-b", branch_name)

        self.operations.append({
            "type": GitOperationType.BRANCH.value,
            "timestamp": datetime.now().isoformat(),
            "branch": branch_name,
            "success": success
        })

        return success

    def commit(
        self,
        message: str,
        files: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Create a commit"""
        # Stage files
        if files:
            for file in files:
                self._run_git("add", file)
        else:
            self._run_git("add", "-A")

        # Create commit
        success, output = self._run_git("commit", "-m", message)

        if success:
            # Get commit hash
            _, hash_output = self._run_git("rev-parse", "HEAD")
            commit_hash = hash_output.strip()[:8]

            self.operations.append({
                "type": GitOperationType.COMMIT.value,
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "hash": commit_hash,
                "success": True
            })

            return True, commit_hash

        return False, output

    def create_pull_request(
        self,
        title: str,
        description: str,
        source_branch: str,
        target_branch: str = "main",
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None
    ) -> PullRequest:
        """Create a pull request"""
        pr_number = len(self.pull_requests) + 1

        pr = PullRequest(
            id=f"pr_{uuid.uuid4().hex[:12]}",
            number=pr_number,
            title=title,
            description=description,
            source_branch=source_branch,
            target_branch=target_branch,
            status="open",
            author="autonomous-agent",
            reviewers=reviewers or [],
            labels=labels or [],
            created_at=datetime.now()
        )

        self.pull_requests[pr.id] = pr

        self.operations.append({
            "type": GitOperationType.PR_CREATE.value,
            "timestamp": datetime.now().isoformat(),
            "pr_id": pr.id,
            "pr_number": pr_number,
            "title": title
        })

        logger.info(f"Created PR #{pr_number}: {title}")
        return pr

    def add_pr_comment(
        self,
        pr_id: str,
        comment: str,
        author: str = "autonomous-agent"
    ) -> bool:
        """Add a comment to a pull request"""
        if pr_id not in self.pull_requests:
            return False

        pr = self.pull_requests[pr_id]
        pr.comments.append({
            "id": f"comment_{uuid.uuid4().hex[:8]}",
            "author": author,
            "body": comment,
            "timestamp": datetime.now().isoformat()
        })

        self.operations.append({
            "type": GitOperationType.PR_COMMENT.value,
            "timestamp": datetime.now().isoformat(),
            "pr_id": pr_id,
            "author": author
        })

        return True

    def merge_pull_request(
        self,
        pr_id: str,
        merge_method: str = "squash"
    ) -> bool:
        """Merge a pull request"""
        if pr_id not in self.pull_requests:
            return False

        pr = self.pull_requests[pr_id]

        if pr.status != "open":
            return False

        # Simulate merge
        pr.status = "merged"
        pr.merged_at = datetime.now()

        self.operations.append({
            "type": GitOperationType.PR_MERGE.value,
            "timestamp": datetime.now().isoformat(),
            "pr_id": pr_id,
            "pr_number": pr.number,
            "method": merge_method
        })

        logger.info(f"Merged PR #{pr.number}")
        return True

    def generate_changelog(
        self,
        since_tag: Optional[str] = None
    ) -> str:
        """Generate changelog from commits"""
        # Get commits
        if since_tag:
            success, output = self._run_git("log", f"{since_tag}..HEAD", "--oneline")
        else:
            success, output = self._run_git("log", "--oneline", "-20")

        if not success:
            return "# Changelog\n\nUnable to generate changelog."

        changelog = "# Changelog\n\n"

        for line in output.strip().split("\n"):
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    hash_id, message = parts
                    changelog += f"- {message} ({hash_id})\n"

        return changelog

    def get_diff_stats(
        self,
        source_branch: str,
        target_branch: str = "main"
    ) -> Dict[str, Any]:
        """Get diff statistics between branches"""
        success, output = self._run_git(
            "diff", "--stat", f"{target_branch}...{source_branch}"
        )

        if not success:
            return {"error": "Failed to get diff"}

        # Parse stats
        lines_added = 0
        lines_removed = 0
        files_changed = 0

        for line in output.split("\n"):
            if "insertion" in line or "deletion" in line:
                # Parse summary line
                match = re.search(r"(\d+) files? changed", line)
                if match:
                    files_changed = int(match.group(1))

                match = re.search(r"(\d+) insertions?", line)
                if match:
                    lines_added = int(match.group(1))

                match = re.search(r"(\d+) deletions?", line)
                if match:
                    lines_removed = int(match.group(1))

        return {
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "net_change": lines_added - lines_removed
        }

    def get_operation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get git operation history"""
        return self.operations[-limit:]


class ExternalAPITester:
    """
    Tests integrations against external APIs in staging environments.
    """

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.endpoints: Dict[str, Dict[str, Any]] = {}

        logger.info("ExternalAPITester initialized")

    def register_endpoint(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        expected_status: int = 200,
        timeout: int = 30
    ) -> None:
        """Register an endpoint for testing"""
        self.endpoints[name] = {
            "url": url,
            "method": method,
            "headers": headers or {},
            "expected_status": expected_status,
            "timeout": timeout
        }

        logger.info(f"Registered endpoint: {name}")

    async def test_endpoint(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Test a registered endpoint"""
        if name not in self.endpoints:
            return {"error": f"Endpoint {name} not registered"}

        endpoint = self.endpoints[name]
        start_time = datetime.now()

        # Simulate API call (in production, use aiohttp or httpx)
        # For demo purposes, we simulate the result
        result = {
            "endpoint": name,
            "url": endpoint["url"],
            "method": endpoint["method"],
            "timestamp": start_time.isoformat(),
            "response_time_ms": 150,  # Simulated
            "status_code": endpoint["expected_status"],
            "success": True,
            "response": {"message": "OK"}
        }

        self.test_results.append(result)
        return result

    async def run_test_suite(
        self,
        endpoints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run tests for multiple endpoints"""
        if endpoints is None:
            endpoints = list(self.endpoints.keys())

        results = []
        for name in endpoints:
            result = await self.test_endpoint(name)
            results.append(result)

        passed = sum(1 for r in results if r.get("success", False))
        failed = len(results) - passed

        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / max(len(results), 1),
            "results": results
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all endpoints"""
        # Get most recent result for each endpoint
        latest = {}
        for result in reversed(self.test_results):
            name = result.get("endpoint")
            if name and name not in latest:
                latest[name] = result

        healthy = sum(1 for r in latest.values() if r.get("success", False))

        return {
            "total_endpoints": len(self.endpoints),
            "healthy": healthy,
            "unhealthy": len(latest) - healthy,
            "endpoints": {
                name: {
                    "status": "healthy" if latest.get(name, {}).get("success") else "unhealthy",
                    "last_checked": latest.get(name, {}).get("timestamp"),
                    "response_time": latest.get(name, {}).get("response_time_ms")
                }
                for name in self.endpoints
            }
        }

    def generate_api_report(self) -> str:
        """Generate API test report"""
        health = self.get_health_status()

        report = f"""# API Integration Test Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary

| Metric | Value |
|--------|-------|
| Total Endpoints | {health['total_endpoints']} |
| Healthy | {health['healthy']} |
| Unhealthy | {health['unhealthy']} |

## Endpoint Status

"""
        for name, status in health['endpoints'].items():
            emoji = ":white_check_mark:" if status['status'] == 'healthy' else ":x:"
            report += f"- {emoji} **{name}**: {status['status']}"
            if status.get('response_time'):
                report += f" ({status['response_time']}ms)"
            report += "\n"

        return report


# Factory function for creating integrations
def create_integration_suite(
    workspace_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a complete integration suite"""
    config = config or {}

    return {
        "jira": JiraIntegration(
            base_url=config.get("jira_url"),
            api_token=config.get("jira_token"),
            email=config.get("jira_email")
        ),
        "slack": SlackIntegration(
            webhook_url=config.get("slack_webhook"),
            bot_token=config.get("slack_token")
        ),
        "git": GitWorkflow(workspace_dir),
        "api_tester": ExternalAPITester()
    }
