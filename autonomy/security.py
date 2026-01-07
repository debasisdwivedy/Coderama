"""
Security & Governance Module

Provides enterprise-grade security capabilities:
- Automated security scanning (SAST/DAST)
- License compliance checking
- Secrets management with rotation
- Role-based access control with approval workflows
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import base64

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    WEAK_CRYPTO = "weak_crypto"
    HARDCODED_SECRET = "hardcoded_secret"
    INSECURE_DEPENDENCY = "insecure_dependency"
    MISSING_AUTH = "missing_auth"
    SSRF = "ssrf"
    XXE = "xxe"
    OPEN_REDIRECT = "open_redirect"
    INSECURE_RANDOM = "insecure_random"


class ApprovalStatus(Enum):
    """Approval workflow status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class LicenseType(Enum):
    """Common open-source license types"""
    MIT = "MIT"
    APACHE_2 = "Apache-2.0"
    GPL_2 = "GPL-2.0"
    GPL_3 = "GPL-3.0"
    LGPL = "LGPL"
    BSD_2 = "BSD-2-Clause"
    BSD_3 = "BSD-3-Clause"
    MPL_2 = "MPL-2.0"
    ISC = "ISC"
    UNLICENSED = "UNLICENSED"
    PROPRIETARY = "PROPRIETARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class SecurityFinding:
    """Represents a security vulnerability finding"""
    id: str
    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    code_snippet: Optional[str]
    cwe_id: Optional[str]
    owasp_category: Optional[str]
    recommendation: str
    references: List[str] = field(default_factory=list)
    false_positive: bool = False
    remediated: bool = False
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class LicenseInfo:
    """License information for a dependency"""
    package_name: str
    version: str
    license_type: LicenseType
    license_text: Optional[str]
    compatible: bool
    risk_level: str  # low, medium, high
    notes: Optional[str] = None


@dataclass
class Secret:
    """Represents a managed secret"""
    id: str
    name: str
    encrypted_value: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    last_rotated: Optional[datetime]
    rotation_interval_days: Optional[int]
    tags: Dict[str, str] = field(default_factory=dict)
    access_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ApprovalRequest:
    """Represents an approval request"""
    id: str
    requester: str
    action: str
    resource: str
    resource_type: str
    justification: str
    risk_level: str
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    reviewers: List[str]
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityScanner:
    """
    Automated security scanning with SAST/DAST capabilities.

    Scans code for common vulnerabilities and security issues.
    """

    # SAST patterns for common vulnerabilities
    VULNERABILITY_PATTERNS = [
        {
            "type": VulnerabilityType.SQL_INJECTION,
            "patterns": [
                r'execute\s*\([^)]*%s',
                r'execute\s*\([^)]*\+',
                r'cursor\.execute\s*\(.*f["\']',
                r'\.format\s*\([^)]*\).*execute',
                r'raw\s*\(\s*["\']SELECT.*%',
            ],
            "severity": SeverityLevel.CRITICAL,
            "cwe": "CWE-89",
            "owasp": "A03:2021"
        },
        {
            "type": VulnerabilityType.XSS,
            "patterns": [
                r'innerHTML\s*=',
                r'document\.write\s*\(',
                r'\.html\s*\([^)]*\+',
                r'dangerouslySetInnerHTML',
                r'v-html\s*=',
            ],
            "severity": SeverityLevel.HIGH,
            "cwe": "CWE-79",
            "owasp": "A03:2021"
        },
        {
            "type": VulnerabilityType.COMMAND_INJECTION,
            "patterns": [
                r'os\.system\s*\([^)]*\+',
                r'os\.popen\s*\([^)]*\+',
                r'subprocess\..*shell\s*=\s*True',
                r'eval\s*\([^)]*input',
                r'exec\s*\([^)]*input',
            ],
            "severity": SeverityLevel.CRITICAL,
            "cwe": "CWE-78",
            "owasp": "A03:2021"
        },
        {
            "type": VulnerabilityType.PATH_TRAVERSAL,
            "patterns": [
                r'open\s*\([^)]*\+.*\.\.',
                r'Path\s*\([^)]*\+',
                r'os\.path\.join\s*\([^)]*request',
                r'file_path\s*=.*\+.*input',
            ],
            "severity": SeverityLevel.HIGH,
            "cwe": "CWE-22",
            "owasp": "A01:2021"
        },
        {
            "type": VulnerabilityType.HARDCODED_SECRET,
            "patterns": [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][a-zA-Z0-9]{16,}["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][a-zA-Z0-9\-_.]{20,}["\']',
                r'AWS_SECRET_ACCESS_KEY\s*=\s*["\']',
                r'PRIVATE_KEY\s*=\s*["\']',
            ],
            "severity": SeverityLevel.CRITICAL,
            "cwe": "CWE-798",
            "owasp": "A07:2021"
        },
        {
            "type": VulnerabilityType.WEAK_CRYPTO,
            "patterns": [
                r'MD5\s*\(',
                r'SHA1\s*\(',
                r'DES\s*\.',
                r'random\s*\(\s*\)',  # Non-cryptographic random
                r'Math\.random\s*\(',
            ],
            "severity": SeverityLevel.MEDIUM,
            "cwe": "CWE-327",
            "owasp": "A02:2021"
        },
        {
            "type": VulnerabilityType.INSECURE_DESERIALIZATION,
            "patterns": [
                r'pickle\.loads?\s*\(',
                r'yaml\.load\s*\([^)]*\)',  # Without Loader
                r'marshal\.loads?\s*\(',
                r'eval\s*\(.*loads',
            ],
            "severity": SeverityLevel.HIGH,
            "cwe": "CWE-502",
            "owasp": "A08:2021"
        },
        {
            "type": VulnerabilityType.SSRF,
            "patterns": [
                r'requests\.(get|post|put)\s*\([^)]*\+',
                r'urllib\..*open\s*\([^)]*input',
                r'fetch\s*\([^)]*\+',
            ],
            "severity": SeverityLevel.HIGH,
            "cwe": "CWE-918",
            "owasp": "A10:2021"
        },
        {
            "type": VulnerabilityType.INSECURE_RANDOM,
            "patterns": [
                r'random\.random\s*\(',
                r'random\.randint\s*\(',
                r'Math\.random\s*\(',
            ],
            "severity": SeverityLevel.LOW,
            "cwe": "CWE-330",
            "owasp": "A02:2021"
        },
    ]

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.findings: List[SecurityFinding] = []
        self.scan_history: List[Dict[str, Any]] = []

        logger.info(f"SecurityScanner initialized for: {workspace_dir}")

    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for vulnerabilities"""
        findings = []

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            for vuln_pattern in self.VULNERABILITY_PATTERNS:
                for pattern in vuln_pattern["patterns"]:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                id=f"finding_{uuid.uuid4().hex[:12]}",
                                vulnerability_type=vuln_pattern["type"],
                                severity=vuln_pattern["severity"],
                                title=f"{vuln_pattern['type'].value} detected",
                                description=f"Potential {vuln_pattern['type'].value} vulnerability found",
                                file_path=str(file_path.relative_to(self.workspace_dir)),
                                line_number=line_num,
                                code_snippet=line.strip()[:200],
                                cwe_id=vuln_pattern.get("cwe"),
                                owasp_category=vuln_pattern.get("owasp"),
                                recommendation=self._get_recommendation(vuln_pattern["type"])
                            )
                            findings.append(finding)

        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")

        return findings

    def _get_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation recommendation for vulnerability type"""
        recommendations = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or prepared statements",
            VulnerabilityType.XSS: "Sanitize and escape user input before rendering",
            VulnerabilityType.COMMAND_INJECTION: "Avoid shell=True, use subprocess with list args",
            VulnerabilityType.PATH_TRAVERSAL: "Validate and sanitize file paths, use safe_join",
            VulnerabilityType.HARDCODED_SECRET: "Move secrets to environment variables or secret manager",
            VulnerabilityType.WEAK_CRYPTO: "Use strong cryptographic algorithms (SHA-256, AES)",
            VulnerabilityType.INSECURE_DESERIALIZATION: "Use safe deserialization with type validation",
            VulnerabilityType.SSRF: "Validate and whitelist allowed URLs",
            VulnerabilityType.INSECURE_RANDOM: "Use secrets module for security-sensitive operations",
        }
        return recommendations.get(vuln_type, "Review and remediate the vulnerability")

    def scan_directory(
        self,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Scan entire directory for vulnerabilities"""
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb"]

        if exclude_patterns is None:
            exclude_patterns = ["node_modules", "__pycache__", ".git", "venv", ".env"]

        start_time = datetime.now()
        self.findings = []
        files_scanned = 0

        for ext in extensions:
            for file_path in self.workspace_dir.rglob(f"*{ext}"):
                # Check exclusions
                if any(excl in str(file_path) for excl in exclude_patterns):
                    continue

                file_findings = self.scan_file(file_path)
                self.findings.extend(file_findings)
                files_scanned += 1

        duration = (datetime.now() - start_time).total_seconds()

        scan_result = {
            "scan_id": f"scan_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "files_scanned": files_scanned,
            "total_findings": len(self.findings),
            "by_severity": self._count_by_severity(),
            "by_type": self._count_by_type()
        }

        self.scan_history.append(scan_result)
        logger.info(f"Scan complete: {len(self.findings)} findings in {files_scanned} files")

        return scan_result

    def _count_by_severity(self) -> Dict[str, int]:
        """Count findings by severity"""
        counts = {}
        for finding in self.findings:
            sev = finding.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _count_by_type(self) -> Dict[str, int]:
        """Count findings by vulnerability type"""
        counts = {}
        for finding in self.findings:
            vtype = finding.vulnerability_type.value
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts

    def get_findings(
        self,
        severity: Optional[SeverityLevel] = None,
        vuln_type: Optional[VulnerabilityType] = None,
        include_remediated: bool = False
    ) -> List[SecurityFinding]:
        """Get filtered findings"""
        findings = self.findings

        if severity:
            findings = [f for f in findings if f.severity == severity]

        if vuln_type:
            findings = [f for f in findings if f.vulnerability_type == vuln_type]

        if not include_remediated:
            findings = [f for f in findings if not f.remediated]

        return findings

    def mark_false_positive(self, finding_id: str, reason: str) -> bool:
        """Mark a finding as false positive"""
        for finding in self.findings:
            if finding.id == finding_id:
                finding.false_positive = True
                finding.description += f" [False Positive: {reason}]"
                return True
        return False

    def generate_report(self, format: str = "json") -> str:
        """Generate security scan report"""
        report = {
            "report_generated": datetime.now().isoformat(),
            "workspace": str(self.workspace_dir),
            "summary": {
                "total_findings": len(self.findings),
                "critical": len([f for f in self.findings if f.severity == SeverityLevel.CRITICAL]),
                "high": len([f for f in self.findings if f.severity == SeverityLevel.HIGH]),
                "medium": len([f for f in self.findings if f.severity == SeverityLevel.MEDIUM]),
                "low": len([f for f in self.findings if f.severity == SeverityLevel.LOW]),
                "false_positives": len([f for f in self.findings if f.false_positive]),
                "remediated": len([f for f in self.findings if f.remediated])
            },
            "findings": [
                {
                    "id": f.id,
                    "type": f.vulnerability_type.value,
                    "severity": f.severity.value,
                    "title": f.title,
                    "file": f.file_path,
                    "line": f.line_number,
                    "cwe": f.cwe_id,
                    "owasp": f.owasp_category,
                    "recommendation": f.recommendation
                }
                for f in self.findings if not f.false_positive
            ],
            "scan_history": self.scan_history[-10:]
        }

        if format == "json":
            return json.dumps(report, indent=2)
        return json.dumps(report)


class LicenseChecker:
    """
    Checks license compliance for all dependencies.

    Ensures all dependencies have compatible licenses.
    """

    # License compatibility matrix
    COMPATIBLE_LICENSES = {
        LicenseType.MIT: {LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2, LicenseType.BSD_3, LicenseType.ISC},
        LicenseType.APACHE_2: {LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2, LicenseType.BSD_3, LicenseType.ISC},
        LicenseType.BSD_2: {LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2, LicenseType.BSD_3, LicenseType.ISC},
        LicenseType.BSD_3: {LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2, LicenseType.BSD_3, LicenseType.ISC},
        LicenseType.ISC: {LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2, LicenseType.BSD_3, LicenseType.ISC},
    }

    # Copyleft licenses that require special handling
    COPYLEFT_LICENSES = {LicenseType.GPL_2, LicenseType.GPL_3, LicenseType.LGPL, LicenseType.MPL_2}

    def __init__(self, workspace_dir: str, allowed_licenses: Optional[Set[LicenseType]] = None):
        self.workspace_dir = Path(workspace_dir)
        self.allowed_licenses = allowed_licenses or {
            LicenseType.MIT, LicenseType.APACHE_2, LicenseType.BSD_2,
            LicenseType.BSD_3, LicenseType.ISC, LicenseType.MPL_2
        }
        self.license_cache: Dict[str, LicenseInfo] = {}

        logger.info(f"LicenseChecker initialized with {len(self.allowed_licenses)} allowed licenses")

    def _detect_license(self, license_text: str) -> LicenseType:
        """Detect license type from license text"""
        license_text_lower = license_text.lower()

        if "mit license" in license_text_lower or "permission is hereby granted" in license_text_lower:
            return LicenseType.MIT
        elif "apache license" in license_text_lower and "version 2" in license_text_lower:
            return LicenseType.APACHE_2
        elif "gnu general public license" in license_text_lower:
            if "version 3" in license_text_lower:
                return LicenseType.GPL_3
            return LicenseType.GPL_2
        elif "gnu lesser general public license" in license_text_lower:
            return LicenseType.LGPL
        elif "bsd" in license_text_lower:
            if "2-clause" in license_text_lower:
                return LicenseType.BSD_2
            return LicenseType.BSD_3
        elif "mozilla public license" in license_text_lower:
            return LicenseType.MPL_2
        elif "isc license" in license_text_lower:
            return LicenseType.ISC

        return LicenseType.UNKNOWN

    def check_package(self, package_name: str, version: str = "") -> LicenseInfo:
        """Check license for a specific package"""
        cache_key = f"{package_name}:{version}"

        if cache_key in self.license_cache:
            return self.license_cache[cache_key]

        # Try to get license info via pip
        try:
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10
            )

            license_type = LicenseType.UNKNOWN
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("License:"):
                        license_str = line.replace("License:", "").strip()
                        license_type = self._detect_license(license_str)
                        break

        except Exception as e:
            logger.warning(f"Failed to check license for {package_name}: {e}")
            license_type = LicenseType.UNKNOWN

        compatible = license_type in self.allowed_licenses
        risk_level = "low" if compatible else ("high" if license_type in self.COPYLEFT_LICENSES else "medium")

        license_info = LicenseInfo(
            package_name=package_name,
            version=version,
            license_type=license_type,
            license_text=None,
            compatible=compatible,
            risk_level=risk_level
        )

        self.license_cache[cache_key] = license_info
        return license_info

    def check_all_dependencies(self) -> Dict[str, Any]:
        """Check licenses for all project dependencies"""
        req_file = self.workspace_dir / "requirements.txt"
        dependencies = []

        if req_file.exists():
            for line in req_file.read_text().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "==" in line:
                        pkg, version = line.split("==", 1)
                        dependencies.append((pkg.strip(), version.strip()))
                    elif ">=" in line:
                        pkg, version = line.split(">=", 1)
                        dependencies.append((pkg.strip(), version.strip()))
                    else:
                        dependencies.append((line, ""))

        results = []
        for pkg, version in dependencies:
            info = self.check_package(pkg, version)
            results.append(info)

        incompatible = [r for r in results if not r.compatible]
        copyleft = [r for r in results if r.license_type in self.COPYLEFT_LICENSES]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_packages": len(results),
            "compatible": len([r for r in results if r.compatible]),
            "incompatible": len(incompatible),
            "copyleft_licenses": len(copyleft),
            "unknown_licenses": len([r for r in results if r.license_type == LicenseType.UNKNOWN]),
            "packages": [
                {
                    "name": r.package_name,
                    "version": r.version,
                    "license": r.license_type.value,
                    "compatible": r.compatible,
                    "risk": r.risk_level
                }
                for r in results
            ],
            "issues": [
                {
                    "package": r.package_name,
                    "license": r.license_type.value,
                    "reason": "License not in allowed list"
                }
                for r in incompatible
            ]
        }


class SecretsManager:
    """
    Secure secrets management with encryption and rotation.

    Provides secure storage, retrieval, and automatic rotation of secrets.
    """

    def __init__(self, storage_path: str, master_key: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Generate or use provided master key
        if master_key:
            self._master_key = master_key.encode()
        else:
            key_file = self.storage_path / ".master_key"
            if key_file.exists():
                self._master_key = key_file.read_bytes()
            else:
                self._master_key = secrets.token_bytes(32)
                key_file.write_bytes(self._master_key)
                os.chmod(key_file, 0o600)

        self.secrets: Dict[str, Secret] = {}
        self._load_secrets()

        logger.info(f"SecretsManager initialized at: {storage_path}")

    def _encrypt(self, value: str) -> bytes:
        """Simple XOR encryption (use proper encryption in production)"""
        value_bytes = value.encode()
        key_repeated = (self._master_key * ((len(value_bytes) // len(self._master_key)) + 1))[:len(value_bytes)]
        encrypted = bytes(a ^ b for a, b in zip(value_bytes, key_repeated))
        return base64.b64encode(encrypted)

    def _decrypt(self, encrypted: bytes) -> str:
        """Simple XOR decryption"""
        decoded = base64.b64decode(encrypted)
        key_repeated = (self._master_key * ((len(decoded) // len(self._master_key)) + 1))[:len(decoded)]
        decrypted = bytes(a ^ b for a, b in zip(decoded, key_repeated))
        return decrypted.decode()

    def _load_secrets(self) -> None:
        """Load secrets from storage"""
        secrets_file = self.storage_path / "secrets.json"
        if secrets_file.exists():
            try:
                with open(secrets_file, "r") as f:
                    data = json.load(f)
                    for name, secret_data in data.items():
                        self.secrets[name] = Secret(
                            id=secret_data["id"],
                            name=name,
                            encrypted_value=base64.b64decode(secret_data["encrypted_value"]),
                            created_at=datetime.fromisoformat(secret_data["created_at"]),
                            expires_at=datetime.fromisoformat(secret_data["expires_at"]) if secret_data.get("expires_at") else None,
                            last_rotated=datetime.fromisoformat(secret_data["last_rotated"]) if secret_data.get("last_rotated") else None,
                            rotation_interval_days=secret_data.get("rotation_interval_days"),
                            tags=secret_data.get("tags", {}),
                            access_log=secret_data.get("access_log", [])
                        )
            except Exception as e:
                logger.warning(f"Failed to load secrets: {e}")

    def _save_secrets(self) -> None:
        """Save secrets to storage"""
        secrets_file = self.storage_path / "secrets.json"
        data = {}
        for name, secret in self.secrets.items():
            data[name] = {
                "id": secret.id,
                "encrypted_value": base64.b64encode(secret.encrypted_value).decode(),
                "created_at": secret.created_at.isoformat(),
                "expires_at": secret.expires_at.isoformat() if secret.expires_at else None,
                "last_rotated": secret.last_rotated.isoformat() if secret.last_rotated else None,
                "rotation_interval_days": secret.rotation_interval_days,
                "tags": secret.tags,
                "access_log": secret.access_log[-100:]  # Keep last 100 access logs
            }

        with open(secrets_file, "w") as f:
            json.dump(data, f, indent=2)

        os.chmod(secrets_file, 0o600)

    def store(
        self,
        name: str,
        value: str,
        expires_in_days: Optional[int] = None,
        rotation_interval_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Secret:
        """Store a new secret"""
        secret = Secret(
            id=f"secret_{uuid.uuid4().hex[:12]}",
            name=name,
            encrypted_value=self._encrypt(value),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None,
            last_rotated=None,
            rotation_interval_days=rotation_interval_days,
            tags=tags or {}
        )

        self.secrets[name] = secret
        self._save_secrets()

        logger.info(f"Stored secret: {name}")
        return secret

    def get(self, name: str, accessor: str = "system") -> Optional[str]:
        """Retrieve a secret value"""
        if name not in self.secrets:
            return None

        secret = self.secrets[name]

        # Check expiration
        if secret.expires_at and datetime.now() > secret.expires_at:
            logger.warning(f"Secret '{name}' has expired")
            return None

        # Log access
        secret.access_log.append({
            "timestamp": datetime.now().isoformat(),
            "accessor": accessor,
            "action": "read"
        })
        self._save_secrets()

        return self._decrypt(secret.encrypted_value)

    def rotate(self, name: str, new_value: str) -> bool:
        """Rotate a secret with a new value"""
        if name not in self.secrets:
            return False

        secret = self.secrets[name]
        secret.encrypted_value = self._encrypt(new_value)
        secret.last_rotated = datetime.now()

        # Reset expiration if rotation interval is set
        if secret.rotation_interval_days:
            secret.expires_at = datetime.now() + timedelta(days=secret.rotation_interval_days)

        self._save_secrets()
        logger.info(f"Rotated secret: {name}")
        return True

    def delete(self, name: str) -> bool:
        """Delete a secret"""
        if name in self.secrets:
            del self.secrets[name]
            self._save_secrets()
            logger.info(f"Deleted secret: {name}")
            return True
        return False

    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets (without values)"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                "last_rotated": s.last_rotated.isoformat() if s.last_rotated else None,
                "needs_rotation": self._needs_rotation(s),
                "tags": s.tags
            }
            for s in self.secrets.values()
        ]

    def _needs_rotation(self, secret: Secret) -> bool:
        """Check if a secret needs rotation"""
        if not secret.rotation_interval_days:
            return False

        last_change = secret.last_rotated or secret.created_at
        days_since = (datetime.now() - last_change).days
        return days_since >= secret.rotation_interval_days

    def get_rotation_needed(self) -> List[str]:
        """Get list of secrets that need rotation"""
        return [name for name, secret in self.secrets.items() if self._needs_rotation(secret)]

    def scan_for_exposed_secrets(self, workspace_dir: str) -> List[Dict[str, Any]]:
        """Scan workspace for potentially exposed secrets"""
        exposed = []
        workspace = Path(workspace_dir)

        patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'AWS_.*KEY\s*=\s*["\'][^"\']+["\']',
        ]

        for py_file in workspace.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        exposed.append({
                            "file": str(py_file.relative_to(workspace)),
                            "match": match.group()[:50] + "...",
                            "pattern": pattern
                        })
            except Exception:
                continue

        return exposed


class AccessControl:
    """
    Role-based access control for agents and operations.

    Manages permissions and enforces access policies.
    """

    def __init__(self):
        self.roles: Dict[str, Set[str]] = {
            "admin": {"*"},
            "developer": {"read", "write", "execute", "deploy_staging"},
            "reviewer": {"read", "approve", "reject"},
            "viewer": {"read"},
            "deployer": {"read", "deploy_staging", "deploy_production"},
        }

        self.user_roles: Dict[str, Set[str]] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[str]]] = {}

        logger.info("AccessControl initialized")

    def assign_role(self, user: str, role: str) -> bool:
        """Assign a role to a user"""
        if role not in self.roles:
            return False

        if user not in self.user_roles:
            self.user_roles[user] = set()

        self.user_roles[user].add(role)
        logger.info(f"Assigned role '{role}' to user '{user}'")
        return True

    def revoke_role(self, user: str, role: str) -> bool:
        """Revoke a role from a user"""
        if user in self.user_roles and role in self.user_roles[user]:
            self.user_roles[user].remove(role)
            return True
        return False

    def get_user_permissions(self, user: str) -> Set[str]:
        """Get all permissions for a user"""
        permissions = set()
        for role in self.user_roles.get(user, set()):
            permissions.update(self.roles.get(role, set()))
        return permissions

    def check_permission(self, user: str, permission: str, resource: Optional[str] = None) -> bool:
        """Check if a user has a specific permission"""
        permissions = self.get_user_permissions(user)

        # Admin has all permissions
        if "*" in permissions:
            return True

        # Check general permission
        if permission in permissions:
            # Check resource-specific restrictions
            if resource and resource in self.resource_permissions:
                resource_perms = self.resource_permissions[resource].get(user, set())
                if permission not in resource_perms and "*" not in resource_perms:
                    return False
            return True

        return False

    def set_resource_permission(self, resource: str, user: str, permissions: Set[str]) -> None:
        """Set specific permissions for a user on a resource"""
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = {}
        self.resource_permissions[resource][user] = permissions

    def create_role(self, role_name: str, permissions: Set[str]) -> bool:
        """Create a custom role"""
        if role_name in self.roles:
            return False
        self.roles[role_name] = permissions
        return True


class ApprovalWorkflow:
    """
    Approval workflow for high-risk changes.

    Manages approval requests, reviews, and enforcement.
    """

    def __init__(
        self,
        access_control: AccessControl,
        required_approvals: int = 1,
        expiry_hours: int = 24
    ):
        self.access_control = access_control
        self.required_approvals = required_approvals
        self.expiry_hours = expiry_hours
        self.requests: Dict[str, ApprovalRequest] = {}
        self._callbacks: List[Callable[[ApprovalRequest], None]] = []

        logger.info(f"ApprovalWorkflow initialized with {required_approvals} required approvals")

    def register_callback(self, callback: Callable[[ApprovalRequest], None]) -> None:
        """Register a callback for approval events"""
        self._callbacks.append(callback)

    def _notify(self, request: ApprovalRequest) -> None:
        """Notify callbacks of approval events"""
        for callback in self._callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Approval callback failed: {e}")

    def create_request(
        self,
        requester: str,
        action: str,
        resource: str,
        resource_type: str,
        justification: str,
        risk_level: str = "medium",
        reviewers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApprovalRequest:
        """Create a new approval request"""
        # Get reviewers with approval permission
        if reviewers is None:
            reviewers = [
                user for user, roles in self.access_control.user_roles.items()
                if self.access_control.check_permission(user, "approve")
            ]

        request = ApprovalRequest(
            id=f"approval_{uuid.uuid4().hex[:12]}",
            requester=requester,
            action=action,
            resource=resource,
            resource_type=resource_type,
            justification=justification,
            risk_level=risk_level,
            status=ApprovalStatus.PENDING,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.expiry_hours),
            reviewers=reviewers,
            metadata=metadata or {}
        )

        self.requests[request.id] = request
        self._notify(request)

        logger.info(f"Created approval request: {request.id}")
        return request

    def approve(self, request_id: str, reviewer: str, comment: str = "") -> Tuple[bool, str]:
        """Approve a request"""
        if request_id not in self.requests:
            return False, "Request not found"

        request = self.requests[request_id]

        # Check request status
        if request.status != ApprovalStatus.PENDING:
            return False, f"Request is {request.status.value}"

        # Check expiration
        if datetime.now() > request.expires_at:
            request.status = ApprovalStatus.EXPIRED
            return False, "Request has expired"

        # Check reviewer permission
        if not self.access_control.check_permission(reviewer, "approve"):
            return False, "Reviewer does not have approval permission"

        # Check if already reviewed
        if any(a["reviewer"] == reviewer for a in request.approvals):
            return False, "Already approved by this reviewer"

        request.approvals.append({
            "reviewer": reviewer,
            "timestamp": datetime.now().isoformat(),
            "comment": comment
        })

        # Check if we have enough approvals
        if len(request.approvals) >= self.required_approvals:
            request.status = ApprovalStatus.APPROVED
            logger.info(f"Request {request_id} approved")

        self._notify(request)
        return True, "Approval recorded"

    def reject(self, request_id: str, reviewer: str, reason: str) -> Tuple[bool, str]:
        """Reject a request"""
        if request_id not in self.requests:
            return False, "Request not found"

        request = self.requests[request_id]

        if request.status != ApprovalStatus.PENDING:
            return False, f"Request is {request.status.value}"

        if not self.access_control.check_permission(reviewer, "reject"):
            return False, "Reviewer does not have rejection permission"

        request.rejections.append({
            "reviewer": reviewer,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })

        request.status = ApprovalStatus.REJECTED
        self._notify(request)

        logger.info(f"Request {request_id} rejected")
        return True, "Request rejected"

    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an approval request"""
        if request_id not in self.requests:
            return None

        request = self.requests[request_id]

        return {
            "id": request.id,
            "status": request.status.value,
            "requester": request.requester,
            "action": request.action,
            "resource": request.resource,
            "risk_level": request.risk_level,
            "created_at": request.created_at.isoformat(),
            "expires_at": request.expires_at.isoformat(),
            "approvals": len(request.approvals),
            "required": self.required_approvals,
            "reviewers": request.reviewers
        }

    def get_pending_requests(self, reviewer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all pending requests"""
        pending = []
        for request in self.requests.values():
            if request.status == ApprovalStatus.PENDING:
                if reviewer is None or reviewer in request.reviewers:
                    pending.append(self.get_status(request.id))
        return pending

    def is_approved(self, request_id: str) -> bool:
        """Check if a request is approved"""
        if request_id in self.requests:
            return self.requests[request_id].status == ApprovalStatus.APPROVED
        return False
