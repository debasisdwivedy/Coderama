"""
Intelligent Context Sharing Module

Provides intelligent context management:
- Vector database for semantic search
- Automatic documentation generation
- Cross-project learning
- Conflict resolution
"""

import hashlib
import json
import logging
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    source: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result of a semantic search"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    highlight: Optional[str] = None


@dataclass
class ConflictResolution:
    """Resolution of a conflict between suggestions"""
    id: str
    conflict_type: str
    options: List[Dict[str, Any]]
    resolution: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: datetime


class VectorKnowledgeBase:
    """
    Vector database for semantic search across project knowledge.

    Stores and retrieves documents using vector similarity.
    """

    def __init__(self, storage_path: Optional[str] = None, embedding_dim: int = 384):
        self.storage_path = Path(storage_path) if storage_path else None
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, Document] = {}
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load()

        logger.info(f"VectorKnowledgeBase initialized (dim={embedding_dim})")

    def _load(self) -> None:
        """Load documents from storage"""
        if not self.storage_path:
            return

        docs_file = self.storage_path / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file, "r") as f:
                    data = json.load(f)
                    for doc_data in data:
                        doc = Document(
                            id=doc_data["id"],
                            content=doc_data["content"],
                            embedding=doc_data["embedding"],
                            metadata=doc_data["metadata"],
                            source=doc_data["source"],
                            created_at=datetime.fromisoformat(doc_data["created_at"]),
                            updated_at=datetime.fromisoformat(doc_data["updated_at"]),
                            tags=doc_data.get("tags", [])
                        )
                        self.documents[doc.id] = doc
                        self._index_document(doc)

                logger.info(f"Loaded {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load documents: {e}")

    def _save(self) -> None:
        """Save documents to storage"""
        if not self.storage_path:
            return

        docs_file = self.storage_path / "documents.json"
        data = [
            {
                "id": doc.id,
                "content": doc.content,
                "embedding": doc.embedding,
                "metadata": doc.metadata,
                "source": doc.source,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "tags": doc.tags
            }
            for doc in self.documents.values()
        ]

        with open(docs_file, "w") as f:
            json.dump(data, f)

    def _simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding (for demo - use real embeddings in production)"""
        # This is a simple hash-based embedding for demonstration
        # In production, use sentence-transformers or similar
        words = text.lower().split()
        embedding = [0.0] * self.embedding_dim

        for i, word in enumerate(words[:100]):
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for j in range(min(len(word), self.embedding_dim)):
                idx = (word_hash + j) % self.embedding_dim
                embedding[idx] += 1.0 / (i + 1)

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def _index_document(self, doc: Document) -> None:
        """Add document to inverted index"""
        words = set(doc.content.lower().split())
        for word in words:
            if len(word) > 2:
                self.inverted_index[word].add(doc.id)

        for tag in doc.tags:
            self.inverted_index[f"tag:{tag}"].add(doc.id)

    def add_document(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None
    ) -> Document:
        """Add a document to the knowledge base"""
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        if embedding is None:
            embedding = self._simple_embedding(content)

        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            source=source,
            created_at=now,
            updated_at=now,
            tags=tags or []
        )

        self.documents[doc_id] = doc
        self._index_document(doc)
        self._save()

        logger.debug(f"Added document: {doc_id}")
        return doc

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.1,
        filter_tags: Optional[List[str]] = None,
        filter_source: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        query_embedding = self._simple_embedding(query)

        results = []

        for doc in self.documents.values():
            # Apply filters
            if filter_tags and not any(tag in doc.tags for tag in filter_tags):
                continue
            if filter_source and doc.source != filter_source:
                continue

            score = self._cosine_similarity(query_embedding, doc.embedding)

            if score >= min_score:
                # Generate highlight
                query_words = set(query.lower().split())
                content_words = doc.content.split()
                highlight_parts = []

                for i, word in enumerate(content_words):
                    if word.lower() in query_words:
                        start = max(0, i - 3)
                        end = min(len(content_words), i + 4)
                        highlight_parts.append(" ".join(content_words[start:end]))

                highlight = " ... ".join(highlight_parts[:2]) if highlight_parts else None

                results.append(SearchResult(
                    document_id=doc.id,
                    content=doc.content[:500],
                    score=score,
                    metadata=doc.metadata,
                    source=doc.source,
                    highlight=highlight
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def get_related_documents(
        self,
        document_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Get documents related to a specific document"""
        if document_id not in self.documents:
            return []

        doc = self.documents[document_id]
        return self.search(doc.content, limit=limit + 1)[1:]  # Exclude self

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the knowledge base"""
        if document_id in self.documents:
            doc = self.documents[document_id]

            # Remove from inverted index
            words = set(doc.content.lower().split())
            for word in words:
                self.inverted_index[word].discard(document_id)

            for tag in doc.tags:
                self.inverted_index[f"tag:{tag}"].discard(document_id)

            del self.documents[document_id]
            self._save()
            return True

        return False

    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Document]:
        """Update an existing document"""
        if document_id not in self.documents:
            return None

        doc = self.documents[document_id]

        if content is not None:
            doc.content = content
            doc.embedding = self._simple_embedding(content)

        if metadata is not None:
            doc.metadata.update(metadata)

        if tags is not None:
            doc.tags = tags

        doc.updated_at = datetime.now()
        self._index_document(doc)
        self._save()

        return doc

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        sources = defaultdict(int)
        tags = defaultdict(int)

        for doc in self.documents.values():
            sources[doc.source] += 1
            for tag in doc.tags:
                tags[tag] += 1

        return {
            "total_documents": len(self.documents),
            "by_source": dict(sources),
            "by_tag": dict(tags),
            "index_terms": len(self.inverted_index)
        }


class DocumentationGenerator:
    """
    Generates and updates documentation automatically.

    Creates documentation from code and keeps it synchronized.
    """

    def __init__(self, workspace_dir: str, knowledge_base: VectorKnowledgeBase):
        self.workspace_dir = Path(workspace_dir)
        self.knowledge_base = knowledge_base
        self.doc_templates = self._load_templates()

        logger.info(f"DocumentationGenerator initialized for: {workspace_dir}")

    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates"""
        return {
            "module": """# {module_name}

## Overview
{overview}

## Functions

{functions}

## Classes

{classes}

## Usage Examples

{examples}

---
*Auto-generated documentation*
""",
            "function": """### `{name}`

{docstring}

**Parameters:**
{parameters}

**Returns:**
{returns}

**Example:**
```python
{example}
```
""",
            "class": """### `{name}`

{docstring}

**Attributes:**
{attributes}

**Methods:**
{methods}
""",
            "api": """## {endpoint}

**Method:** `{method}`

**Description:** {description}

**Request:**
```json
{request_body}
```

**Response:**
```json
{response_body}
```

**Status Codes:**
{status_codes}
"""
        }

    def generate_module_docs(
        self,
        module_path: str,
        code: str
    ) -> str:
        """Generate documentation for a Python module"""
        # Extract module components
        functions = self._extract_functions(code)
        classes = self._extract_classes(code)
        module_docstring = self._extract_module_docstring(code)

        # Generate function docs
        func_docs = []
        for func in functions:
            func_doc = self.doc_templates["function"].format(
                name=func["name"],
                docstring=func.get("docstring", "No description"),
                parameters=self._format_parameters(func.get("params", [])),
                returns=func.get("returns", "None"),
                example=f'{func["name"]}()'
            )
            func_docs.append(func_doc)

        # Generate class docs
        class_docs = []
        for cls in classes:
            class_doc = self.doc_templates["class"].format(
                name=cls["name"],
                docstring=cls.get("docstring", "No description"),
                attributes=self._format_attributes(cls.get("attributes", [])),
                methods=self._format_methods(cls.get("methods", []))
            )
            class_docs.append(class_doc)

        # Generate full module documentation
        module_doc = self.doc_templates["module"].format(
            module_name=Path(module_path).stem,
            overview=module_docstring or "Module documentation",
            functions="\n".join(func_docs) if func_docs else "No functions",
            classes="\n".join(class_docs) if class_docs else "No classes",
            examples="See function documentation above"
        )

        # Add to knowledge base
        self.knowledge_base.add_document(
            content=module_doc,
            source=module_path,
            metadata={"type": "module_doc"},
            tags=["documentation", "module", Path(module_path).stem]
        )

        return module_doc

    def _extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code"""
        functions = []
        pattern = r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?\s*:'

        for match in re.finditer(pattern, code):
            name = match.group(1)
            params_str = match.group(2)
            returns = match.group(3) if match.group(3) else "None"

            # Parse parameters
            params = []
            if params_str.strip():
                for param in params_str.split(","):
                    param = param.strip()
                    if param and param != "self":
                        params.append(param)

            # Try to find docstring
            func_start = match.end()
            docstring_match = re.search(
                r'"""([^"]*)"""',
                code[func_start:func_start + 500]
            )
            docstring = docstring_match.group(1).strip() if docstring_match else None

            functions.append({
                "name": name,
                "params": params,
                "returns": returns.strip(),
                "docstring": docstring
            })

        return functions

    def _extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """Extract class definitions from code"""
        classes = []
        pattern = r'class\s+(\w+)(?:\([^)]*\))?\s*:'

        for match in re.finditer(pattern, code):
            name = match.group(1)

            # Try to find docstring
            class_start = match.end()
            docstring_match = re.search(
                r'"""([^"]*)"""',
                code[class_start:class_start + 500]
            )
            docstring = docstring_match.group(1).strip() if docstring_match else None

            classes.append({
                "name": name,
                "docstring": docstring,
                "attributes": [],
                "methods": []
            })

        return classes

    def _extract_module_docstring(self, code: str) -> Optional[str]:
        """Extract module-level docstring"""
        match = re.match(r'^"""([^"]*)"""', code.strip())
        return match.group(1).strip() if match else None

    def _format_parameters(self, params: List[str]) -> str:
        """Format parameter list"""
        if not params:
            return "None"
        return "\n".join(f"- `{p}`" for p in params)

    def _format_attributes(self, attributes: List[str]) -> str:
        """Format attribute list"""
        if not attributes:
            return "None"
        return "\n".join(f"- `{a}`" for a in attributes)

    def _format_methods(self, methods: List[str]) -> str:
        """Format method list"""
        if not methods:
            return "See class documentation"
        return "\n".join(f"- `{m}()`" for m in methods)

    def generate_api_docs(
        self,
        endpoints: List[Dict[str, Any]]
    ) -> str:
        """Generate API documentation"""
        docs = ["# API Documentation\n"]

        for endpoint in endpoints:
            endpoint_doc = self.doc_templates["api"].format(
                endpoint=endpoint.get("path", "/"),
                method=endpoint.get("method", "GET"),
                description=endpoint.get("description", "No description"),
                request_body=json.dumps(endpoint.get("request", {}), indent=2),
                response_body=json.dumps(endpoint.get("response", {}), indent=2),
                status_codes=self._format_status_codes(endpoint.get("status_codes", {}))
            )
            docs.append(endpoint_doc)

        full_doc = "\n".join(docs)

        # Add to knowledge base
        self.knowledge_base.add_document(
            content=full_doc,
            source="api",
            metadata={"type": "api_doc"},
            tags=["documentation", "api"]
        )

        return full_doc

    def _format_status_codes(self, codes: Dict[int, str]) -> str:
        """Format status codes"""
        if not codes:
            return "- `200`: Success"
        return "\n".join(f"- `{code}`: {desc}" for code, desc in codes.items())

    def sync_documentation(
        self,
        source_files: List[str]
    ) -> Dict[str, Any]:
        """Synchronize documentation with source code"""
        updated = []
        created = []
        unchanged = []

        for source_file in source_files:
            file_path = self.workspace_dir / source_file
            if not file_path.exists():
                continue

            code = file_path.read_text()
            code_hash = hashlib.md5(code.encode()).hexdigest()

            # Check if documentation exists and is current
            existing = self.knowledge_base.search(
                f"module {source_file}",
                filter_source=source_file,
                limit=1
            )

            if existing:
                doc = self.knowledge_base.documents.get(existing[0].document_id)
                if doc and doc.metadata.get("code_hash") == code_hash:
                    unchanged.append(source_file)
                    continue
                else:
                    updated.append(source_file)
            else:
                created.append(source_file)

            # Generate new documentation
            doc_content = self.generate_module_docs(source_file, code)

        return {
            "updated": updated,
            "created": created,
            "unchanged": unchanged,
            "total_processed": len(source_files)
        }


class CrossProjectLearning:
    """
    Enables learning and knowledge sharing across different projects.

    Identifies patterns and insights that can be applied across projects.
    """

    def __init__(self, knowledge_base: VectorKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.project_insights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info("CrossProjectLearning initialized")

    def register_project(
        self,
        project_name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Register a project for cross-learning"""
        self.project_insights[project_name].append({
            "type": "registration",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })

        # Add project metadata to knowledge base
        self.knowledge_base.add_document(
            content=f"Project: {project_name}. {json.dumps(metadata)}",
            source=f"project:{project_name}",
            metadata={"type": "project_metadata", "project": project_name},
            tags=["project", project_name]
        )

    def share_insight(
        self,
        project_name: str,
        insight_type: str,
        content: str,
        tags: Optional[List[str]] = None
    ) -> None:
        """Share an insight from a project"""
        insight = {
            "type": insight_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or []
        }

        self.project_insights[project_name].append(insight)

        # Add to knowledge base for cross-project search
        self.knowledge_base.add_document(
            content=content,
            source=f"project:{project_name}",
            metadata={"type": "insight", "insight_type": insight_type, "project": project_name},
            tags=["insight", project_name] + (tags or [])
        )

    def find_relevant_insights(
        self,
        query: str,
        exclude_project: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find insights from other projects relevant to a query"""
        results = self.knowledge_base.search(
            query,
            limit=limit * 2,
            filter_tags=["insight"]
        )

        relevant = []
        for result in results:
            project = result.metadata.get("project")
            if exclude_project and project == exclude_project:
                continue

            relevant.append({
                "project": project,
                "insight_type": result.metadata.get("insight_type"),
                "content": result.content,
                "relevance_score": result.score
            })

            if len(relevant) >= limit:
                break

        return relevant

    def get_project_similarities(
        self,
        project_name: str
    ) -> List[Dict[str, Any]]:
        """Find projects similar to the given project"""
        # Get project documents
        project_docs = self.knowledge_base.search(
            f"project {project_name}",
            filter_source=f"project:{project_name}",
            limit=5
        )

        if not project_docs:
            return []

        # Search for similar projects
        combined_content = " ".join(d.content for d in project_docs)
        similar = self.knowledge_base.search(
            combined_content,
            limit=20,
            filter_tags=["project"]
        )

        # Group by project
        project_scores = defaultdict(list)
        for result in similar:
            proj = result.metadata.get("project")
            if proj and proj != project_name:
                project_scores[proj].append(result.score)

        # Calculate average similarity
        similarities = []
        for proj, scores in project_scores.items():
            avg_score = sum(scores) / len(scores)
            similarities.append({
                "project": proj,
                "similarity_score": avg_score,
                "matching_documents": len(scores)
            })

        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:5]

    def suggest_reusable_components(
        self,
        project_name: str,
        requirements: str
    ) -> List[Dict[str, Any]]:
        """Suggest components from other projects that could be reused"""
        # Search for relevant components
        results = self.knowledge_base.search(
            requirements,
            limit=20,
            filter_tags=["component", "pattern", "utility"]
        )

        suggestions = []
        seen_components = set()

        for result in results:
            component_id = result.metadata.get("component_id", result.document_id)
            if component_id in seen_components:
                continue

            project = result.metadata.get("project")
            if project == project_name:
                continue

            seen_components.add(component_id)
            suggestions.append({
                "source_project": project,
                "component": result.content[:200],
                "relevance": result.score,
                "metadata": result.metadata
            })

        return suggestions[:10]


class ConflictResolver:
    """
    Resolves conflicts when multiple agents suggest contradictory approaches.

    Uses voting, priority, and context analysis to determine best resolution.
    """

    def __init__(self):
        self.resolutions: List[ConflictResolution] = []
        self.resolution_strategies = {
            "voting": self._resolve_by_voting,
            "priority": self._resolve_by_priority,
            "context": self._resolve_by_context,
            "hybrid": self._resolve_hybrid
        }

        logger.info("ConflictResolver initialized")

    def detect_conflict(
        self,
        suggestions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect if there's a conflict between suggestions"""
        if len(suggestions) < 2:
            return None

        # Check for contradictions
        actions = set()
        targets = set()

        for suggestion in suggestions:
            action = suggestion.get("action", "").lower()
            target = suggestion.get("target", "")

            actions.add(action)
            targets.add(target)

        # Contradictory actions on the same target
        if len(targets) == 1 and len(actions) > 1:
            contradictory_pairs = [
                ("add", "remove"),
                ("create", "delete"),
                ("enable", "disable"),
                ("increase", "decrease")
            ]

            for a1, a2 in contradictory_pairs:
                if a1 in actions and a2 in actions:
                    return {
                        "type": "contradictory_actions",
                        "target": list(targets)[0],
                        "conflicting_actions": [a1, a2],
                        "suggestions": suggestions
                    }

        # Different approaches to the same problem
        approaches = [s.get("approach", "") for s in suggestions]
        if len(set(approaches)) > 1 and "" not in approaches:
            return {
                "type": "different_approaches",
                "approaches": list(set(approaches)),
                "suggestions": suggestions
            }

        return None

    def resolve(
        self,
        conflict: Dict[str, Any],
        strategy: str = "hybrid",
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictResolution:
        """Resolve a conflict using the specified strategy"""
        resolver = self.resolution_strategies.get(strategy, self._resolve_hybrid)
        resolution = resolver(conflict, context or {})

        self.resolutions.append(resolution)
        return resolution

    def _resolve_by_voting(
        self,
        conflict: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """Resolve by counting votes/support for each option"""
        suggestions = conflict.get("suggestions", [])

        # Count support
        votes = defaultdict(int)
        for suggestion in suggestions:
            approach = suggestion.get("approach", suggestion.get("action", "unknown"))
            votes[approach] += suggestion.get("confidence", 1.0)

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])

        winning_suggestion = next(
            (s for s in suggestions if s.get("approach", s.get("action")) == winner[0]),
            suggestions[0]
        )

        return ConflictResolution(
            id=f"resolution_{uuid.uuid4().hex[:12]}",
            conflict_type=conflict.get("type", "unknown"),
            options=[{"approach": k, "votes": v} for k, v in votes.items()],
            resolution=winning_suggestion,
            reasoning=f"Selected by voting: {winner[0]} received highest support ({winner[1]:.2f})",
            confidence=winner[1] / sum(votes.values()),
            timestamp=datetime.now()
        )

    def _resolve_by_priority(
        self,
        conflict: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """Resolve based on agent priorities"""
        suggestions = conflict.get("suggestions", [])

        # Define agent priority order
        priority_order = context.get("priority_order", [
            "security",
            "reliability",
            "performance",
            "maintainability",
            "cost"
        ])

        # Score suggestions by priority alignment
        scored = []
        for suggestion in suggestions:
            score = 0
            tags = suggestion.get("tags", [])
            for i, priority in enumerate(priority_order):
                if priority in tags:
                    score += len(priority_order) - i

            scored.append((suggestion, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        return ConflictResolution(
            id=f"resolution_{uuid.uuid4().hex[:12]}",
            conflict_type=conflict.get("type", "unknown"),
            options=[{"suggestion": s, "priority_score": sc} for s, sc in scored],
            resolution=winner,
            reasoning=f"Selected by priority alignment with organizational goals",
            confidence=0.7 if scored[0][1] > 0 else 0.5,
            timestamp=datetime.now()
        )

    def _resolve_by_context(
        self,
        conflict: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """Resolve based on current context"""
        suggestions = conflict.get("suggestions", [])

        # Extract context features
        project_phase = context.get("phase", "development")
        constraints = context.get("constraints", [])
        recent_issues = context.get("recent_issues", [])

        # Score based on context fit
        scored = []
        for suggestion in suggestions:
            score = 0.5  # Base score

            # Phase alignment
            suitable_phases = suggestion.get("suitable_phases", [])
            if project_phase in suitable_phases:
                score += 0.2

            # Constraint satisfaction
            meets_constraints = suggestion.get("meets_constraints", [])
            for constraint in constraints:
                if constraint in meets_constraints:
                    score += 0.1

            # Addresses recent issues
            addresses = suggestion.get("addresses_issues", [])
            for issue in recent_issues:
                if issue in addresses:
                    score += 0.1

            scored.append((suggestion, min(score, 1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        return ConflictResolution(
            id=f"resolution_{uuid.uuid4().hex[:12]}",
            conflict_type=conflict.get("type", "unknown"),
            options=[{"suggestion": s, "context_score": sc} for s, sc in scored],
            resolution=winner,
            reasoning=f"Selected based on best fit with current project context",
            confidence=scored[0][1],
            timestamp=datetime.now()
        )

    def _resolve_hybrid(
        self,
        conflict: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """Resolve using a combination of strategies"""
        voting_result = self._resolve_by_voting(conflict, context)
        priority_result = self._resolve_by_priority(conflict, context)
        context_result = self._resolve_by_context(conflict, context)

        # Combine scores
        suggestion_scores = defaultdict(float)

        for resolution, weight in [
            (voting_result, 0.3),
            (priority_result, 0.3),
            (context_result, 0.4)
        ]:
            res_key = json.dumps(resolution.resolution, sort_keys=True)
            suggestion_scores[res_key] += resolution.confidence * weight

        # Find winner
        winner_key = max(suggestion_scores.items(), key=lambda x: x[1])
        winner = json.loads(winner_key[0])

        return ConflictResolution(
            id=f"resolution_{uuid.uuid4().hex[:12]}",
            conflict_type=conflict.get("type", "unknown"),
            options=[
                {"strategy": "voting", "result": voting_result.resolution},
                {"strategy": "priority", "result": priority_result.resolution},
                {"strategy": "context", "result": context_result.resolution}
            ],
            resolution=winner,
            reasoning="Selected using hybrid approach combining voting, priority, and context analysis",
            confidence=winner_key[1],
            timestamp=datetime.now()
        )

    def get_resolution_history(
        self,
        conflict_type: Optional[str] = None,
        limit: int = 20
    ) -> List[ConflictResolution]:
        """Get history of conflict resolutions"""
        resolutions = self.resolutions

        if conflict_type:
            resolutions = [r for r in resolutions if r.conflict_type == conflict_type]

        return resolutions[-limit:]
