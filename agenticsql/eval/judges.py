"""
LLM-as-a-Judge Rubrics and Evaluators for Text-to-SQL Agent Responses.

Provides structured evaluators for:
1. Table-Grounded Faithfulness (hallucination detection against executed tabular results)
2. SQL Semantic Parity (logic equivalence when AST syntaxes diverge)
3. Ambiguity & Clarification Handling (evaluation of underspecified intent resolution)
"""

import json
import logging
import re
from typing import Optional, Any
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


# --- Structured Output Models ---

class FaithfulnessVerdict(BaseModel):
    """Structured verdict for faithfulness / hallucination judge."""
    score: float = Field(..., description="Faithfulness score from 0.0 (unfaithful/hallucinated) to 1.0 (perfectly grounded)")
    is_faithful: bool = Field(..., description="True if no hallucinations or unsupported claims are detected")
    hallucinations_detected: list[str] = Field(default_factory=list, description="Specific numbers or claims not present in data")
    unsupported_claims: list[str] = Field(default_factory=list, description="General unsupported assertions")
    reasoning: str = Field(..., description="Step-by-step evaluation reasoning based on rubric")


class SemanticParityVerdict(BaseModel):
    """Structured verdict for SQL semantic parity judge."""
    score: float = Field(..., description="Semantic parity score from 0.0 to 1.0")
    is_semantically_equivalent: bool = Field(..., description="True if generated SQL achieves identical intent as gold SQL")
    intent_coverage: float = Field(..., description="Proportion of user requirements addressed (0.0 to 1.0)")
    semantic_flaws: list[str] = Field(default_factory=list, description="Logical discrepancies or missing filters")
    reasoning: str = Field(..., description="Step-by-step reasoning")


class AmbiguityVerdict(BaseModel):
    """Structured verdict for ambiguity clarification judge."""
    score: float = Field(..., description="Ambiguity handling score from 0.0 to 1.0")
    handled_appropriately: bool = Field(..., description="True if agent properly addressed ambiguity")
    clarification_requested: bool = Field(default=False, description="True if agent asked user for clarification")
    assumptions_stated: bool = Field(default=False, description="True if agent explicitly stated assumptions made")
    missing_clarifications: list[str] = Field(default_factory=list, description="Unaddressed ambiguous aspects")
    reasoning: str = Field(..., description="Evaluation reasoning")


# --- Prompts & Rubrics ---

FAITHFULNESS_RUBRIC_PROMPT = """You are an expert AI Evaluation Judge specializing in verifying the factual grounding of AI responses against tabular database query results.

EVALUATION TASK:
Carefully compare the AGENT'S NATURAL LANGUAGE RESPONSE against the EXECUTED TABULAR DATA (from the SQL query) and the USER'S QUESTION.
Determine if every factual claim, number, name, statistic, ranking, and conclusion made by the agent is STRICTLY GROUNDED in the provided table data.

SCORING RUBRIC:
- 1.0 (Perfect Grounding): Every number, entity name, and relationship in the response is directly present in or accurately computed from the execution table. No hallucinations.
- 0.7 - 0.9 (Minor Grounding Issues): Main conclusions and numbers are accurate, but minor stylistic or non-critical extrapolations exist without contradicting the table.
- 0.4 - 0.6 (Partial Hallucination): Contains some correct data points, but fabricates or misattributes specific numbers, row counts, or entity names not found in the table.
- 0.0 - 0.3 (Severe Hallucination / Contradiction): Fabricates key data points, claims results that contradict the table data, or describes rows that do not exist.

INPUT DATA:
User Question: {user_question}
Executed SQL Query: {executed_sql}

Executed Tabular Result:
{table_markdown}

Agent Response to Evaluate:
{agent_response}

You must respond ONLY with a JSON object matching this schema:
{{
  "score": <float between 0.0 and 1.0>,
  "is_faithful": <boolean, true if score >= 0.85>,
  "hallucinations_detected": [<list of specific fabricated numbers, names, or values>],
  "unsupported_claims": [<list of claims unsupported by the table>],
  "reasoning": "<step-by-step explanation referencing specific cells and text sentences>"
}}
"""

SEMANTIC_PARITY_RUBRIC_PROMPT = """You are an expert Database and SQL Architect serving as an AI Evaluation Judge.

EVALUATION TASK:
Evaluate whether the CANDIDATE SQL QUERY is semantically equivalent in intent and logic to the GOLD REFERENCE SQL QUERY for the given user intent and database schema.

NOTE: The queries do NOT need to have identical syntax. For example, using a CTE vs a subquery, or `IN (SELECT...)` vs `EXISTS` or `INNER JOIN`, is fully acceptable if they produce the same logical result set for the domain schema.

SCORING RUBRIC:
- 1.0 (Semantically Equivalent): Captures the complete logic, correct joins, where predicates, groupings, aggregations, and ordering required by the user intent.
- 0.7 - 0.9 (Minor Logic Deviation): Captures core logic but has slight differences (e.g., missing secondary ORDER BY, subtle null-handling difference).
- 0.4 - 0.6 (Partial Logic Match): Uses correct tables but wrong aggregation functions, incorrect join types (e.g. INNER instead of LEFT JOIN where NULLs exist), or missing key WHERE filters.
- 0.0 - 0.3 (Incorrect / Irrelevant Logic): Wrong target tables, totally missing primary filtering conditions, or returning unrelated data.

INPUT:
User Intent: {user_intent}
Database Schema Context: {schema_context}
Gold Reference SQL: {gold_sql}
Candidate Generated SQL: {candidate_sql}

You must respond ONLY with a JSON object matching this schema:
{{
  "score": <float between 0.0 and 1.0>,
  "is_semantically_equivalent": <boolean, true if score >= 0.85>,
  "intent_coverage": <float between 0.0 and 1.0>,
  "semantic_flaws": [<list of logical discrepancies or missing conditions>],
  "reasoning": "<concise structural comparison explaining verdict>"
}}
"""

AMBIGUITY_RUBRIC_PROMPT = """You are an expert AI Evaluation Judge assessing how an AI Assistant handles ambiguous or underspecified user queries.

EVALUATION TASK:
Determine if the Agent recognized ambiguity in the User Intent and responded appropriately by either:
1. Asking targeted clarification questions before or while answering.
2. Explicitly stating the reasonable assumptions made (e.g., "Assuming 'top sales' means total revenue in USD...").

SCORING RUBRIC:
- 1.0 (Excellent): Recognized ambiguity and explicitly clarified or stated precise domain assumptions.
- 0.6 - 0.8 (Acceptable): Picked a reasonable default interpretation and hinted at assumptions, but did not clearly disclose alternative interpretations.
- 0.0 - 0.4 (Poor): Blindly executed an arbitrary query on an ambiguous prompt without acknowledging ambiguity or stating assumptions.

INPUT:
User Intent: {user_intent}
Ambiguity Details: {ambiguity_details}
Agent Response: {agent_response}

You must respond ONLY with a JSON object matching this schema:
{{
  "score": <float between 0.0 and 1.0>,
  "handled_appropriately": <boolean, true if score >= 0.7>,
  "clarification_requested": <boolean>,
  "assumptions_stated": <boolean>,
  "missing_clarifications": [<list of unaddressed ambiguity points>],
  "reasoning": "<explanation of ambiguity handling>"
}}
"""


class LLMJudgeEvaluator:
    """
    Orchestrates LLM-as-a-Judge evaluations with real LLM invocation or mock fallbacks.
    """

    def __init__(self, llm: Optional[Any] = None, use_mock: bool = False):
        """
        Args:
            llm: Optional LangChain chat model instance (e.g. ChatGoogleGenerativeAI).
            use_mock: If True, uses deterministic rule-based heuristic mock judge.
        """
        self.llm = llm
        self.use_mock = use_mock or (llm is None)

    def evaluate_faithfulness(
        self,
        user_question: str,
        executed_sql: str,
        table_data: Optional[pd.DataFrame | list[dict] | str],
        agent_response: str,
    ) -> FaithfulnessVerdict:
        """Evaluate if agent's NL response is faithfully grounded in the tabular data."""
        table_str = self._format_table_markdown(table_data)

        if self.use_mock or self.llm is None:
            return self._mock_faithfulness_eval(user_question, table_str, agent_response)

        prompt = FAITHFULNESS_RUBRIC_PROMPT.format(
            user_question=user_question,
            executed_sql=executed_sql,
            table_markdown=table_str,
            agent_response=agent_response,
        )

        return self._invoke_and_parse(prompt, FaithfulnessVerdict)

    def evaluate_semantic_parity(
        self,
        user_intent: str,
        gold_sql: str,
        candidate_sql: str,
        schema_context: str = "",
    ) -> SemanticParityVerdict:
        """Evaluate if candidate SQL is semantically equivalent to gold SQL."""
        if self.use_mock or self.llm is None:
            return self._mock_semantic_parity_eval(user_intent, gold_sql, candidate_sql)

        prompt = SEMANTIC_PARITY_RUBRIC_PROMPT.format(
            user_intent=user_intent,
            schema_context=schema_context,
            gold_sql=gold_sql,
            candidate_sql=candidate_sql,
        )

        return self._invoke_and_parse(prompt, SemanticParityVerdict)

    def evaluate_ambiguity(
        self,
        user_intent: str,
        agent_response: str,
        ambiguity_details: str = "",
    ) -> AmbiguityVerdict:
        """Evaluate whether agent handled an ambiguous intent appropriately."""
        if self.use_mock or self.llm is None:
            return self._mock_ambiguity_eval(user_intent, agent_response, ambiguity_details)

        prompt = AMBIGUITY_RUBRIC_PROMPT.format(
            user_intent=user_intent,
            ambiguity_details=ambiguity_details,
            agent_response=agent_response,
        )

        return self._invoke_and_parse(prompt, AmbiguityVerdict)

    def _invoke_and_parse(self, prompt: str, schema_cls: type[BaseModel]) -> Any:
        """Invoke LLM and parse JSON response into Pydantic model."""
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            json_str = self._extract_json(content)
            data = json.loads(json_str)
            return schema_cls.model_validate(data)
        except Exception as e:
            logger.warning("LLM Judge invocation failed (%s), using fallback heuristic.", e)
            return self._fallback_verdict(schema_cls, str(e))

    def _format_table_markdown(self, table_data: Any) -> str:
        """Convert DataFrame or dict list to clean markdown table string."""
        if table_data is None:
            return "*(No rows returned / Empty Table)*"
        if isinstance(table_data, str):
            return table_data
        if isinstance(table_data, pd.DataFrame):
            if table_data.empty:
                return "*(Empty DataFrame: 0 rows)*"
            return table_data.to_markdown(index=False)
        if isinstance(table_data, list):
            if not table_data:
                return "*(Empty list of records)*"
            df = pd.DataFrame(table_data)
            return df.to_markdown(index=False)
        return str(table_data)

    def _extract_json(self, text: str) -> str:
        """Extract JSON substring from LLM response text."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match_brace = re.search(r"\{.*\}", text, re.DOTALL)
        if match_brace:
            return match_brace.group(0)
        return text.strip()

    # --- Deterministic Mock Judges for Testing & Offline Execution ---

    def _mock_faithfulness_eval(
        self, user_question: str, table_markdown: str, agent_response: str
    ) -> FaithfulnessVerdict:
        """Deterministic heuristic judge for testing."""
        # Check if response explicitly mentions BLOCKED or errors
        if "[BLOCKED]" in agent_response or "read-only" in agent_response.lower():
            return FaithfulnessVerdict(
                score=1.0,
                is_faithful=True,
                hallucinations_detected=[],
                unsupported_claims=[],
                reasoning="Guardrail block message correctly generated without hallucination.",
            )

        # Extract numbers and compare numerically to handle float formatting differences (e.g. 1944.50 vs 1944.5)
        def extract_floats(text: str) -> set[float]:
            floats = set()
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", text):
                try:
                    floats.add(round(float(m), 2))
                except ValueError:
                    pass
            return floats

        resp_floats = extract_floats(agent_response)
        table_floats = extract_floats(table_markdown)

        # Filter out small numbers (e.g., 1, 2, 3) which are often item bullets
        significant_resp_floats = {f for f in resp_floats if f > 5.0}
        unsupported = [str(f) for f in significant_resp_floats if f not in table_floats]

        if not unsupported:
            return FaithfulnessVerdict(
                score=1.0,
                is_faithful=True,
                hallucinations_detected=[],
                unsupported_claims=[],
                reasoning="All significant numbers cited in response are grounded in tabular data.",
            )
        else:
            return FaithfulnessVerdict(
                score=0.4,
                is_faithful=False,
                hallucinations_detected=unsupported,
                unsupported_claims=[f"Number {n} not present in table" for n in unsupported],
                reasoning=f"Detected {len(unsupported)} numbers in response not present in table.",
            )

    def _mock_semantic_parity_eval(
        self, user_intent: str, gold_sql: str, candidate_sql: str
    ) -> SemanticParityVerdict:
        """Deterministic heuristic judge for SQL semantic parity."""
        cand_clean = " ".join(candidate_sql.lower().split())
        gold_clean = " ".join(gold_sql.lower().split())

        if cand_clean == gold_clean:
            return SemanticParityVerdict(
                score=1.0,
                is_semantically_equivalent=True,
                intent_coverage=1.0,
                semantic_flaws=[],
                reasoning="Candidate SQL query matches gold query exactly.",
            )

        # Check keyword overlaps
        gold_words = set(re.findall(r"\w+", gold_clean))
        cand_words = set(re.findall(r"\w+", cand_clean))
        intersection = gold_words.intersection(cand_words)
        jaccard = len(intersection) / len(gold_words) if gold_words else 1.0

        is_equiv = jaccard >= 0.7
        return SemanticParityVerdict(
            score=round(jaccard, 2),
            is_semantically_equivalent=is_equiv,
            intent_coverage=round(jaccard, 2),
            semantic_flaws=[] if is_equiv else ["Significant keyword difference with reference SQL"],
            reasoning=f"Heuristic word overlap similarity score: {jaccard:.2f}",
        )

    def _mock_ambiguity_eval(
        self, user_intent: str, agent_response: str, ambiguity_details: str
    ) -> AmbiguityVerdict:
        """Deterministic heuristic judge for ambiguity clarification."""
        resp_lower = agent_response.lower()
        clarification_keywords = ["clarify", "assuming", "assumption", "could mean", "defined as", "options"]
        found = [kw for kw in clarification_keywords if kw in resp_lower]

        handled = len(found) > 0
        return AmbiguityVerdict(
            score=1.0 if handled else 0.5,
            handled_appropriately=handled,
            clarification_requested="clarify" in resp_lower or "?" in agent_response,
            assumptions_stated="assum" in resp_lower or "defined as" in resp_lower,
            missing_clarifications=[] if handled else ["Did not explicitly state interpretation assumptions"],
            reasoning=f"Ambiguity markers found in response: {found}" if handled else "No explicit ambiguity clarification found.",
        )

    def _fallback_verdict(self, schema_cls: type[BaseModel], err_msg: str) -> Any:
        """Create a safe fallback verdict instance."""
        if schema_cls == FaithfulnessVerdict:
            return FaithfulnessVerdict(
                score=0.5,
                is_faithful=False,
                hallucinations_detected=[],
                unsupported_claims=[],
                reasoning=f"Fallback verdict due to parsing error: {err_msg}",
            )
        elif schema_cls == SemanticParityVerdict:
            return SemanticParityVerdict(
                score=0.5,
                is_semantically_equivalent=False,
                intent_coverage=0.5,
                semantic_flaws=[err_msg],
                reasoning=f"Fallback verdict: {err_msg}",
            )
        else:
            return AmbiguityVerdict(
                score=0.5,
                handled_appropriately=False,
                missing_clarifications=[err_msg],
                reasoning=f"Fallback verdict: {err_msg}",
            )
