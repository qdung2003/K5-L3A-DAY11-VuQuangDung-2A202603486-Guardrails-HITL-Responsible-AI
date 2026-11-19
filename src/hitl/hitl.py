"""
Lab 11 — Optional enrichment: Human-in-the-Loop Design
  (Không chấm — tham khảo. Tóm tắt nộp do scripts/grade.py tự sinh,
   không viết report/*.md tay.)
  - Confidence Router
  - 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# Optional enrichment: ConfidenceRouter (không chấm)
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # Confidence ngoài [0, 1] hoặc không phải số → coi như thấp nhất (fail-safe)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        if not (0.0 <= conf <= 1.0):  # NaN hoặc ngoài khoảng → không tin được
            conf = 0.0

        # 1. Hành động rủi ro cao → luôn cần người duyệt, bất kể confidence
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=conf,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        # 2. Theo ngưỡng confidence
        if conf >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=conf,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )
        if conf >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=conf,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )
        return RoutingDecision(
            action="escalate",
            confidence=conf,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# Optional enrichment: 3 HITL decision points (không chấm)
# Không bắt buộc điền. Tóm tắt bài nộp: chạy scripts/grade.py
# (tự sinh lab_report.md) — không viết report tay.
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "High-value / new-beneficiary money transfer",
        "trigger": (
            "Agent proposes transfer_money where amount ≥ 50,000,000 VND, OR the "
            "beneficiary was added < 24h ago, OR the request originated from "
            "untrusted content (email/RAG) rather than the authenticated user."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Authenticated customer ID, amount, source/destination accounts, "
            "beneficiary age, the user's original message, any external document "
            "that influenced the request, device/IP risk score, last 5 transactions."
        ),
        "example": (
            "Customer asks to 'pay the invoice in this email': 120,000,000 VND to an "
            "account first seen today. Agent drafts the transfer; it is held until a "
            "payments officer verifies with the customer by phone."
        ),
        "approval_path": (
            "Approve → execute once with approval_id HITL-XXXXXXXX bound to the exact "
            "amount/destination. Reject → cancel and notify customer. Timeout (15 min) "
            "→ auto-reject (fail-closed), never auto-approve."
        ),
        "audit_fields": (
            "correlation_id, customer_id, intent=transfer_money, proposed action JSON, "
            "diff vs. customer's stated intent, reviewer_id, decision, decided_at, "
            "approval_id"
        ),
    },
    {
        "id": 2,
        "name": "Account profile / credential change",
        "trigger": (
            "Agent proposes change_password, update_personal_info (phone, email, "
            "address) or close_account — actions that enable account takeover."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Fields before/after (masked), authentication strength of the session "
            "(OTP/biometric), recent failed logins, whether the new phone/email is "
            "already linked to another customer."
        ),
        "example": (
            "Chat session asks to change the registered phone number and then reset "
            "the password. The change is queued; a reviewer checks KYC and sends an "
            "OTP to the OLD phone before approving."
        ),
        "approval_path": (
            "Approve → apply change, send confirmation to old and new contact. "
            "Reject → keep current data, flag session for fraud review. Timeout "
            "(30 min) → reject and ask the customer to visit a branch."
        ),
        "audit_fields": (
            "correlation_id, customer_id, intent, field-level diff (masked), "
            "auth method, reviewer_id, decision, reason_code, decided_at"
        ),
    },
    {
        "id": 3,
        "name": "Low-confidence or guardrail-flagged answer",
        "trigger": (
            "ConfidenceRouter returns queue_review/escalate (confidence < 0.9), the "
            "output guardrail redacted PII, or monitoring raises a block_rate alert "
            "for the same session."
        ),
        "hitl_model": "human-on-the-loop",
        "context_needed": (
            "User question, draft answer, confidence score, guardrail issues found, "
            "retrieved policy documents used as grounding, session history."
        ),
        "example": (
            "Customer asks about early-withdrawal penalties on a 12-month deposit; "
            "the model answers with confidence 0.74. The answer is queued; an agent "
            "supervisor corrects the penalty rate before it is sent."
        ),
        "approval_path": (
            "Approve → send draft as-is. Edit → send corrected answer and store the "
            "edit as feedback. Reject → send a safe fallback ('a specialist will "
            "contact you'). Timeout (5 min) → safe fallback, ticket stays open."
        ),
        "audit_fields": (
            "correlation_id, session_id, intent, confidence, guardrail issues, "
            "draft vs. final diff, reviewer_id, decision, latency_to_decision"
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
