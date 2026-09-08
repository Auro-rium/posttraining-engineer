"""Approval, rejection, and rollback state machine for candidate cycles."""

from __future__ import annotations

from .models import CycleEvent, CycleState, PostTrainingCycle


class InvalidTransition(ValueError):
    """Raised when a cycle transition is not legal for its current state."""


_ALLOWED: dict[CycleState, frozenset[CycleState]] = {
    CycleState.CREATED: frozenset({CycleState.BENCHMARKED}),
    CycleState.BENCHMARKED: frozenset({CycleState.EVALUATED}),
    CycleState.EVALUATED: frozenset({CycleState.APPROVED, CycleState.REJECTED}),
    CycleState.APPROVED: frozenset({CycleState.ROLLED_BACK}),
    CycleState.REJECTED: frozenset(),
    CycleState.ROLLED_BACK: frozenset(),
}


class CycleStateMachine:
    """Small explicit state machine with append-only transition history."""

    def __init__(
        self,
        cycle: PostTrainingCycle | None = None,
        *,
        cycle_id: str | None = None,
        **cycle_kwargs: object,
    ) -> None:
        if cycle is None:
            if cycle_id is None:
                raise ValueError("cycle or cycle_id is required")
            cycle = PostTrainingCycle(cycle_id=cycle_id, **cycle_kwargs)
        elif cycle_id is not None or cycle_kwargs:
            raise ValueError("cycle_id and cycle_kwargs cannot be combined with cycle")
        self.cycle = cycle

    @classmethod
    def create(cls, cycle_id: str, **kwargs: object) -> CycleStateMachine:
        return cls(PostTrainingCycle(cycle_id=cycle_id, **kwargs))

    @property
    def state(self) -> CycleState:
        return self.cycle.state

    @property
    def current_state(self) -> CycleState:
        """Compatibility spelling for callers that model state explicitly."""

        return self.state

    @property
    def history(self) -> tuple[CycleEvent, ...]:
        return self.cycle.events

    def transition(self, to_state: CycleState, reason: str) -> PostTrainingCycle:
        if not reason.strip():
            raise ValueError("transition reason cannot be blank")
        allowed = _ALLOWED[self.state]
        if to_state not in allowed:
            raise InvalidTransition(
                f"cannot transition from {self.state.value} to {to_state.value}"
            )
        event = CycleEvent(
            event_id=f"{self.cycle.cycle_id}:transition:{len(self.cycle.events) + 1}",
            from_state=self.state,
            to_state=to_state,
            reason=reason,
        )
        self.cycle = self.cycle.model_copy(
            update={"state": to_state, "events": (*self.cycle.events, event)}
        )
        return self.cycle

    def mark_benchmarked(self, reason: str = "benchmark completed") -> PostTrainingCycle:
        return self.transition(CycleState.BENCHMARKED, reason)

    def mark_evaluated(self, reason: str = "evaluation completed") -> PostTrainingCycle:
        return self.transition(CycleState.EVALUATED, reason)

    def approve(self, reason: str = "promotion gate passed") -> PostTrainingCycle:
        return self.transition(CycleState.APPROVED, reason)

    def reject(self, reason: str = "promotion gate failed") -> PostTrainingCycle:
        return self.transition(CycleState.REJECTED, reason)

    def rollback(self, reason: str = "approved candidate rolled back") -> PostTrainingCycle:
        return self.transition(CycleState.ROLLED_BACK, reason)

    approve_candidate = approve
    reject_candidate = reject
    rollback_candidate = rollback


ApprovalStateMachine = CycleStateMachine
PostTrainingStateMachine = CycleStateMachine
StateTransitionError = InvalidTransition
