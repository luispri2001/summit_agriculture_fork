from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL

from furrow_following.state_machines.odom_check_end_furrow_state_machine import (
    OdomCheckEndFurrowStateMachine,
)
from furrow_following.state_machines.furrow_following_state_machine import (
    FurrowFollowingStateMachine,
)
from furrow_following.state_machines.move_to_next_furrow_state_machine import (
    MoveToNextFurrowStateMachine,
)
from furrow_following.states.outcomes import ENDS, CONTINUES


class MainStateMachine(StateMachine):

    def __init__(self) -> None:
        super().__init__([SUCCEED, ABORT, CANCEL])

        self.add_state(
            "FURROW_FOLLOWING",
            FurrowFollowingStateMachine(),
            {
                SUCCEED: "FURROW_FOLLOWING",
                ABORT: ABORT,
                CANCEL: CANCEL,
                ENDS: "MOVING_TO_NEXT_FURROW",
            },
        )

        self.add_state(
            "MOVING_TO_NEXT_FURROW",
            MoveToNextFurrowStateMachine(),
            {
                SUCCEED: "FURROW_FOLLOWING",
                CANCEL: CANCEL,
            },
        )
