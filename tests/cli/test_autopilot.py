
from src.cli.autopilot import build_parser
from src.ml.autopilot import AutopilotAction, AutopilotRun


def make_runs(*actions: AutopilotAction) -> list[AutopilotRun]:
    return [
        AutopilotRun(taxi_type=f"fleet{index}", action=action, reason="motivo registrado")
        for index, action in enumerate(actions)
    ]


class TestParser:
    def test_month_is_optional(self):
        assert build_parser().parse_args([]).month is None
