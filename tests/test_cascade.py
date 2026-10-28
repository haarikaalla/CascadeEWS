import json

import pytest

from cascadeews.cascade import (
    ELEMENTS,
    STAGE_ORDER,
    scenario_summary,
    simulate_cascade,
    write_scenario,
)


def test_scenario_has_all_elements_weeks_and_stages():
    frame = simulate_cascade(0.60, weeks=12)
    assert len(frame) == len(ELEMENTS) * 13
    assert set(frame["element"]) == set(ELEMENTS)
    assert set(frame["stage"]).issubset(STAGE_ORDER)
    assert frame["risk"].between(0, 1).all()
    assert frame["week"].min() == 0
    assert frame["week"].max() == 12


def test_more_pressure_never_reduces_final_scenario_risk():
    low = simulate_cascade(0.30, weeks=20).query("week == 20").set_index("element")
    high = simulate_cascade(0.70, weeks=20).query("week == 20").set_index("element")
    assert (high["risk"] >= low["risk"]).all()


def test_summary_and_files_are_machine_readable(tmp_path):
    result = write_scenario(tmp_path, global_pressure=0.60, weeks=10)
    assert result["scenario_type"].startswith("illustrative")
    assert len(result["elements"]) == len(ELEMENTS)
    assert (tmp_path / "cascade_stages.csv").exists()
    assert (tmp_path / "cascade_scenario.png").exists()
    persisted = json.loads((tmp_path / "cascade_summary.json").read_text())
    assert persisted == result


@pytest.mark.parametrize("pressure", [-0.1, 1.1])
def test_invalid_pressure_is_rejected(pressure):
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate_cascade(pressure)


def test_summary_records_first_threshold_crossings():
    frame = simulate_cascade(0.90, weeks=52)
    result = scenario_summary(frame, 0.90)
    assert any(item["first_warning_week"] is not None for item in result["elements"])
