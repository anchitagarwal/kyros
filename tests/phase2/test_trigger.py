"""test_trigger.py — hard gates + soft triggers."""

from trading.trigger import TriggerEngine, TriggerResult
from trading.cooldown import CooldownState
from trading.alert import AlertPayload


def _snap(**kw):
    """Build a snapshot stub with overridable fields."""
    class _S:
        pass
    s = _S()
    s.current_killzone = kw.get("current_killzone", "ny_am_kz")
    s.htf_bias = kw.get("htf_bias", "bullish")
    s.nearest_dol = kw.get("nearest_dol", object())  # truthy by default
    s.fvgs = kw.get("fvgs", {"5m": [], "15m": []})
    s.ifvgs = kw.get("ifvgs", {"5m": [], "15m": []})
    s.recent_sweeps = kw.get("recent_sweeps", {"5m": [], "15m": []})
    s.displacements = kw.get("displacements", {"5m": [], "1m": []})
    s.timestamp = kw.get("timestamp", None)
    return s


def _engine(cooling=False):
    cd = CooldownState()
    if cooling:
        cd.last_alert_bias = "no_trade"
        # Make is_cooling_down return True via a stub.
        cd.is_cooling_down = lambda snap: True
    return TriggerEngine(cd)


# ── Hard gates (each independently blocks) ────────────────────────────────────


def test_gate_a_no_killzone_blocks():
    eng = _engine()
    snap = _snap(current_killzone=None, htf_bias="bullish", nearest_dol=object())
    r = eng.evaluate(snap)
    assert r.should_trigger is False
    assert r.reason == "no_killzone"


def test_gate_b_no_htf_bias_blocks():
    eng = _engine()
    snap = _snap(current_killzone="ny_am_kz", htf_bias=None, nearest_dol=object())
    r = eng.evaluate(snap)
    assert r.should_trigger is False
    assert r.reason == "no_htf_bias"


def test_gate_c_no_dol_blocks():
    eng = _engine()
    snap = _snap(current_killzone="ny_am_kz", htf_bias="bullish", nearest_dol=None)
    r = eng.evaluate(snap)
    assert r.should_trigger is False
    assert r.reason == "no_dol"


def test_gate_d_cooldown_blocks():
    eng = _engine(cooling=True)
    snap = _snap(current_killzone="ny_am_kz", htf_bias="bullish", nearest_dol=object())
    r = eng.evaluate(snap)
    assert r.should_trigger is False
    assert r.reason == "cooldown_active"


def test_gate_ordering_first_failure_wins():
    """When killzone is None AND htf_bias is None, reason == 'no_killzone'."""
    eng = _engine()
    snap = _snap(current_killzone=None, htf_bias=None, nearest_dol=None)
    r = eng.evaluate(snap)
    assert r.reason == "no_killzone"


# ── Soft triggers (each independently fires) ──────────────────────────────────


def test_soft_fvg_fires():
    eng = _engine()
    snap = _snap(fvgs={"5m": [{"type": "fvg_bullish"}], "15m": []})
    r = eng.evaluate(snap)
    assert r.should_trigger is True
    assert r.reason == "fvg"


def test_soft_fvg_15m_fires():
    eng = _engine()
    snap = _snap(fvgs={"5m": [], "15m": [{"type": "fvg_bullish"}]})
    r = eng.evaluate(snap)
    assert r.should_trigger is True
    assert r.reason == "fvg"


def test_soft_ifvg_fires():
    eng = _engine()
    snap = _snap(ifvgs={"5m": [{"type": "ifvg_bullish"}], "15m": []})
    r = eng.evaluate(snap)
    assert r.should_trigger is True
    assert r.reason == "ifvg"


def test_soft_sweep_fires():
    eng = _engine()
    snap = _snap(recent_sweeps={"5m": [{"type": "sweep_ssl"}], "15m": []})
    r = eng.evaluate(snap)
    assert r.should_trigger is True
    assert r.reason == "sweep"


def test_soft_displacement_fires():
    eng = _engine()
    snap = _snap(displacements={"5m": [{"type": "displacement_bullish"}], "1m": []})
    r = eng.evaluate(snap)
    assert r.should_trigger is True
    assert r.reason == "displacement"


def test_no_soft_trigger_when_all_false():
    eng = _engine()
    snap = _snap(
        fvgs={"5m": [], "15m": []},
        ifvgs={"5m": [], "15m": []},
        recent_sweeps={"5m": [], "15m": []},
        displacements={"5m": [], "1m": []},
    )
    r = eng.evaluate(snap)
    assert r.should_trigger is False
    assert r.reason == "no_soft_trigger"


def test_trigger_result_always_has_reason():
    eng = _engine()
    for snap in [
        _snap(current_killzone=None),
        _snap(htf_bias=None),
        _snap(nearest_dol=None),
        _snap(fvgs={"5m": [{"x": 1}], "15m": []}),
        _snap(),
    ]:
        r = eng.evaluate(snap)
        assert isinstance(r, TriggerResult)
        assert r.reason  # non-empty
