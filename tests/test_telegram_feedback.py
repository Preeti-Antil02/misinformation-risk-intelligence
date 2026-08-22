import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.telegram_bot import format_telegram_report
from risklens.feedback import record_prediction, record_feedback


def test_telegram_report_label():
    report = format_telegram_report({
        "claim": "Test claim about vaccines",
        "verdict": "This claim has been debunked by authorities.",
        "risk_score": 0.88,
        "risk_level": "High"
    })
    
    assert "Misinformation probability: 88%" in report["raw_text"], "Report should say 'Misinformation probability'"
    assert "Confidence:" not in report["raw_text"], "Report must NOT say 'Confidence:'"
    assert report["prob_pct"] == 88


def test_telegram_feedback_cycle():
    # 1. Record prediction
    pid = record_prediction(
        text="Test claim about vaccines",
        language="en",
        probability=0.88,
        risk_level="High",
        model_used="Ensemble v2.1",
        source="telegram",
        user_id="test_user_456"
    )
    assert pid > 0, f"Valid prediction id expected, got {pid}"

    # 2. Record positive feedback
    res = record_feedback(
        prediction_id=pid,
        user_feedback="✅ Correct",
        correct_label="real",
        user_id="test_user_456"
    )
    assert res.get("success") is True, f"Feedback should succeed, got {res}"


if __name__ == "__main__":
    test_telegram_report_label()
    test_telegram_feedback_cycle()
    print("ALL TESTS PASSED SUCCESSFULLY!")
