from mvst_bts.utils.metrics import format_metrics

def test_format_metrics():
    assert format_metrics({"a": 1}) == "a=1"
