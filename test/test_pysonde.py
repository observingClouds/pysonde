import pysonde._helpers as h


def test_find_files():
    assert h.find_files(["x.mwx", "a.mwx"]) == ["a.mwx", "x.mwx"]
