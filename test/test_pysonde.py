import pysonde.pysonde as pysonde


def test_find_files():
    assert pysonde.find_files(["x.mwx", "a.mwx"]) == ["a.mwx", "x.mwx"]
