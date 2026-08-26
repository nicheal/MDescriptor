from tests._golden import assert_descriptor_golden


def test_valleoganov_golden():
    assert_descriptor_golden("ValleOganov")
