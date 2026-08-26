from tests._golden import assert_descriptor_golden


def test_coulombmatrix_golden():
    assert_descriptor_golden("CoulombMatrix")
