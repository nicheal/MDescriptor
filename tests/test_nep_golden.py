import pytest

from tests._golden import assert_descriptor_golden

pytestmark = pytest.mark.model


def test_nep_golden():
    assert_descriptor_golden("NEP")
