"Tests for the `segment` module"

# pylint: disable=redefined-outer-name

from unittest import mock

import pytest

from arbitragelab.util import segment

@pytest.fixture
def mock_getmac():
    """Mock for the getmac dependency"""
    with mock.patch("arbitragelab.util.segment.getmac") as mock_getmac:
        yield mock_getmac


def test_get_mac_default_interface(mock_getmac):
    """Test if we can get the MAC address for the default interface"""
    mock_getmac.get_mac_address.return_value = "MAC_ADDRESS"

    actual_result = segment.get_mac()
    expected_result = "MAC_ADDRESS"
    assert actual_result == expected_result
    mock_getmac.get_mac_address.assert_called_with(None)

def test_get_mac_custom_interface(monkeypatch, mock_getmac):
    """Test if we can get the MAC address for a custom interface"""
    monkeypatch.setenv("ARBLAB_MAC_INTERFACE", "CUSTOM_INTERFACE")
    mock_getmac.get_mac_address.return_value = "MAC_ADDRESS"

    actual_result = segment.get_mac()
    expected_result = "MAC_ADDRESS"
    assert actual_result == expected_result
    mock_getmac.get_mac_address.assert_called_with("CUSTOM_INTERFACE")

def test_get_mac_raises_error_if_no_mac_address_found(mock_getmac):  # pylint: disable=invalid-name
    """Test if get_mac raises an error if a MAC address cannot be found"""
    mock_getmac.get_mac_address.return_value = None

    with pytest.raises(ValueError):
        segment.get_mac()
