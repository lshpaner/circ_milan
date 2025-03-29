import pytest
import subprocess
from unittest import mock

from circ_milan.functions import get_true_terminal_width


def test_get_true_terminal_width_success():
    with mock.patch("subprocess.check_output", return_value=b"180\n"):
        width = get_true_terminal_width()
        assert width == 180


def test_get_true_terminal_width_strip_spaces():
    with mock.patch("subprocess.check_output", return_value=b" 150 \n"):
        width = get_true_terminal_width()
        assert width == 150


def test_get_true_terminal_width_invalid_output():
    with mock.patch("subprocess.check_output", return_value=b"not_a_number"):
        width = get_true_terminal_width()
        assert width == 200  # fallback


def test_get_true_terminal_width_exception():
    with mock.patch(
        "subprocess.check_output",
        side_effect=OSError("tput not found"),
    ):
        width = get_true_terminal_width()
        assert width == 200  # fallback
