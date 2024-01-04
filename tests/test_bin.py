import logging
import pathlib

import pytest

from turtlemd.bin import main
from turtlemd.version import __version__

HERE = pathlib.Path(__file__).resolve().parent


def test_main_info(monkeypatch, capsys):
    """Test that we can run turtlemd with "-v"."""

    for arg in ("-v", "--version"):
        test_args = ["turtlemd", arg]

        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert __version__ in captured.out


def test_main_input(monkeypatch, caplog, capsys):
    """Test that we can read an input file."""
    for arg in ("-i", "--input_file"):
        test_args = ["turtlemd", arg]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "expected one argument" in captured.err

        tomlfile = HERE / "inout" / "md.toml"

        test_args = ["turtlemd", arg, str(tomlfile)]
        monkeypatch.setattr("sys.argv", test_args)
        with caplog.at_level(logging.INFO):
            main()
            assert "Reading settings from file" in caplog.text
            assert "Created system" in caplog.text
