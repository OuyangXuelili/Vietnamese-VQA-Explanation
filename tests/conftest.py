"""Let the full CPU verification job reject silently skipped checks."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--fail-on-skip", action="store_true", default=False,
                     help="Fail verification if any test was skipped")


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if (session.config.getoption("--fail-on-skip") and reporter
            and reporter.stats.get("skipped") and exitstatus == 0):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
