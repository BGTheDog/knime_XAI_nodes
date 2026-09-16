import os
import sys
from types import SimpleNamespace

import knime.extension.testing as ktest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for path in (SRC, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


class FakeContext(ktest.TestingExecutionContext):
    """Configuration/execution context with credentials and port-group counts."""

    def __init__(
        self,
        credentials_name: str = "xai",
        password: str = "test-key",
        output_ports: list[int] | None = None,
        input_specs=None,
    ):
        super().__init__()
        self._credentials_name = credentials_name
        self._password = password
        self._output_ports = output_ports or [1, 0]
        self._input_specs = input_specs or []

    def get_credential_names(self):
        return [self._credentials_name]

    def get_credentials(self, identifier: str):
        if identifier != self._credentials_name:
            raise KeyError(identifier)
        return SimpleNamespace(
            username="",
            password=self._password,
            credential_name=identifier,
        )

    def get_connected_output_port_numbers(self):
        return list(self._output_ports)

    def get_input_specs(self):
        return list(self._input_specs)
