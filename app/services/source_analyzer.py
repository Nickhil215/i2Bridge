import os
import sys

from app.services.base_analyzer import BaseAnalyzer


class SourceAnalyzer(BaseAnalyzer):
    """Analyzer for Python source packages (directories)."""

    def analyze_package(self) -> None:
        """Analyze a source package by walking through its files."""
        try:
            sys.path.insert(0, os.path.dirname(self.package_path))

            for root, _, files in os.walk(self.package_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        self._analyze_file(file_path)
        finally:
            sys.path.pop(0)