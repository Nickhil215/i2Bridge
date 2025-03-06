import os

from app.services.base_analyzer import BaseAnalyzer
from app.services.compressed_source_analyzer import CompressedSourceAnalyzer
from app.services.git_repository_analyzer import GitRepositoryAnalyzer
from app.services.source_analyzer import SourceAnalyzer
from app.services.wheel_analyzer import WheelAnalyzer


class PackageAnalyzerFactory:
    """Factory for creating appropriate analyzer based on package type."""

    @staticmethod
    def create_analyzer(package_path: str, **kwargs) -> BaseAnalyzer:
        """
        Create and return the appropriate analyzer based on the package path.

        Args:
            package_path: Path to the package or Git URL
            **kwargs: Additional arguments for specific analyzers

        Returns:
            An appropriate analyzer instance
        """
        # Check if it's a Git URL
        if package_path.startswith(('http://', 'https://', 'git@')):
            return GitRepositoryAnalyzer(package_path, **kwargs)

        # Check if the path exists
        if not os.path.exists(package_path):
            raise ValueError(f"Package path does not exist: {package_path}")

        # Handle other package types
        if package_path.endswith('.whl'):
            return WheelAnalyzer(package_path)
        elif any(package_path.endswith(ext) for ext in
                 ('.zip', '.tar', '.tar.gz', '.tgz', '.gz')):
            return CompressedSourceAnalyzer(package_path)
        elif os.path.isdir(package_path):
            return SourceAnalyzer(package_path)
        else:
            raise ValueError(
                f"Invalid package path: {package_path}. "
                "Must be a directory, .whl file, compressed file, or Git URL"
            )
