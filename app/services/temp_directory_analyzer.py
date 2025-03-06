import os
import tempfile
from abc import abstractmethod
from typing import Optional

from app.services import logger
from app.services.base_analyzer import BaseAnalyzer


class TemporaryDirectoryAnalyzer(BaseAnalyzer):
    """Base class for analyzers that need to work with temporary directories."""

    def __init__(self, package_path: str):
        super().__init__(package_path)
        self.original_path = package_path

    def create_temp_dir(self):
        """Create a temporary directory for extraction."""
        self.temp_dir = tempfile.mkdtemp(prefix=f'{self.__class__.__name__.lower()}_')
        logger.info(f"Created temporary directory: {self.temp_dir}")

    @abstractmethod
    def extract_contents(self):
        """Extract contents to temporary directory."""
        pass

    def analyze_package(self) -> None:
        """Template method for package analysis."""
        try:
            self.create_temp_dir()
            self.extract_contents()
            self.process_extracted_contents()
        finally:
            self.cleanup()

    def process_extracted_contents(self):
        """Process the extracted contents in temporary directory."""
        # Store the original package path
        original_path = self.package_path

        # Update package path to point to the extracted contents
        self.package_path = self.temp_dir

        # Find the root Python package directory
        package_root = self._find_package_root()
        if package_root:
            self.package_path = package_root

        try:
            for root, _, files in os.walk(self.temp_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        self._analyze_file(file_path)
        finally:
            # Restore the original package path for reporting
            self.package_path = original_path

    def _find_package_root(self) -> Optional[str]:
        """
        Find the root directory of the Python package in the extracted contents.
        This helps handle cases where the compressed file might have a root directory.
        """
        # Look for the first directory containing an __init__.py file
        for root, dirs, files in os.walk(self.temp_dir):
            if '__init__.py' in files:
                return root

            # Check first-level directories only
            if root == self.temp_dir:
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if os.path.isfile(os.path.join(dir_path, '__init__.py')):
                        return dir_path

        # If no __init__.py is found, return the first directory containing .py files
        for root, _, files in os.walk(self.temp_dir):
            if any(f.endswith('.py') for f in files):
                return root

        return self.temp_dir
