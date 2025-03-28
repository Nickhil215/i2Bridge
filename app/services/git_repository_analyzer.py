import os
import shutil
import tempfile
from typing import Optional, List, Set

from git import GitCommandError, Repo

from app import logger
from app.services.base_analyzer import BaseAnalyzer


def parse_requirements(file_path: str, requirement_info: Set[str]):

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip lines starting with '-e' or containing '[', which are not valid package names
            if line.startswith('-e') or '[' in line:
                continue
            if line == '.':
                continue

            # Ignore empty lines and comments
            if not line or line.startswith("#"):
                continue

            requirement_info.add(line)

    return requirement_info


class GitRepositoryAnalyzer(BaseAnalyzer):
    """Analyzer for Git repositories"""

    def __init__(self, git_url: str, branch: str):
        super().__init__(git_url)
        self.git_url = git_url
        self.branch = branch
        self.temp_dir: Optional[str] = None  # Add temp_dir attribute

    def create_temp_dir(self):
        """Create temporary directory for cloning"""
        self.temp_dir = tempfile.mkdtemp(prefix='git_analyzer_')
        logger.info(f"Created temp directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

    def analyze_package(self) -> None:
        """Analyze Git repository"""
        self.create_temp_dir()
        try:
            self._clone_repository()
            self._process_cloned_repo()
        finally:
            self.cleanup()

    def _clone_repository(self):
        """Clone the Git repository with fallback branch detection"""
        try:
            logger.info(f"Cloning {self.git_url}...")
            try:
                Repo.clone_from(
                    self.git_url,
                    self.temp_dir,
                    depth=1,
                    branch=self.branch
                )
            except GitCommandError:
                # Fallback to master
                Repo.clone_from(
                    self.git_url,
                    self.temp_dir,
                    depth=1,
                    branch='master'
                )
                self.branch = 'master'
            logger.info(f"Cloning from {self.branch} branch")
        except GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

    def _process_cloned_repo(self):
        """Process the cloned repository"""
        # Find the actual package root
        package_root = self._find_package_root()
        if package_root:
            self.package_path = package_root

        # Analyze requirements to get packages to install
        requirement_info = set()
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if "requirement" in file and file.endswith(".txt"):
                    requirement_info = parse_requirements(os.path.join(root, file), requirement_info)

        # Analyze all Python files
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.py'):
                    # print("------------root",root )
                    # print("------------file",file )
                    self._analyze_file(os.path.join(root, file), requirement_info, self.git_url, self.branch)

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
