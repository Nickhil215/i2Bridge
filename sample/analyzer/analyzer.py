#!/usr/bin/env python3
"""
Python Package Analyzer
======================

A comprehensive tool for analyzing Python packages in various formats:
- Source code directories
- Wheel packages (.whl)
- Compressed files (.zip, .tar, .tar.gz, .tgz, .gz)

This analyzer provides detailed information about:
- Code structure (classes, functions, methods)
- Complexity metrics
- Dependencies
- Package metadata
- Compression statistics (for compressed files)
"""

import gzip
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import astroid
import uncompyle6
from astroid import nodes
from wheel.wheelfile import WheelFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
# Data Classes
#-----------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    """Store detailed information about a function or method."""
    name: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    complexity: int
    line_number: int
    end_line: int
    is_method: bool
    is_async: bool

@dataclass
class ClassInfo:
    """Store detailed information about a class definition."""
    name: str
    bases: List[str]
    methods: Dict[str, FunctionInfo]
    attributes: List[str]
    docstring: Optional[str]
    decorators: List[str]
    line_number: int
    end_line: int

#-----------------------------------------------------------------------------
# Base Analyzer Classes
#-----------------------------------------------------------------------------

class BaseAnalyzer(ABC):
    """Abstract base class defining the interface for package analyzers."""

    def __init__(self, package_path: str):
        """Initialize the base analyzer."""
        self.package_path = os.path.abspath(package_path)
        self.modules: Dict[str, nodes.Module] = {}
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.classes: Dict[str, ClassInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.errors: List[tuple] = []
        self.temp_dir: Optional[str] = None

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when used as context manager."""
        self.cleanup()

    def cleanup(self):
        """Clean up any temporary files created during analysis."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    @abstractmethod
    def analyze_package(self) -> None:
        """Analyze the package and extract its information."""
        pass

    def _analyze_file(self, file_path: str) -> None:
        """
        Analyze a single Python file and extract its AST information.

        Args:
            file_path: Path to the Python file to analyze
        """
        try:
            relative_path = os.path.relpath(file_path, self.package_path)
            logger.info(f"Analyzing file: {relative_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the file using astroid
            module = astroid.parse(content, path=file_path)
            self.modules[relative_path] = module

            # Analyze different aspects of the module
            self._analyze_imports(module, relative_path)
            self._analyze_classes(module, relative_path)
            self._analyze_functions(module, relative_path)

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            self.errors.append((file_path, str(e)))

    def _analyze_imports(self, module: nodes.Module, file_path: str) -> None:
        """Extract and track all imports in a module."""
        for node in module.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(node, nodes.Import):
                for name, asname in node.names:
                    self.imports[file_path].add(name)
            elif isinstance(node, nodes.ImportFrom):
                module_name = node.modname
                for name, asname in node.names:
                    if module_name:
                        self.imports[file_path].add(f"{module_name}.{name}")
                    else:
                        self.imports[file_path].add(name)

    def _analyze_functions(self, module: nodes.Module, file_path: str) -> None:
        """Analyze all functions in a module, including methods."""
        for node in module.nodes_of_class(nodes.FunctionDef):
            if isinstance(node.parent, nodes.Module):  # Only module-level functions
                func_info = self._extract_function_info(node)
                self.functions[f"{file_path}::{func_info.name}"] = func_info

    def _analyze_classes(self, module: nodes.Module, file_path: str) -> None:
        """Analyze all classes in a module."""
        for node in module.nodes_of_class(nodes.ClassDef):
            class_info = self._extract_class_info(node)
            self.classes[f"{file_path}::{class_info.name}"] = class_info

    def _extract_function_info(self, node: nodes.FunctionDef) -> FunctionInfo:
        """Extract detailed information about a function."""
        args = [arg.name for arg in node.args.args]
        returns = node.returns.as_string() if node.returns else None
        docstring = None
        if isinstance(node.doc_node, nodes.Const):
            docstring = node.doc_node.value
        complexity = self._calculate_complexity(node)

        # Fix decorator handling
        decorators = []
        if node.decorators:
            for decorator in node.decorators.nodes:
                decorators.append(decorator.as_string())

        return FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            docstring=docstring,
            decorators=decorators,  # Use the properly extracted decorators
            complexity=complexity,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            is_method=isinstance(node.parent, nodes.ClassDef),
            is_async=isinstance(node, nodes.AsyncFunctionDef)
        )

    def _extract_class_info(self, node: nodes.ClassDef) -> ClassInfo:
        """Extract detailed information about a class."""
        methods = {}
        attributes = []

        for child in node.get_children():
            if isinstance(child, nodes.FunctionDef):
                method_info = self._extract_function_info(child)
                methods[child.name] = method_info
            elif isinstance(child, nodes.AssignName):
                attributes.append(child.name)

        docstring = None
        if isinstance(node.doc_node, nodes.Const):
            docstring = node.doc_node.value

        # Fix decorator handling
        decorators = []
        if node.decorators:
            for decorator in node.decorators.nodes:
                decorators.append(decorator.as_string())

        return ClassInfo(
            name=node.name,
            bases=[base.as_string() for base in node.bases],
            methods=methods,
            attributes=attributes,
            docstring=docstring,
            decorators=decorators,  # Use the properly extracted decorators
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno
        )

    def _calculate_complexity(self, node: nodes.NodeNG) -> int:
        """
        Calculate cyclomatic complexity of a function.

        Counts decision points in the code:
        - if/elif statements
        - for/while loops
        - except blocks
        - boolean operations
        """
        complexity = 1  # Base complexity

        # Count branching statements
        for child in node.nodes_of_class((
            nodes.If, nodes.While, nodes.For,
            nodes.ExceptHandler, nodes.With,
            nodes.BoolOp
        )):
            complexity += 1

            # Add complexity for boolean operations
            if isinstance(child, nodes.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def get_package_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics for the package."""
        return {
            'total_files': len(self.modules),
            'total_classes': len(self.classes),
            'total_functions': len(self.functions),
            'average_complexity': sum(f.complexity for f in self.functions.values()) / len(self.functions) if self.functions else 0,
            'total_lines': sum(f.end_line - f.line_number for f in self.functions.values()),
            'errors': len(self.errors)
        }

    def generate_report(self) -> str:
        """Generate a detailed report of the package analysis."""
        metrics = self.get_package_metrics()

        report = [
            "Package Analysis Report",
            "=====================\n",
            f"Package: {self.package_path}\n",
            "Metrics:",
            f"- Total Files: {metrics['total_files']}",
            f"- Total Classes: {metrics['total_classes']}",
            f"- Total Functions: {metrics['total_functions']}",
            f"- Average Complexity: {metrics['average_complexity']:.2f}",
            f"- Total Lines of Code: {metrics['total_lines']}",
            f"- Analysis Errors: {metrics['errors']}\n",
            "Class Hierarchy:"
        ]

        # Add class information
        for class_path, class_info in sorted(self.classes.items()):
            report.extend([
                f"\n{class_path}:",
                f"  Bases: {', '.join(class_info.bases)}",
                f"  Methods: {len(class_info.methods)}",
                f"  Attributes: {len(class_info.attributes)}"
            ])

        # Add error information if any
        if self.errors:
            report.extend([
                "\nErrors:",
                *[f"- {path}: {error}" for path, error in self.errors]
            ])

        return '\n'.join(report)

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
            self._analyze_extracted_contents()
        finally:
            self.cleanup()

    def _analyze_extracted_contents(self):
        """Analyze all Python files in the extracted contents."""
        logger.info("Starting analysis of extracted contents")

        try:
            # Add extracted directory to Python path for proper imports
            sys.path.insert(0, self.temp_dir)

            # Change: Walk through temp_dir instead of package_path
            for root, _, files in os.walk(self.temp_dir):
                python_files = [f for f in files if f.endswith('.py')]

                for file in python_files:
                    file_path = os.path.join(root, file)
                    try:
                        logger.info(f"Analyzing file: {file_path}")
                        self._analyze_file(file_path)
                    except Exception as e:
                        logger.error(f"Error analyzing {file_path}: {str(e)}")
                        self.errors.append((file_path, str(e)))

        finally:
            # Change: Make sure we remove the correct path
            if sys.path and sys.path[0] == self.temp_dir:
                sys.path.pop(0)

    """
    Base class for analyzers that need to work with temporary directories.
    This includes wheel packages and compressed source files.
    """

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

#-----------------------------------------------------------------------------
# Specific Analyzer Implementations
#-----------------------------------------------------------------------------

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

class WheelAnalyzer(TemporaryDirectoryAnalyzer):
    """Analyzer for wheel packages (.whl files)."""

    def __init__(self, package_path: str):
        super().__init__(package_path)
        self.wheel_metadata = {}

    def extract_contents(self):
        """Extract the wheel file contents."""
        try:
            with WheelFile(self.package_path) as wf:
                wf.extractall(self.temp_dir)
                logger.info("Successfully extracted wheel contents")
                self._extract_wheel_metadata()
        except Exception as e:
            logger.error(f"Error extracting wheel file: {str(e)}")
            raise

    def _extract_wheel_metadata(self):
        """Extract wheel-specific metadata."""
        for item in os.listdir(self.temp_dir):
            if item.endswith('.dist-info'):
                dist_info_dir = os.path.join(self.temp_dir, item)

                metadata_path = os.path.join(dist_info_dir, 'METADATA')
                if os.path.exists(metadata_path):
                    self.wheel_metadata['metadata'] = self._parse_metadata_file(metadata_path)

                wheel_path = os.path.join(dist_info_dir, 'WHEEL')
                if os.path.exists(wheel_path):
                    self.wheel_metadata['wheel'] = self._parse_wheel_file(wheel_path)
                break

    def _parse_metadata_file(self, metadata_path: str) -> Dict[str, Any]:
        """Parse the METADATA file from wheel's dist-info directory."""
        metadata = {}
        current_section = None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                    current_section = key.strip()
                elif current_section:
                    metadata[current_section] += '\n' + line

        return metadata

    def _parse_wheel_file(self, wheel_path: str) -> Dict[str, Any]:
        """Parse the WHEEL file from the dist-info directory."""
        wheel_data = {}

        with open(wheel_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    wheel_data[key.strip()] = value.strip()

        return wheel_data

    def generate_report(self) -> str:
        """Generate a detailed report including wheel-specific information."""
        # Get base report from parent class
        report = super().generate_report()

        # Add wheel metadata section
        wheel_section = [
            "\nWheel Package Information",
            "========================",
            f"Package Name: {self.wheel_metadata.get('metadata', {}).get('Name', 'Unknown')}",
            f"Version: {self.wheel_metadata.get('metadata', {}).get('Version', 'Unknown')}",
            f"Python Version: {self.wheel_metadata.get('wheel', {}).get('Python-Version', 'Unknown')}"
        ]

        # Add dependencies if available
        if 'metadata' in self.wheel_metadata:
            requires = [key for key in self.wheel_metadata['metadata'].keys()
                      if key.startswith('Requires-')]
            if requires:
                wheel_section.extend([
                    "\nDependencies:",
                    *[f"- {req}" for req in requires]
                ])

        return report + '\n\n' + '\n'.join(wheel_section)

class CompressedSourceAnalyzer(TemporaryDirectoryAnalyzer):
    """Analyzer for compressed source packages (.zip, .tar.gz, etc.)."""

    def _is_tar_file(self) -> bool:
        """Check if the file is a tar archive."""
        return (self.package_path.endswith('.tar') or
                self.package_path.endswith('.tar.gz') or
                self.package_path.endswith('.tgz'))

    def _is_zip_file(self) -> bool:
        """Check if the file is a zip archive."""
        return self.package_path.endswith('.zip')

    def _is_gzip_file(self) -> bool:
        """Check if the file is a gzip archive."""
        return self.package_path.endswith('.gz') and not self._is_tar_file()

    def extract_contents(self):
        """Extract contents based on the compression type."""
        logger.info(f"Extracting compressed file: {self.package_path}")

        if self._is_zip_file():
            self._extract_zip()
        elif self._is_tar_file():
            self._extract_tar()
        elif self._is_gzip_file():
            self._extract_gzip()
        else:
            raise ValueError(f"Unsupported compression format: {self.package_path}")

    def _extract_zip(self):
        """Extract a ZIP archive."""
        with zipfile.ZipFile(self.package_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

    def _extract_tar(self):
        """Extract a TAR archive (including .tar.gz)."""
        with tarfile.open(self.package_path, 'r:*') as tar_ref:
            # Check for potential path traversal vulnerabilities
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted path traversal in tar file")
                tar.extractall(path)

            safe_extract(tar_ref, self.temp_dir)

    def _extract_gzip(self):
        """Extract a GZIP file."""
        output_path = os.path.join(self.temp_dir,
                                 os.path.basename(self.package_path)[:-3])
        with gzip.open(self.package_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def generate_report(self) -> str:
        """Generate a report including compression-specific information."""
        # Store the original package path
        original_path = self.package_path

        try:
            # Temporarily set package path to the extracted contents for correct relative paths
            if self.temp_dir:
                package_root = self._find_package_root()
                if package_root:
                    self.package_path = package_root

            # Get base report with correct paths
            report = super().generate_report()

            # Add compression information
            compression_info = self._get_compression_info()
            compression_section = [
                "\nCompressed Package Information",
                "===========================",
                f"Compression Type: {compression_info['type']}",
                f"Original File: {original_path}",
                f"Original Size: {compression_info['original_size']}",
                f"Compressed Size: {compression_info['compressed_size']}",
                f"Compression Ratio: {compression_info['ratio']:.2f}%"
            ]

            return report + '\n\n' + '\n'.join(compression_section)

        finally:
            # Restore the original package path
            self.package_path = original_path

    def _get_compression_info(self) -> Dict[str, Any]:
        """Get information about the compression."""
        original_size = 0
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                original_size += os.path.getsize(file_path)

        compressed_size = os.path.getsize(self.original_path)

        compression_type = 'Unknown'
        if self._is_zip_file():
            compression_type = 'ZIP'
        elif self._is_tar_file():
            compression_type = 'TAR' + ('.GZ' if self.package_path.endswith('.gz') else '')
        elif self._is_gzip_file():
            compression_type = 'GZIP'

        return {
            'type': compression_type,
            'original_size': self._format_size(original_size),
            'compressed_size': self._format_size(compressed_size),
            'ratio': (compressed_size / original_size * 100) if original_size > 0 else 0
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

class GitRepositoryAnalyzer(CompressedSourceAnalyzer):
    """
    Analyzer for Git repositories. This class handles downloading and analyzing
    source code from various Git hosting platforms.
    """

    # Mapping of Git hosting platforms to their API endpoints
    GIT_PLATFORMS = {
        'github.com': {
            'api_url': 'https://api.github.com/repos/{owner}/{repo}',
            'download_url': 'https://github.com/{owner}/{repo}/archive/{branch}.zip',
            'raw_pattern': r'github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/\.]+)'
        },
        'gitlab.com': {
            'api_url': 'https://gitlab.com/api/v4/projects/{path}',
            'download_url': 'https://gitlab.com/{path}/-/archive/{branch}/{repo}-{branch}.zip',
            'raw_pattern': r'gitlab\.com[:/](?P<path>[^/]+/[^/\.]+)'
        },
        'bitbucket.org': {
            'api_url': 'https://api.bitbucket.org/2.0/repositories/{owner}/{repo}',
            'download_url': 'https://bitbucket.org/{owner}/{repo}/get/{branch}.zip',
            'raw_pattern': r'bitbucket\.org[:/](?P<owner>[^/]+)/(?P<repo>[^/\.]+)'
        }
    }

    def __init__(self, git_url: str, branch: str = 'main'):
        """
        Initialize the Git repository analyzer.

        Args:
            git_url: URL to the Git repository (HTTPS or SSH format)
            branch: Branch to analyze (default: 'main')
        """
        self.git_url = git_url
        self.branch = branch
        self.repo_info = {}

        # Create a temporary file for the downloaded repository
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            self.temp_zip_path = temp_file.name

        # Initialize the parent class with the temporary zip file
        super().__init__(self.temp_zip_path)

        # Store additional repository information
        self.platform = None
        self.owner = None
        self.repo_name = None
        self.repo_path = None

    def _parse_git_url(self) -> Tuple[str, str, str, str]:
        """
        Parse a Git URL into its components.

        Returns:
            Tuple containing (platform, owner/path, repository name, repository path)
        """
        # Convert SSH URLs to HTTPS format for parsing
        if self.git_url.startswith('git@'):
            self.git_url = self.git_url.replace(':', '/').replace('git@', 'https://')

        # Parse the URL
        parsed_url = urllib.parse.urlparse(self.git_url)
        platform = parsed_url.netloc

        # Find the matching platform and extract information
        for host, config in self.GIT_PLATFORMS.items():
            if host in platform:
                match = re.search(config['raw_pattern'], self.git_url)
                if match:
                    groups = match.groupdict()
                    if 'path' in groups:
                        # GitLab style URL
                        path = groups['path']
                        owner, repo = path.split('/')
                        return host, owner, repo, path
                    else:
                        # GitHub/Bitbucket style URL
                        return host, groups['owner'], groups['repo'], f"{groups['owner']}/{groups['repo']}"

        raise ValueError(f"Unsupported Git platform or invalid URL: {self.git_url}")

    def _download_repository(self):
        """
        Download the repository from the Git hosting platform.
        This method handles authentication and different hosting platforms.
        """
        try:
            # Parse the Git URL
            self.platform, self.owner, self.repo_name, self.repo_path = self._parse_git_url()
            platform_config = self.GIT_PLATFORMS[self.platform]

            # Get repository information from API
            api_url = platform_config['api_url'].format(
                owner=self.owner,
                repo=self.repo_name,
                path=urllib.parse.quote(self.repo_path, safe='')
            )

            # Prepare headers for API request
            headers = {'Accept': 'application/json'}

            # Add authentication if environment variables are set
            if 'GITHUB_TOKEN' in os.environ and self.platform == 'github.com':
                headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
            elif 'GITLAB_TOKEN' in os.environ and self.platform == 'gitlab.com':
                headers['PRIVATE-TOKEN'] = os.environ['GITLAB_TOKEN']

            # Get repository information
            try:
                req = urllib.request.Request(api_url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    self.repo_info = json.loads(response.read().decode())
            except Exception as e:
                logger.warning(f"Could not fetch repository information: {e}")

            # Get the default branch if none specified
            if not self.branch:
                self.branch = self.repo_info.get('default_branch', 'main')

            # Download the repository
            download_url = platform_config['download_url'].format(
                owner=self.owner,
                repo=self.repo_name,
                path=self.repo_path,
                branch=self.branch
            )

            logger.info(f"Downloading repository from {download_url}")

            req = urllib.request.Request(download_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(self.temp_zip_path, 'wb') as f:
                    f.write(response.read())

            logger.info("Repository downloaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to download repository: {str(e)}")

    def analyze_package(self) -> None:
        """
        Download and analyze the Git repository.
        """
        try:
            self._download_repository()
            super().analyze_package()
        finally:
            # Clean up the temporary zip file
            if os.path.exists(self.temp_zip_path):
                os.unlink(self.temp_zip_path)
                logger.info("Cleaned up temporary zip file")

    def generate_report(self) -> str:
        """
        Generate a detailed report including Git repository information.
        """
        # Get the base report
        report = super().generate_report()

        # Add Git repository information
        git_section = [
            "\nGit Repository Information",
            "========================",
            f"Repository: {self.git_url}",
            f"Platform: {self.platform}",
            f"Branch: {self.branch}",
        ]

        # Add additional repository information if available
        if self.repo_info:
            git_section.extend([
                f"Description: {self.repo_info.get('description', 'N/A')}",
                f"Stars: {self.repo_info.get('stargazers_count', 'N/A')}",
                f"Forks: {self.repo_info.get('forks_count', 'N/A')}",
                f"Last Updated: {self.repo_info.get('updated_at', 'N/A')}",
                f"Language: {self.repo_info.get('language', 'N/A')}"
            ])

        return report + '\n\n' + '\n'.join(git_section)

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

def analyze_package(package_path: str, output_file: Optional[str] = None, **kwargs) -> str:
    """
    Analyze a Python package and optionally save the report.
    This function serves as the main entry point for package analysis.

    Args:
        package_path: Path to the Python package or Git URL
        output_file: Optional path to save the analysis report
        **kwargs: Additional arguments passed to specific analyzers

    Returns:
        String containing the analysis report

    Example:
        # Analyze different package types
        report = analyze_package("./my_package")              # Source directory
        report = analyze_package("package-1.0.whl")           # Wheel package
        report = analyze_package("package.tar.gz")            # Compressed file
        report = analyze_package("https://github.com/user/repo") # Git repository

        # Analyze with specific options
        report = analyze_package("https://github.com/user/repo",
                               branch="develop",
                               output_file="report.txt")
    """
    try:
        with PackageAnalyzerFactory.create_analyzer(package_path, **kwargs) as analyzer:
            analyzer.analyze_package()
            report = analyzer.generate_report()

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                    logger.info(f"Analysis report saved to: {output_file}")

            return report

    except Exception as e:
        error_msg = f"Error analyzing package: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def main():
    """
    Command-line interface for the package analyzer.
    Provides a simple way to analyze packages from the command line.
    """
    if len(sys.argv) < 2:
        print("Usage: python package_analyzer.py <package_path> [output_file]")
        print("\nSupported package types:")
        print("  - Python source directories")
        print("  - Wheel packages (.whl)")
        print("  - ZIP archives (.zip)")
        print("  - TAR archives (.tar, .tar.gz, .tgz)")
        print("  - GZIP files (.gz)")
        print("\nArguments:")
        print("  package_path: Path to the Python package to analyze")
        print("  output_file: Optional path to save the analysis report")
        sys.exit(1)

    package_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        report = analyze_package(package_path, output_file)
        print(report)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
