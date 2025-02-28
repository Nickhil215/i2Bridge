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
import shutil
import sys
import tarfile
import tempfile
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Set
import re
import astroid
import requests
from astroid import nodes
from git import GitCommandError, Repo
from requests_toolbelt.multipart.encoder import MultipartEncoder
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
class ImportInfo:
  """Store detailed information about an import."""
  module_name: str
  alias: Optional[str]

@dataclass
class ApiInfo:
    """Store detailed information about a API endpoint"""
    path: str
    http_method: str
    name: str
    path_params: List[str]
    request_params: List[str]
    request_body: Optional[str]
    response_body: Optional[str]

@dataclass
class FunctionInfo:
    """Store detailed information about a function or method."""
    name: str
    module: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    comments: List[str]
    decorators: List[str]
    complexity: int
    line_number: int
    end_line: int
    is_method: bool
    is_async: bool
    signature: str

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
        self.apis: Dict[str, ApiInfo] = {}
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
        """Analyze a single Python file and extract its AST information."""
        try:
            relative_path = os.path.relpath(file_path, self.package_path)
            logger.info(f"Analyzing file: {relative_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the file using astroid
            module = astroid.parse(source, path=file_path)
            self.modules[relative_path] = module

            # Pass source to analysis methods
            self._analyze_imports(module, relative_path, source)
            self._analyze_classes(module, relative_path, source)
            self._analyze_functions(module, relative_path, source)
            self._analyze_api_endpoint(module, file_path, source)

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            self.errors.append((file_path, str(e)))

    def _analyze_imports(self, module: nodes.Module, file_path: str, source: str) -> None:
      """Extract and track all imports in a module."""
      for node in module.nodes_of_class((nodes.Import, nodes.ImportFrom)):
        if isinstance(node, nodes.Import):
          for name, asname in node.names:
            import_name = name if asname is None else asname
            if file_path not in self.imports:
              self.imports[file_path] = set()
            self.imports[file_path].add(import_name)
        elif isinstance(node, nodes.ImportFrom):
          module_name = node.modname
          for name, asname in node.names:
            import_name = f"{module_name}.{name}" if asname is None else f"{module_name}.{name} as {asname}"
            if file_path not in self.imports:
              self.imports[file_path] = set()
            self.imports[file_path].add(import_name)

    def _analyze_functions(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all functions in a module, including methods."""
        for node in module.nodes_of_class(nodes.FunctionDef):
            if isinstance(node.parent, nodes.Module):  # Only module-level functions
                func_info = self._extract_function_info(node, source)
                self.functions[f"{file_path}::{func_info.name}"] = func_info

    def _analyze_api_endpoint(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all functions in a module, including methods."""
        for node in module.nodes_of_class(nodes.FunctionDef):
            if isinstance(node.parent, nodes.Module):  # Only module-level functions
                api_info = self._extract_api_endpoint_info(node)
                self.apis[f"{file_path}::{api_info.name}"] = api_info

    def _analyze_classes(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all classes in a module."""
        for node in module.nodes_of_class(nodes.ClassDef):
            class_info = self._extract_class_info(node, source)
            self.classes[f"{file_path}::{class_info.name}"] = class_info

    def _extract_function_info(self, node: nodes.FunctionDef, source: str) -> FunctionInfo:
        """Extract detailed information about a function."""
        args = [arg.name for arg in node.args.args]
        returns = node.returns.as_string() if node.returns else None
        docstring = None
        if isinstance(node.doc_node, nodes.Const):
            docstring = node.doc_node.value.strip()
        complexity = self._calculate_complexity(node)

        # Fix decorator handling
        decorators = []
        if node.decorators:
            for decorator in node.decorators.nodes:
                decorators.append(decorator.as_string())

        comments = []
        line_number=node.lineno
        end_line=node.end_lineno or node.lineno

        for line_num in range(line_number, end_line + 1):
            line = source.splitlines()[line_num - 1].strip()
            if '#' in line:
                comment_text = line.split('#', 1)[1].strip()
                comments.append(comment_text)

        return FunctionInfo(
            name=node.name,
            module=node.root().name,
            args=args,
            returns=returns,
            docstring=docstring,
            comments=comments,
            decorators=decorators,  # Use the properly extracted decorators
            complexity=complexity,
            line_number=line_number,
            end_line=end_line,
            is_method=isinstance(node.parent, nodes.ClassDef),
            is_async=isinstance(node, nodes.AsyncFunctionDef),
            signature=f"{node.name}({', '.join(arg.name for arg in node.args.args)})"
        )

    def _extract_class_info(self, node: nodes.ClassDef, source: str) -> ClassInfo:
        """Extract detailed information about a class."""
        methods = {}
        attributes = []

        for child in node.get_children():
            if isinstance(child, nodes.FunctionDef):
                method_info = self._extract_function_info(child, source)
                methods[child.name] = method_info
            elif isinstance(child, nodes.AnnAssign):  # Type-hinted attributes
                attributes.append(child.target.name)
            elif isinstance(child, nodes.Assign):  # Regular assignments
                attributes.extend(
                    target.name for target in child.targets if isinstance(target, nodes.AssignName)
                )

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

    def _extract_api_endpoint_info(self, node: nodes.FunctionDef) -> ApiInfo | None:
        """Extract detailed information about a api endpoint."""

        endpoint_info = {}
        path_params = []
        request_params = []
        request_body = None
        response_body = None

        if node.decorators:
            for decorator in node.decorators.nodes:
                decorator_str = decorator.as_string()

                if 'app' in decorator_str or 'router' in decorator_str:
                    method_mapping = {
                        'post': 'POST',
                        'get': 'GET',
                        'put': 'PUT',
                        'delete': 'DELETE',
                        'patch': 'PATCH',
                        'options': 'OPTIONS',
                        'head': 'HEAD'
                    }

                    for httpMethod, method_name in method_mapping.items():
                        if httpMethod in decorator_str.lower():
                            endpoint_info['method'] = method_name
                            break
                    else:
                        endpoint_info['method'] = 'UNKNOWN'  # Default if no method is matched
                    endpoint_info['path'] = decorator.args[0].value

                    path = endpoint_info['path']
                    path_params = re.findall(r'\{([^}]+)\}', path)

            # Extract function parameters
            if node.args:
                request_params = [arg.name for arg in node.args.args]

            for stmt in node.body:
                # Extract request body (typically found in POST/PUT/PATCH requests)
                if isinstance(stmt, astroid.Assign):
                    if isinstance(stmt.value, astroid.Attribute) and stmt.value.attrname in ['json', 'data', 'form']:
                        request_body = stmt.value.attrname

                # Extract response body
                if isinstance(stmt, astroid.Return):
                    response_body = str(stmt.value)

        return ApiInfo(
            name=node.name,
            path=endpoint_info.get('path'),
            http_method=endpoint_info.get('method', None),
            path_params=path_params if request_params else None,
            request_params=request_params if request_params else None,
            request_body=request_body if request_body else None,
            response_body=response_body if response_body else None
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
        complexity = 1

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
            'total_imports' : sum(len(imports) for imports in self.imports.values()),
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
            f"- Total Imports: {metrics['total_imports']}",
            f"- Average Complexity: {metrics['average_complexity']:.2f}",
            f"- Total Lines of Code: {metrics['total_lines']}",
            f"- Analysis Errors: {metrics['errors']}\n",
        ]

        # Add apis information
        report.extend(["\nAPIs:"])
        for func_path, api_info in sorted(self.apis.items()):
            if api_info.http_method is not None:
                report.extend([
                    f"\n  {func_path}:",
                    f"    name: {api_info.name}",
                    f"    path: {api_info.path}",
                    f"    method: {api_info.http_method}",
                    f"    path_param: {api_info.path_params}",
                    f"    request_param: {api_info.request_params}",
                    f"    request_body: {api_info.request_body}",
                    f"    response_body: {api_info.response_body}"
                ])

        # Add function information
        report.extend(["\nFunctions:"])
        for func_path, func_info in sorted(self.functions.items()):
            report.extend([
                f"\n  {func_path}:",
                f"    name: {func_info.name}",
                f"    Module: {func_info.module if func_info.module else 'None'}",
                f"    Signature: {func_info.signature}",
                f"    Parameters: {func_info.args if func_info.args else 'None'}",
                f"    Returns: {func_info.returns if func_info.returns else 'None'}",
                f"    Start Line: {func_info.line_number}",
                f"    End Line: {func_info.end_line}",
                f"    Complexity: {func_info.complexity}",
                f"    Is Method: {func_info.is_method}",
                f"    Is Async: {func_info.is_async}",
                f"    Decorators: {', '.join(func_info.decorators) or 'None'}",
                f"    Docstring: {func_info.docstring if func_info.docstring else 'None'}",
                f"    Comments: {func_info.comments if func_info.comments else 'None'}"
            ])

        # Add class information
        report.extend(["\nClass Hierarchy:"])
        for class_path, class_info in sorted(self.classes.items()):
            report.extend([
                f"\n  {class_path}:",
                f"    Bases: {', '.join(class_info.bases) if class_info.bases else 'None'}",
                f"    Methods: {len(class_info.methods)}",
                f"    Attributes: {len(class_info.attributes)}",
                f"    Start Line: {class_info.line_number}",
                f"    End Line: {class_info.end_line}",
                f"    Decorators: {', '.join(class_info.decorators) if class_info.decorators else 'None'}",
                f"    Docstring: {class_info.docstring.strip() if class_info.docstring else 'None'}"
            ])

        # Add import information
        if self.imports:
          report.extend(["\nImports:"])
          for file_path, imports in sorted(self.imports.items()):
            report.append(f"  {file_path}:")
            for import_name in sorted(imports):
              report.append(f"    - {import_name}")

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

class GitRepositoryAnalyzer(BaseAnalyzer):
    """Analyzer for Git repositories"""

    def __init__(self, git_url: str, branch: str = 'master'):
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
            # Try main first
            try:
                Repo.clone_from(
                    self.git_url,
                    self.temp_dir,
                    depth=1,
                    branch='main'
                )
            except GitCommandError:
                # Fallback to master
                Repo.clone_from(
                    self.git_url,
                    self.temp_dir,
                    depth=1,
                    branch='master'
                )
        except GitCommandError as e:
            raise RuntimeError(f"Failed to clone repository: {str(e)}")

    def _process_cloned_repo(self):
        """Process the cloned repository"""
        # Find the actual package root
        package_root = self._find_package_root()
        if package_root:
            self.package_path = package_root

        # Analyze all Python files
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.py'):
                    self._analyze_file(os.path.join(root, file))

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

class ApiException(Exception):
    """ Custom exception for API-related errors """
    pass


def upload_to_content_service(content):
    file_name = f"FILE_{uuid.uuid1().hex}.txt"

    url = "https://ig.aidtaas.com/mobius-content-service/v1.0/content/upload?filePath=python"
    bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3Ny1NUVdFRTNHZE5adGlsWU5IYmpsa2dVSkpaWUJWVmN1UmFZdHl5ejFjIn0.eyJleHAiOjE3NDA2OTA3MDcsImlhdCI6MTc0MDY1NDcwNywianRpIjoiY2Y0NzRlMzMtZTIwMy00MDdiLWI0MTItZDk3OWQyODAzNGNmIiwiaXNzIjoiaHR0cDovL2tleWNsb2FrLXNlcnZpY2Uua2V5Y2xvYWsuc3ZjLmNsdXN0ZXIubG9jYWw6ODA4MC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIkJPTFRaTUFOTl9CT1RfbW9iaXVzIiwiUEFTQ0FMX0lOVEVMTElHRU5DRV9tb2JpdXMiLCJNT05FVF9tb2JpdXMiLCJWSU5DSV9tb2JpdXMiLCJhY2NvdW50Il0sInN1YiI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInR5cCI6IkJlYXJlciIsImF6cCI6IkhPTEFDUkFDWV9tb2JpdXMiLCJzaWQiOiIzZjg0Mjg1NS05MTYxLTRjY2UtYWQ0Yi01YjlhYzU5NmZlOTMiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIkhDWV9URU5BTlRfQ1VTVE9NIiwib2ZmbGluZV9hY2Nlc3MiLCJCT0JfVEVOQU5UX0NVU1RPTSIsInVtYV9hdXRob3JpemF0aW9uIiwiUElfVEVOQU5UX0NVU1RPTSIsIk1PTkVUX1RFTkFOVF9DVVNUT00iXX0sInJlc291cmNlX2FjY2VzcyI6eyJCT0xUWk1BTk5fQk9UX21vYml1cyI6eyJyb2xlcyI6WyJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9QUk9DRVNTX0RFRklOSVRJT04iLCJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9JTlNUQU5DRSIsIlRFTkFOVF9XT1JLRkxPV19SRVRSSUVWRV9BTEwiLCJURU5BTlRfV09SS0ZMT1dfQUNUSVZBVEVfSU5TVEFOQ0UiLCJURU5BTlRfV09SS0ZMT1dfRVhFQ1VURSIsIlRFTkFOVF9XT1JLRkxPV19SRVRSSUVWRV9IT01FIiwiVEVOQU5UX1dPUktGTE9XX1JFVFJJRVZFX1NVU1BFTkRFRF9QUk9DRVNTX0RFRlMiLCJURU5BTlRfV09SS0ZMT1dfQUNUSVZBVEVfUlVOX0pPQiIsIlRFTkFOVF9XT1JLRkxPV19TVEFUVVMiLCJURU5BTlRfV09SS0ZMT1dfQ1JFQVRFIiwiVEVOQU5UX1dPUktGTE9XX0FDVElWSVRZX0RBVEFfQ09VTlQiLCJCT0xUWk1BTk5fQk9UX1VTRVIiLCJURU5BTlRfV09SS0ZMT1dfREVQTE9ZX1ZFUlNJT04iLCJURU5BTlRfV09SS0ZMT1dfSU1QT1JUIiwiVEVOQU5UX1dPUktGTE9XX1VQREFURSIsIlRFTkFOVF9XT1JLRkxPV19ERVBMT1kiLCJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9SVU5fSk9CIiwiVEVOQU5UX1dPUktGTE9XX1JFVFJJRVZFIiwiVEVOQU5UX1dPUktGTE9XX0RFTEVURSIsIlRFTkFOVF9XT1JLRkxPV19UUklHR0VSX01FU1NBR0VfTElTVEVORVIiXX0sIkhPTEFDUkFDWV9tb2JpdXMiOnsicm9sZXMiOlsiVEVOQU5UX1BST0RVQ1RfUkVUUklFVkVfTUFTVEVSX0NPTkZJRyIsIlRFTkFOVF9JTlZJVEFUSU9OX1JFU1BPTkQiLCJURU5BTlRfTkVHT1RJQVRJT05fQVBQUk9WQUxfQlVZRVIiLCJURU5BTlRfSU5WSVRBVElPTl9SRVRSSUVWRSIsIlRFTkFOVF9PUkdBTklaQVRJT05fQ1JFQVRFIiwiVEVOQU5UX0NSRUFURV9URU5BTlQiLCJURU5BTlRfU1VCX0FMTElBTkNFX1JFVFJJRVZFIiwiVEVOQU5UX1BMQVRGT1JNX0NSRUFURSIsIkhPTEFDUkFDWV9VU0VSIiwiVEVOQU5UX1JBVEVDQVJEX0NSRUFURSIsIlRFTkFOVF9QTEFURk9STV9SRVRSSUVWRSIsIlRFTkFOVF9SRVRSSUVWRV9CWV9CVVlFUl9QTEFURk9STSIsIlRFTkFOVF9TVUJfQUxMSUFOQ0VfUkVUUklFVkVfQUxMIiwiVEVOQU5UX1BST0RVQ1RfUkVUUklFVkUiLCJURU5BTlRfQUNDT1VOVF9ERUxFVEUiLCJURU5BTlRfQUxMSUFOQ0VfUkVUUklFVkUiLCJURU5BTlRfUExBVEZPUk1fVVBEQVRFIiwiVEVOQU5UX0lOVklUQVRJT05fQ1JFQVRFIiwiVEVOQU5UX1JBVEVDQVJEX1JFVFJJRVZFIiwiVEVOQU5UX09SR0FOSVpBVElPTl9VUERBVEUiLCJURU5BTlRfUkFURUNBUkRfVVBEQVRFIiwiVEVOQU5UX09SR0FOSVpBVElPTl9ERUxFVEUiLCJURU5BTlRfUExBVEZPUk1fREVMRVRFIiwiVEVOQU5UX1BST0RVQ1RfQ1JFQVRFIiwiVEVOQU5UX1NVQl9BTExJQU5DRV9DUkVBVEUiLCJURU5BTlRfUkFURUNBUkRfREVMRVRFIiwiVEVOQU5UX0FDQ09VTlRfQ1JFQVRFIiwiVEVOQU5UX0lOVklUQVRJT05fVVBEQVRFIiwiVEVOQU5UX1JFVFJJRVZFX1RFTkFOVCIsIlRFTkFOVF9JTlZJVEFUSU9OX0RFTEVURSIsIlRFTkFOVF9ORUdPVElBVElPTl9BUFBST1ZBTF9TVVBFUl9BRE1JTiIsIlRFTkFOVF9ORUdPVElBVElPTl9SRVRSSUVWRV9BTExfQkVUV0VFTl9CVVlFUl9TRUxMRVIiLCJURU5BTlRfUkVUUklFVkVfVEVOQU5UX0FMTCIsIlRFTkFOVF9ORUdPVElBVElPTl9DUkVBVEUiLCJURU5BTlRfVVBEQVRFX1RFTkFOVCIsIlRFTkFOVF9PUkdBTklaQVRJT05fUkVUUklFVkUiLCJURU5BTlRfQUNDT1VOVF9VUERBVEUiLCJURU5BTlRfUFJPRFVDVF9ERUxFVEUiLCJURU5BTlRfU1VCX0FMTElBTkNFX0RFTEVURSIsIlRFTkFOVF9ORUdPVElBVElPTl9BUFBST1ZBTF9TRUxMRVIiLCJURU5BTlRfU1VCX0FMTElBTkNFX1VQREFURSIsIlRFTkFOVF9QUk9EVUNUX1VQREFURSIsIlRFTkFOVF9BQ0NPVU5UX1JFVFJJRVZFIiwiVEVOQU5UX05FR09USUFUSU9OX1JFVFJJRVZFIiwiVEVOQU5UX0RFTEVURV9URU5BTlQiXX0sIlBBU0NBTF9JTlRFTExJR0VOQ0VfbW9iaXVzIjp7InJvbGVzIjpbIlRFTkFOVF9DT05URVhUX0RFTEVURSIsIlRFTkFOVF9DT05URVhUX1VQREFURSIsIlRFTkFOVF9CSUdRVUVSWV9SRVRSSUVWRSIsIlRFTkFOVF9CSUdRVUVSWV9VUERBVEUiLCJURU5BTlRfVU5JVkVSU0VfQ1JFQVRFIiwiVEVOQU5UX0NPSE9SVFNfUkVUUklFVkUiLCJURU5BTlRfQklHUVVFUllfQ1JFQVRFIiwiVEVOQU5UX0JJR1FVRVJZX0VWQUxVQVRFIiwiVEVOQU5UX0NPSE9SVFNfVVBEQVRFIiwiVEVOQU5UX0NPTlRFWFRfUkVUUklFVkUiLCJURU5BTlRfVU5JVkVSU0VfUkVUUklFVkUiLCJURU5BTlRfQ09IT1JUU19FVkFMVUFURSIsIlRFTkFOVF9CSUdRVUVSWV9ERUxFVEUiLCJURU5BTlRfU0NIRU1BX0NSRUFURSIsIlBBU0NBTF9JTlRFTExJR0VOQ0VfVVNFUiIsIlRFTkFOVF9DT0hPUlRTX0NSRUFURSIsIlRFTkFOVF9TQ0hFTUFfVVBEQVRFIiwiVEVOQU5UX0NPTlRFWFRfQVBQUk9WRSIsIlRFTkFOVF9VTklWRVJTRV9VUERBVEUiLCJURU5BTlRfQklHUVVFUllfQVBQUk9WRSIsIlRFTkFOVF9TQ0hFTUFfUkVUUklFVkUiLCJURU5BTlRfU0NIRU1BX0FQUFJPVkUiLCJURU5BTlRfQ09OVEVYVF9FVkFMVUFURSIsIlRFTkFOVF9TQ0hFTUFfREVMRVRFIiwiVEVOQU5UX0NPTlRFWFRfQ1JFQVRFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX1JFVFJJRVZFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX0RFTEVURSIsIlRFTkFOVF9TQ0hFTUFfREFUQV9VUERBVEUiLCJURU5BTlRfQ09IT1JUU19ERUxFVEUiLCJURU5BTlRfVU5JVkVSU0VfREVMRVRFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX0lOR0VTVElPTiIsIlRFTkFOVF9DT0hPUlRTX0FQUFJPVkUiXX0sIk1PTkVUX21vYml1cyI6eyJyb2xlcyI6WyJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9BTExfUFVCTElTSEVEIiwiVEVOQU5UX1dJREdFVFNfUkVUUklFVkVfRFJBRlQiLCJURU5BTlRfV0lER0VUU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19ERUxFVEUiLCJURU5BTlRfV0lER0VUU19ERUxFVEUiLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX1JFVFJJRVZFX0FMTCIsIlRFTkFOVF9BUFBMRVRTX1VQREFURV9ISVNUT1JZIiwiVEVOQU5UX0VYUEVSSUVOQ0VTX0NSRUFURSIsIlRFTkFOVF9XSURHRVRTX1JFVFJJRVZFX1BVQkxJU0giLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX0NSRUFURSIsIlRFTkFOVF9ERVBFTkRFTkNZX1JFVFJJRVZFX0FMTCIsIlRFTkFOVF9XSURHRVRTX1BVQkxJU0giLCJURU5BTlRfUkVUUklFVkVfRklFTERTX09GX1BVQkxJU0hFRCIsIlRFTkFOVF9FWFBFUklFTkNFU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19DUkVBVEUiLCJURU5BTlRfUExVR0lOU19ERUxFVEUiLCJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9BTExfRFJBRlQiLCJURU5BTlRfUExVR0lOU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9GUk9NX0hJU1RPUlkiLCJURU5BTlRfUkVUUklFVkVfRklFTERTX09GX0RSQUZUIiwiVEVOQU5UX0FQUExFVFNfUkVUUklFVkVfUFVCTElTSEVEIiwiVEVOQU5UX0FQUExFVFNfUFVCTElTSEVEIiwiVEVOQU5UX1dJREdFVFNfUkVUUklFVkUiLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX0RFTEVURSIsIlRFTkFOVF9QTFVHSU5fQ0FURUdPUllfUFVCTElTSCIsIlRFTkFOVF9QTFVHSU5fQ0FURUdPUllfREVMRVRFX0NPTVBfRlJPTSIsIlRFTkFOVF9SRVRSSUVWRV9QVUJMSVNIRUQiLCJURU5BTlRfUExVR0lOU19SRVRSSUVWRSIsIlRFTkFOVF9QTFVHSU5TX0NSRUFURSIsIlRFTkFOVF9XSURHRVRTX0NSRUFURSIsIlRFTkFOVF9BUFBMRVRTX1JFVFJJRVZFX0FMTF9QVUJMSVNIRURfVEhVTUJOQUlMUyIsIk1PTkVUX1VTRVIiLCJURU5BTlRfRVhQRVJJRU5DRVNfREVMRVRFIiwiVEVOQU5UX0VYUEVSSUVOQ0VTX1BVQkxJU0giLCJURU5BTlRfUkVUUklFVkVfRlJPTV9ISVNUT1JZIiwiVEVOQU5UX1BMVUdJTl9DQVRFR09SWV9VUERBVEUiXX0sIlZJTkNJX21vYml1cyI6eyJyb2xlcyI6WyJWSU5DSV9VU0VSIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJyZXF1ZXN0ZXJUeXBlIjoiVEVOQU5UIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJBaWR0YWFzIEFpZHRhYXMiLCJ0ZW5hbnRJZCI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInByZWZlcnJlZF91c2VybmFtZSI6InBhc3N3b3JkX3RlbmFudF9haWR0YWFzQGdhaWFuc29sdXRpb25zLmNvbSIsImdpdmVuX25hbWUiOiJBaWR0YWFzIiwiZmFtaWx5X25hbWUiOiJBaWR0YWFzIiwiZW1haWwiOiJwYXNzd29yZF90ZW5hbnRfYWlkdGFhc0BnYWlhbnNvbHV0aW9ucy5jb20ifQ.kKu9YJ7T-pembGjO9YKE0eAExS250Fp1Tnfs9uG8aXiekU31xlKf0Uf9XCikv_D_R5wEv3GE7uU3S7t9UVRYydzX54Ccr3w1Q-BRkX2XLiOfZmnpJ7xKVUYYX-6S42Of16Exub06aXEKPJgF3aFp3pnDqNQUc9Rtr-xzckCK5t4ruZ4NKprAq5at4Wg5pjMpIMlVQ9c3mcB6zhgb6xZjGGy29Q6-PrqYB1k48tX1LQSPmqaVsDyUc4I1841IPmfu042vhOiRecBNwqDGJEaW2z7LgnD8tfaevjDXdKMdBDy2kBTrMVWAekFAuzmDd-_hbCEAKxN0_43MeHsvFBILSg"

    # Headers including Bearer token
    headers = {
        'Authorization': f'Bearer {bearer_token}',
    }

    # Create the file part of the multipart body
    multipart_data = MultipartEncoder(
        fields={
            'file': (file_name, BytesIO(content.encode('utf-8')), 'application/octet-stream'),
        }
    )

    # Update the headers with the correct multipart content type
    headers['Content-Type'] = multipart_data.content_type

    try:
        response = requests.post(url, data=multipart_data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
        logging.info(f"------- Response body:  {response.text} --------")
        return json.loads(json.dumps(response.json())).get("id")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        raise ApiException("Failed to make API call")
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure")


def import_ontology(content_id):

    url = 'https://ig.aidtaas.com/mobius-graph-operations/graphrag/v2.0/initialize'
    bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3Ny1NUVdFRTNHZE5adGlsWU5IYmpsa2dVSkpaWUJWVmN1UmFZdHl5ejFjIn0.eyJleHAiOjE3NDA2OTA3MDcsImlhdCI6MTc0MDY1NDcwNywianRpIjoiY2Y0NzRlMzMtZTIwMy00MDdiLWI0MTItZDk3OWQyODAzNGNmIiwiaXNzIjoiaHR0cDovL2tleWNsb2FrLXNlcnZpY2Uua2V5Y2xvYWsuc3ZjLmNsdXN0ZXIubG9jYWw6ODA4MC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIkJPTFRaTUFOTl9CT1RfbW9iaXVzIiwiUEFTQ0FMX0lOVEVMTElHRU5DRV9tb2JpdXMiLCJNT05FVF9tb2JpdXMiLCJWSU5DSV9tb2JpdXMiLCJhY2NvdW50Il0sInN1YiI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInR5cCI6IkJlYXJlciIsImF6cCI6IkhPTEFDUkFDWV9tb2JpdXMiLCJzaWQiOiIzZjg0Mjg1NS05MTYxLTRjY2UtYWQ0Yi01YjlhYzU5NmZlOTMiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIkhDWV9URU5BTlRfQ1VTVE9NIiwib2ZmbGluZV9hY2Nlc3MiLCJCT0JfVEVOQU5UX0NVU1RPTSIsInVtYV9hdXRob3JpemF0aW9uIiwiUElfVEVOQU5UX0NVU1RPTSIsIk1PTkVUX1RFTkFOVF9DVVNUT00iXX0sInJlc291cmNlX2FjY2VzcyI6eyJCT0xUWk1BTk5fQk9UX21vYml1cyI6eyJyb2xlcyI6WyJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9QUk9DRVNTX0RFRklOSVRJT04iLCJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9JTlNUQU5DRSIsIlRFTkFOVF9XT1JLRkxPV19SRVRSSUVWRV9BTEwiLCJURU5BTlRfV09SS0ZMT1dfQUNUSVZBVEVfSU5TVEFOQ0UiLCJURU5BTlRfV09SS0ZMT1dfRVhFQ1VURSIsIlRFTkFOVF9XT1JLRkxPV19SRVRSSUVWRV9IT01FIiwiVEVOQU5UX1dPUktGTE9XX1JFVFJJRVZFX1NVU1BFTkRFRF9QUk9DRVNTX0RFRlMiLCJURU5BTlRfV09SS0ZMT1dfQUNUSVZBVEVfUlVOX0pPQiIsIlRFTkFOVF9XT1JLRkxPV19TVEFUVVMiLCJURU5BTlRfV09SS0ZMT1dfQ1JFQVRFIiwiVEVOQU5UX1dPUktGTE9XX0FDVElWSVRZX0RBVEFfQ09VTlQiLCJCT0xUWk1BTk5fQk9UX1VTRVIiLCJURU5BTlRfV09SS0ZMT1dfREVQTE9ZX1ZFUlNJT04iLCJURU5BTlRfV09SS0ZMT1dfSU1QT1JUIiwiVEVOQU5UX1dPUktGTE9XX1VQREFURSIsIlRFTkFOVF9XT1JLRkxPV19ERVBMT1kiLCJURU5BTlRfV09SS0ZMT1dfU1VTUEVORF9SVU5fSk9CIiwiVEVOQU5UX1dPUktGTE9XX1JFVFJJRVZFIiwiVEVOQU5UX1dPUktGTE9XX0RFTEVURSIsIlRFTkFOVF9XT1JLRkxPV19UUklHR0VSX01FU1NBR0VfTElTVEVORVIiXX0sIkhPTEFDUkFDWV9tb2JpdXMiOnsicm9sZXMiOlsiVEVOQU5UX1BST0RVQ1RfUkVUUklFVkVfTUFTVEVSX0NPTkZJRyIsIlRFTkFOVF9JTlZJVEFUSU9OX1JFU1BPTkQiLCJURU5BTlRfTkVHT1RJQVRJT05fQVBQUk9WQUxfQlVZRVIiLCJURU5BTlRfSU5WSVRBVElPTl9SRVRSSUVWRSIsIlRFTkFOVF9PUkdBTklaQVRJT05fQ1JFQVRFIiwiVEVOQU5UX0NSRUFURV9URU5BTlQiLCJURU5BTlRfU1VCX0FMTElBTkNFX1JFVFJJRVZFIiwiVEVOQU5UX1BMQVRGT1JNX0NSRUFURSIsIkhPTEFDUkFDWV9VU0VSIiwiVEVOQU5UX1JBVEVDQVJEX0NSRUFURSIsIlRFTkFOVF9QTEFURk9STV9SRVRSSUVWRSIsIlRFTkFOVF9SRVRSSUVWRV9CWV9CVVlFUl9QTEFURk9STSIsIlRFTkFOVF9TVUJfQUxMSUFOQ0VfUkVUUklFVkVfQUxMIiwiVEVOQU5UX1BST0RVQ1RfUkVUUklFVkUiLCJURU5BTlRfQUNDT1VOVF9ERUxFVEUiLCJURU5BTlRfQUxMSUFOQ0VfUkVUUklFVkUiLCJURU5BTlRfUExBVEZPUk1fVVBEQVRFIiwiVEVOQU5UX0lOVklUQVRJT05fQ1JFQVRFIiwiVEVOQU5UX1JBVEVDQVJEX1JFVFJJRVZFIiwiVEVOQU5UX09SR0FOSVpBVElPTl9VUERBVEUiLCJURU5BTlRfUkFURUNBUkRfVVBEQVRFIiwiVEVOQU5UX09SR0FOSVpBVElPTl9ERUxFVEUiLCJURU5BTlRfUExBVEZPUk1fREVMRVRFIiwiVEVOQU5UX1BST0RVQ1RfQ1JFQVRFIiwiVEVOQU5UX1NVQl9BTExJQU5DRV9DUkVBVEUiLCJURU5BTlRfUkFURUNBUkRfREVMRVRFIiwiVEVOQU5UX0FDQ09VTlRfQ1JFQVRFIiwiVEVOQU5UX0lOVklUQVRJT05fVVBEQVRFIiwiVEVOQU5UX1JFVFJJRVZFX1RFTkFOVCIsIlRFTkFOVF9JTlZJVEFUSU9OX0RFTEVURSIsIlRFTkFOVF9ORUdPVElBVElPTl9BUFBST1ZBTF9TVVBFUl9BRE1JTiIsIlRFTkFOVF9ORUdPVElBVElPTl9SRVRSSUVWRV9BTExfQkVUV0VFTl9CVVlFUl9TRUxMRVIiLCJURU5BTlRfUkVUUklFVkVfVEVOQU5UX0FMTCIsIlRFTkFOVF9ORUdPVElBVElPTl9DUkVBVEUiLCJURU5BTlRfVVBEQVRFX1RFTkFOVCIsIlRFTkFOVF9PUkdBTklaQVRJT05fUkVUUklFVkUiLCJURU5BTlRfQUNDT1VOVF9VUERBVEUiLCJURU5BTlRfUFJPRFVDVF9ERUxFVEUiLCJURU5BTlRfU1VCX0FMTElBTkNFX0RFTEVURSIsIlRFTkFOVF9ORUdPVElBVElPTl9BUFBST1ZBTF9TRUxMRVIiLCJURU5BTlRfU1VCX0FMTElBTkNFX1VQREFURSIsIlRFTkFOVF9QUk9EVUNUX1VQREFURSIsIlRFTkFOVF9BQ0NPVU5UX1JFVFJJRVZFIiwiVEVOQU5UX05FR09USUFUSU9OX1JFVFJJRVZFIiwiVEVOQU5UX0RFTEVURV9URU5BTlQiXX0sIlBBU0NBTF9JTlRFTExJR0VOQ0VfbW9iaXVzIjp7InJvbGVzIjpbIlRFTkFOVF9DT05URVhUX0RFTEVURSIsIlRFTkFOVF9DT05URVhUX1VQREFURSIsIlRFTkFOVF9CSUdRVUVSWV9SRVRSSUVWRSIsIlRFTkFOVF9CSUdRVUVSWV9VUERBVEUiLCJURU5BTlRfVU5JVkVSU0VfQ1JFQVRFIiwiVEVOQU5UX0NPSE9SVFNfUkVUUklFVkUiLCJURU5BTlRfQklHUVVFUllfQ1JFQVRFIiwiVEVOQU5UX0JJR1FVRVJZX0VWQUxVQVRFIiwiVEVOQU5UX0NPSE9SVFNfVVBEQVRFIiwiVEVOQU5UX0NPTlRFWFRfUkVUUklFVkUiLCJURU5BTlRfVU5JVkVSU0VfUkVUUklFVkUiLCJURU5BTlRfQ09IT1JUU19FVkFMVUFURSIsIlRFTkFOVF9CSUdRVUVSWV9ERUxFVEUiLCJURU5BTlRfU0NIRU1BX0NSRUFURSIsIlBBU0NBTF9JTlRFTExJR0VOQ0VfVVNFUiIsIlRFTkFOVF9DT0hPUlRTX0NSRUFURSIsIlRFTkFOVF9TQ0hFTUFfVVBEQVRFIiwiVEVOQU5UX0NPTlRFWFRfQVBQUk9WRSIsIlRFTkFOVF9VTklWRVJTRV9VUERBVEUiLCJURU5BTlRfQklHUVVFUllfQVBQUk9WRSIsIlRFTkFOVF9TQ0hFTUFfUkVUUklFVkUiLCJURU5BTlRfU0NIRU1BX0FQUFJPVkUiLCJURU5BTlRfQ09OVEVYVF9FVkFMVUFURSIsIlRFTkFOVF9TQ0hFTUFfREVMRVRFIiwiVEVOQU5UX0NPTlRFWFRfQ1JFQVRFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX1JFVFJJRVZFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX0RFTEVURSIsIlRFTkFOVF9TQ0hFTUFfREFUQV9VUERBVEUiLCJURU5BTlRfQ09IT1JUU19ERUxFVEUiLCJURU5BTlRfVU5JVkVSU0VfREVMRVRFIiwiVEVOQU5UX1NDSEVNQV9EQVRBX0lOR0VTVElPTiIsIlRFTkFOVF9DT0hPUlRTX0FQUFJPVkUiXX0sIk1PTkVUX21vYml1cyI6eyJyb2xlcyI6WyJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9BTExfUFVCTElTSEVEIiwiVEVOQU5UX1dJREdFVFNfUkVUUklFVkVfRFJBRlQiLCJURU5BTlRfV0lER0VUU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19ERUxFVEUiLCJURU5BTlRfV0lER0VUU19ERUxFVEUiLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX1JFVFJJRVZFX0FMTCIsIlRFTkFOVF9BUFBMRVRTX1VQREFURV9ISVNUT1JZIiwiVEVOQU5UX0VYUEVSSUVOQ0VTX0NSRUFURSIsIlRFTkFOVF9XSURHRVRTX1JFVFJJRVZFX1BVQkxJU0giLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX0NSRUFURSIsIlRFTkFOVF9ERVBFTkRFTkNZX1JFVFJJRVZFX0FMTCIsIlRFTkFOVF9XSURHRVRTX1BVQkxJU0giLCJURU5BTlRfUkVUUklFVkVfRklFTERTX09GX1BVQkxJU0hFRCIsIlRFTkFOVF9FWFBFUklFTkNFU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19DUkVBVEUiLCJURU5BTlRfUExVR0lOU19ERUxFVEUiLCJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9BTExfRFJBRlQiLCJURU5BTlRfUExVR0lOU19VUERBVEUiLCJURU5BTlRfQVBQTEVUU19SRVRSSUVWRV9GUk9NX0hJU1RPUlkiLCJURU5BTlRfUkVUUklFVkVfRklFTERTX09GX0RSQUZUIiwiVEVOQU5UX0FQUExFVFNfUkVUUklFVkVfUFVCTElTSEVEIiwiVEVOQU5UX0FQUExFVFNfUFVCTElTSEVEIiwiVEVOQU5UX1dJREdFVFNfUkVUUklFVkUiLCJURU5BTlRfUExVR0lOX0NBVEVHT1JZX0RFTEVURSIsIlRFTkFOVF9QTFVHSU5fQ0FURUdPUllfUFVCTElTSCIsIlRFTkFOVF9QTFVHSU5fQ0FURUdPUllfREVMRVRFX0NPTVBfRlJPTSIsIlRFTkFOVF9SRVRSSUVWRV9QVUJMSVNIRUQiLCJURU5BTlRfUExVR0lOU19SRVRSSUVWRSIsIlRFTkFOVF9QTFVHSU5TX0NSRUFURSIsIlRFTkFOVF9XSURHRVRTX0NSRUFURSIsIlRFTkFOVF9BUFBMRVRTX1JFVFJJRVZFX0FMTF9QVUJMSVNIRURfVEhVTUJOQUlMUyIsIk1PTkVUX1VTRVIiLCJURU5BTlRfRVhQRVJJRU5DRVNfREVMRVRFIiwiVEVOQU5UX0VYUEVSSUVOQ0VTX1BVQkxJU0giLCJURU5BTlRfUkVUUklFVkVfRlJPTV9ISVNUT1JZIiwiVEVOQU5UX1BMVUdJTl9DQVRFR09SWV9VUERBVEUiXX0sIlZJTkNJX21vYml1cyI6eyJyb2xlcyI6WyJWSU5DSV9VU0VSIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJyZXF1ZXN0ZXJUeXBlIjoiVEVOQU5UIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJBaWR0YWFzIEFpZHRhYXMiLCJ0ZW5hbnRJZCI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInByZWZlcnJlZF91c2VybmFtZSI6InBhc3N3b3JkX3RlbmFudF9haWR0YWFzQGdhaWFuc29sdXRpb25zLmNvbSIsImdpdmVuX25hbWUiOiJBaWR0YWFzIiwiZmFtaWx5X25hbWUiOiJBaWR0YWFzIiwiZW1haWwiOiJwYXNzd29yZF90ZW5hbnRfYWlkdGFhc0BnYWlhbnNvbHV0aW9ucy5jb20ifQ.kKu9YJ7T-pembGjO9YKE0eAExS250Fp1Tnfs9uG8aXiekU31xlKf0Uf9XCikv_D_R5wEv3GE7uU3S7t9UVRYydzX54Ccr3w1Q-BRkX2XLiOfZmnpJ7xKVUYYX-6S42Of16Exub06aXEKPJgF3aFp3pnDqNQUc9Rtr-xzckCK5t4ruZ4NKprAq5at4Wg5pjMpIMlVQ9c3mcB6zhgb6xZjGGy29Q6-PrqYB1k48tX1LQSPmqaVsDyUc4I1841IPmfu042vhOiRecBNwqDGJEaW2z7LgnD8tfaevjDXdKMdBDy2kBTrMVWAekFAuzmDd-_hbCEAKxN0_43MeHsvFBILSg"
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    data = {
        "content_id": content_id,
        "tenant_id": "2cf76e5f-26ad-4f2c-bccc-f4bc1e7bfb64",
        "ontology_id": "67c0564b7afbaa5f5a6957f6",
        "transaction_id": "example_transaction_id",
        "ontology_name": "test123",
        "is_pdf": False
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises HTTPError for bad responses
        logging.info(f"------- Response body:  {response.text} --------")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        raise ApiException("Failed to make API call")
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure")


def main():
    """Command-line interface for the package analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <package_path> [output_file]")
        print("\nSupported package types:")
        print("  - Python source directories")
        print("  - Wheel packages (.whl)")
        print("  - ZIP archives (.zip)")
        print("  - TAR archives (.tar, .tar.gz, .tgz)")
        print("  - GZIP files (.gz)")
        print("  - Git repositories (HTTPS/SSH URLs)")
        print("\nOptions:")
        print("  package_path: Path to package or Git URL")
        print("  output_file: Optional JSON report path")
        print("\nExamples:")
        print("  python analyzer.py ./my_package")
        print("  python analyzer.py package-1.0.whl")
        print("  python analyzer.py https://github.com/user/repo")
        sys.exit(1)

    package_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        report = analyze_package(package_path, output_file)

        # content_id = upload_to_content_service(report)
        # logging.info(f"------- content_id:  {content_id} --------")
        # import_ontology(content_id)

        print(report)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
