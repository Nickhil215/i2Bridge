import os
import re
import shutil
import uuid
import sys
import importlib.util
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, List, Set, Tuple

import astroid
from astroid import nodes
import pkg_resources

from app import logger
from app.models.data_classes import ClassInfo, ModuleInfo, FunctionInfo, TestsInfo, ApiInfo
from app.services.rdf_exporter import RDFExporter


def extract_repo_name(github_url):
    # Use regular expression to extract the repo name from the URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if match:
        return match.group(2).replace('.git', '')  # The second group is the repo name
    return None

def get_pypi_name(repo_name):
    """
    Convert GitHub repo name to standard Python package name.
    Attempt to map a GitHub repo name to its PyPI package name.
    This is a heuristic that tries common patterns.
    """
    # Common Python package renaming patterns
    if repo_name.startswith('python-'):
        # python-package -> package
        possible_name = repo_name[7:]
    elif '-python-' in repo_name:
        # lib-python-client -> lib
        possible_name = repo_name.split('-python-')[0]
    elif repo_name.endswith('-python'):
        # package-python -> package
        possible_name = repo_name[:-7]
    elif repo_name.endswith('-py'):
        # package-py -> package
        possible_name = repo_name[:-3]
    else:
        # Default case - try the repo name directly
        possible_name = repo_name

    # Replace hyphens with underscores for Python import compatibility
    import_name = possible_name.replace('-', '_')

    return {
        'repo_name': repo_name,
        'possible_pypi_name': possible_name,
        'import_name': import_name,
        'display_name': repo_name  # Original name for display purposes
    }


def format_path(path):
    # Split the path by "/"
    split_path = path.split('/')

    # Remove the first element ("..")
    split_path = split_path[1:]

    # Join the remaining parts with "."
    joined_path = '.'.join(split_path)

    # Remove the ".py" extension using slicing
    joined_path = joined_path[:-3]  # Removes last 3 characters (".py")

    return joined_path


def _extract_api_endpoint_info(node: nodes.FunctionDef, source: str) -> ApiInfo | None:
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
        module_path="",
        path=endpoint_info.get('path'),
        http_method=endpoint_info.get('method', None),
        path_params=path_params if request_params else None,
        request_params=request_params if request_params else None,
        request_body=request_body if request_body else None,
        response_body=response_body if response_body else None,
        description="",
        description_embedding="",
        is_active=True,
        is_async=isinstance(node, nodes.AsyncFunctionDef),
        runtime="python"
    )


def _extract_class_info(node: nodes.ClassDef, source: str) -> ClassInfo:
    """Extract detailed information about a class."""
    functions = []
    attributes = []

    for child in node.get_children():
        if (isinstance(child, nodes.FunctionDef)
                and not child.name.startswith("_")):
            functions.append(child.name)
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
        functions=functions,
        attributes=attributes,
        docstring=docstring,
        decorators=decorators,  # Use the properly extracted decorators
        start_line=node.lineno,
        end_line=node.end_lineno or node.lineno,
        description="",
        description_embedding=""
    )


def _calculate_complexity(node: nodes.NodeNG) -> int:
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


class BaseAnalyzer(ABC):
    """Abstract base class defining the interface for package analyzers."""

    def __init__(self, package_path: str):
        """Initialize the base analyzer."""
        self.package_path = os.path.abspath(package_path)
        self.modules: Dict[str, nodes.Module] = {}
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.classes: Dict[str, ClassInfo] = {}
        self.module_info: Dict[str, ModuleInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.tests: Dict[str, TestsInfo] = {}
        self.apis: Dict[str, ApiInfo] = {}
        self.errors: List[tuple] = []
        self.temp_dir: Optional[str] = None
        # New: cache for stdlib packages
        self.stdlib_cache = {}

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

    def _analyze_file(self, temp_file_path: str, requirement_info: Set[str] = None,
                      git_url: str = None, branch: str = None) -> None:
        """Analyze a single Python file and extract its AST information."""
        try:
            file_path = os.path.relpath(temp_file_path, self.package_path)
            self.relative_path = file_path

            self.requirement_info = requirement_info
            self.git_url = git_url
            self.branch = branch

            self.package_name = None
            self.package_info = None

            if git_url:
                repo_name = extract_repo_name(git_url)
                self.package_info = get_pypi_name(repo_name)
                self.package_name = self.package_info['import_name']  # Use import-compatible name
                self.display_name = self.package_info['display_name']  # Original name for display

            # Handle module path formatting properly
            path_parts = format_path(self.relative_path).split('.')

            # Special handling for setup.py at the root level
            if path_parts and path_parts[0] == 'setup':
                # For setup.py, use packagename.setup.function_name format
                self.formatted_path = f"{self.package_name}.setup"
            else:
                # For normal modules, construct path properly with dots
                try:
                    # Try to find the package name in the path parts
                    if self.package_name and self.package_name in path_parts:
                        # Find where the package name is in the path
                        pkg_index = path_parts.index(self.package_name)
                        # Join from that point forward
                        self.formatted_path = '.'.join(path_parts[pkg_index:])
                    else:
                        # If package name not in path, assume format is package_name.relative_path
                        self.formatted_path = f"{self.package_name}.{'.'.join(path_parts)}"
                except (ValueError, IndexError):
                    # Fallback: just use package_name.relative_path
                    self.formatted_path = f"{self.package_name}.{'.'.join(path_parts)}"

            with open(temp_file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the file using astroid
            module = astroid.parse(source, path=temp_file_path)
            self.modules[file_path] = module

            self._analyze_modules(module, file_path, source)
            self._analyze_classes(module, file_path, source)

            # Pass source to analysis methods
            file_path_parts = file_path.split("/")
            filename = file_path.split('/')[-1]

            # List of terms to check in file path
            terms_to_check = ["tests", "test", "example", "examples", "sample"]

            # Check if any of the conditions fail
            if any(term not in file_path_parts for term in terms_to_check) and "_" not in filename:
              logger.info(f"Analyzing file: {file_path}")
              self._analyze_functions(module, file_path, source)
              self._analyze_api_endpoint(module, file_path, source)
        except Exception as e:
            logger.error(f"Error analyzing {temp_file_path}: {str(e)}")
            self.errors.append((temp_file_path, str(e)))

    def _analyze_functions(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all functions in a module, including methods."""
        for node in module.body:
            if isinstance(node, nodes.ClassDef):
                # If it's a class, find all functions inside it
                for class_node in node.body:
                    if isinstance(class_node, nodes.FunctionDef):
                        func_info = self._extract_function_info(class_node, source, file_path)
                        self.functions[f"{file_path}::{func_info.function_name}"] = func_info
            elif isinstance(node, nodes.FunctionDef):
                # If it's a function at the module level
                func_info = self._extract_function_info(node, source, file_path)
                self.functions[f"{file_path}::{func_info.function_name}"] = func_info


    def _analyze_api_endpoint(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all api endpoints in a module"""
        for node in module.body:
            if isinstance(node, nodes.ClassDef):
                # If it's a class, find all functions inside it
                for class_node in node.body:
                    if isinstance(class_node, nodes.FunctionDef):
                        api_info = _extract_api_endpoint_info(class_node, source)
                        if api_info and api_info.http_method is not None:
                            self.apis[f"{file_path}::{api_info.name}"] = api_info
            elif isinstance(node, nodes.FunctionDef):
                # If it's a function at the module level
                api_info = _extract_api_endpoint_info(node, source)
                if api_info and api_info.http_method is not None:
                    self.apis[f"{file_path}::{api_info.name}"] = api_info

    def _analyze_tests(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all tests in a module"""
        for node in module.body:
            if isinstance(node, nodes.ClassDef):
                # If it's a class, find all functions inside it
                for class_node in node.body:
                    if isinstance(class_node, nodes.FunctionDef):
                        test_info = self._extract_tests_info(class_node, source)
                        self.tests[f"{file_path}::{test_info.name}"] = test_info
            elif isinstance(node, nodes.FunctionDef):
                # If it's a function at the module level
                test_info = self._extract_tests_info(node, source)
                self.tests[f"{file_path}::{test_info.name}"] = test_info

    def _analyze_classes(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all classes in a module."""
        for node in module.nodes_of_class(nodes.ClassDef):
            class_info = _extract_class_info(node, source)
            self.classes[f"{file_path}::{class_info.name}"] = class_info

    def _analyze_modules(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Extract and track all imports in a module, and create CoreComponentInfo"""

        self.packages = {}  # {str: List[str]}
        self.imports = {}   # {str: List[str]}

        for node in module.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if file_path not in self.packages:
                self.packages[file_path] = []
            if file_path not in self.imports:
                self.imports[file_path] = []
            if isinstance(node, nodes.Import):
                for name, asname in node.names:  # Unpacking (name, alias) tuples
                    import_name = name if asname is None else f"{name} as {asname}"
                    package_name = name.split('.')[0]

                    # Get the correct library name dynamically
                    library_name = self.get_library_name(package_name)

                    if package_name not in self.packages[file_path]:
                        self.packages[file_path].append((library_name, import_name))  # Add library name and import name

                    self.imports[file_path].append(f"import {import_name}")
            elif isinstance(node, nodes.ImportFrom):
                module_name = node.modname
                for name, asname in node.names:  # Unpacking (name, alias) tuples
                    if not module_name:
                        import_name = f"{name}" if asname is None else f"{name} as {asname}"
                        self.imports[file_path].append(f"from {self.formatted_path} import {import_name}")
                    else:
                        import_name = f"{name}" if asname is None else f"{name} as {asname}"
                        package_name = module_name.split('.')[0]
                        library_name = self.get_library_name(package_name)  # Get the correct library name dynamically

                        if package_name not in self.packages[file_path]:
                            self.packages[file_path].append((library_name, module_name))  # Add library name and import name
                        self.imports[file_path].append(f"from {module_name} import {import_name}")

        self.module_info[file_path] = ModuleInfo(
            name=file_path.split('/')[-1],
            description="",
            description_embedding="",
            classes={},
            functions={})

    # NEW METHODS FOR IMPROVED ANALYSIS

    def _is_stdlib_package(self, package_name: str) -> bool:
        """Check if a package is part of the Python standard library."""
        # Check cache first
        if package_name in self.stdlib_cache:
            return self.stdlib_cache[package_name]

        # Check built-in modules
        if package_name in sys.builtin_module_names:
            self.stdlib_cache[package_name] = True
            return True

        try:
            # Try to find the package spec
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                self.stdlib_cache[package_name] = False
                return False

            # Most stdlib modules are in the Python installation directory
            # and not in site-packages or dist-packages
            path = spec.origin
            if path and ('site-packages' not in path and 'dist-packages' not in path):
                self.stdlib_cache[package_name] = True
                return True
        except (ImportError, AttributeError):
            pass

        self.stdlib_cache[package_name] = False
        return False

    def _extract_imported_name(self, import_stmt: str) -> str:
        """Extract the imported name from an import statement."""
        if ' as ' in import_stmt:
            # Handle aliased imports
            return import_stmt.split(' as ')[1].strip()
        elif 'from ' in import_stmt and ' import ' in import_stmt:
            # Handle from ... import ...
            parts = import_stmt.split(' import ')
            if len(parts) > 1:
                return parts[1].strip()
        elif 'import ' in import_stmt:
            # Handle plain imports
            parts = import_stmt.split('import ')
            if len(parts) > 1:
                return parts[1].strip()

        # Default case
        return ""

    def classify_dependencies(self, packages: Set[str]) -> Tuple[List[str], List[str]]:
        """Classify packages as runtime or development dependencies based on usage patterns."""
        runtime_deps = []
        dev_deps = []

        dev_patterns = ['test', 'dev', 'doc', 'debug', 'mock', 'build', 'setup', 'lint', 'coverage']

        for package_info in packages:
            # Extract base package name without version constraints
            package_name = package_info.split('==')[0].split('>=')[0].split('~=')[0].strip()

            # Check for common development dependency patterns
            if any(pattern in package_name.lower() for pattern in dev_patterns):
                dev_deps.append(package_info)
            # Check version constraints that suggest dev dependencies
            elif '; python_version' in package_info or 'dev' in package_info:
                dev_deps.append(package_info)
            else:
                runtime_deps.append(package_info)

        return runtime_deps, dev_deps

    def get_function_imports(self, node: nodes.NodeNG, file_path: str, source: str) -> List[str]:
        """Extract imports that are actually used within a function."""
        # Get all names referenced in the function
        referenced_names = set()
        for name_node in node.nodes_of_class(astroid.Name):
            referenced_names.add(name_node.name)

        # Get all imports from the file
        file_imports = {}
        if file_path in self.imports:
            for import_stmt in self.imports[file_path]:
                # Extract the imported name and its source
                imported_name = self._extract_imported_name(import_stmt)
                if imported_name:
                    file_imports[imported_name] = import_stmt

        # Find which imports are actually used
        used_imports = []
        for name in referenced_names:
            if name in file_imports:
                used_imports.append(file_imports[name])

        # Also check if the function uses any imports indirectly through method calls
        for call in node.nodes_of_class(astroid.Call):
            try:
                if isinstance(call.func, astroid.Attribute) and hasattr(call.func.expr, 'name'):
                    if call.func.expr.name in file_imports:
                        used_imports.append(file_imports[call.func.expr.name])
            except AttributeError:
                pass

        return used_imports

    def infer_package_dependencies(self, imports: Set[str]) -> List[str]:
        """Infer which packages are needed based on imports."""
        required_packages = set()

        # Always include the package itself - use PyPI name if available
        if self.package_info and 'possible_pypi_name' in self.package_info:
            # Add both the PyPI name and GitHub name to be safe
            required_packages.add(self.package_info['possible_pypi_name'])
            required_packages.add(self.package_info['repo_name'])

            # Also try common variants
            if '-' in self.package_info['repo_name']:
                # Try with underscores instead of hyphens
                required_packages.add(self.package_info['repo_name'].replace('-', '_'))
            if '_' in self.package_info['repo_name']:
                # Try with hyphens instead of underscores
                required_packages.add(self.package_info['repo_name'].replace('_', '-'))
        elif self.package_name:
            required_packages.add(self.package_name)

        for import_stmt in imports:
            # Extract the top-level package
            if import_stmt.startswith('import '):
                top_pkg = import_stmt.split()[1].split('.')[0]
            elif import_stmt.startswith('from '):
                parts = import_stmt.split()
                if len(parts) >= 2:
                    # Handle relative imports
                    module_path = parts[1]
                    if module_path.startswith('.'):
                        continue  # Skip relative imports
                    top_pkg = module_path.split('.')[0]
                else:
                    continue
            else:
                continue

            # Skip relative imports and the package itself
            if top_pkg == '.' or (self.package_name and top_pkg == self.package_name):
                continue

            # Check if this is a standard library package
            if self._is_stdlib_package(top_pkg):
                continue

            # Add the package name
            required_packages.add(top_pkg)

            # Try to map common package names to their PyPI equivalents
            # This is incomplete and would need to be expanded based on your needs
            package_map = {
                'tomlkit': 'tomlkit',
                'pytest': 'pytest',
                'requests': 'requests',
                'numpy': 'numpy',
                'pandas': 'pandas',
                # Add more mappings as needed
            }

            if top_pkg in package_map:
                required_packages.add(package_map[top_pkg])

        return list(required_packages)

    # Enhanced version for direct function use
    def get_used_imports(self, start_line, end_line, source_code, file_path):
        """
        Extract imports actually used in a specific code region.
        This is more intelligent than the original implementation.
        """
        # Extract the function code
        function_code = source_code.splitlines()[start_line-1:end_line]
        function_text = '\n'.join(function_code)
        #
        # Track imports explicitly declared within the function
        direct_imports = []
        # for line in function_code:
        #     if re.match(r'^\s*import\s+', line) or re.match(r'^\s*from\s+\S+\s+import', line):
        #         direct_imports.append(line.strip())

        # Check for imports from the module that are used in the function
        module_imports = []
        if file_path in self.imports:
            for import_stmt in self.imports[file_path]:
                # Extract what was imported
                imported_name = ""
                if ' as ' in import_stmt:
                    imported_name = import_stmt.split(' as ')[1].strip()
                elif 'from ' in import_stmt and ' import ' in import_stmt:
                    imported_name = import_stmt.split(' import ')[1].strip()
                elif 'import ' in import_stmt:
                    imported_name = import_stmt.split('import ')[1].strip()

                # Check if the imported name is used in the function
                if imported_name and re.search(r'\b' + re.escape(imported_name) + r'\b', function_text):
                    if import_stmt.startswith("from"):
                        module_imports.append(re.sub(r'^from (\w+)(\.\s*)?(\w+)(,?\s*\w+)*$', r'import \1', import_stmt))
                    else:
                        module_imports.append(import_stmt)

        # Combine explicit and module imports
        return set(module_imports)

    def _extract_function_info(self, node: nodes.FunctionDef, source: str,
                               file_path: str) -> FunctionInfo:
        """Extract detailed information about a function with improved import and dependency analysis."""
        args = [arg.name for arg in node.args.args if arg.name != 'self']  # Exclude 'self'
        returns = node.returns.as_string() if node.returns else None
        docstring = None
        if isinstance(node.doc_node, nodes.Const):
            docstring = node.doc_node.value.strip()
        complexity = _calculate_complexity(node)

        # Fix decorator handling
        decorators = []
        if node.decorators:
            for decorator in node.decorators.nodes:
                decorators.append(decorator.as_string())

        class_name = None
        if isinstance(node.parent, nodes.ClassDef):
            class_name = node.parent.name

        comments = []
        start_line=node.lineno
        end_line=node.end_lineno or node.lineno

        for line_num in range(start_line, end_line + 1):
            line = source.splitlines()[line_num - 1].strip()
            if '#' in line:
                comment_text = line.split('#', 1)[1].strip()
                comments.append(comment_text)

        formated_args = ', '.join(arg.name for arg in node.args.args if arg.name != 'self' and arg.name != '**kwargs')

        # Improved function_exe_cmd construction with proper path formatting
        if class_name:
            function_exe_cmd = f"{self.formatted_path}.{class_name}.{node.name}()"
        else:
            # Handle setup.py special case
            if file_path.endswith('setup.py') or '/setup.py' in file_path:
                function_exe_cmd = f"{self.display_name}.setup.{node.name}()"
            else:
                function_exe_cmd = f"{self.formatted_path}.{node.name}()"

        signature = f"{node.name}({formated_args})"

        # Get function-specific imports with improved detection
        imports = self.get_used_imports(start_line, end_line, source, file_path)

        # Make sure imports list includes any import for the package itself if needed
        if self.package_name and self.package_info:
            # Check if there are any imports from this package
            package_import_found = False
            for imp in imports:
                if self.package_name in imp or self.package_info['display_name'] in imp:
                    package_import_found = True
                    break

            # If function is in a module that needs to be imported, add it
            if not package_import_found and node.name != '__init__':
                imports.add(f"import {self.package_name}")

        # Determine required packages based on imports with improved logic
        required_packages = self.infer_package_dependencies(imports)

        # Filter to just runtime dependencies
        runtime_packages, _ = self.classify_dependencies(self.requirement_info) if self.requirement_info else ([], [])

        # Filter the runtime packages to only include those that match our required packages
        filtered_packages = set()
        if required_packages and runtime_packages:
            for pkg_info in runtime_packages:
                pkg_name = pkg_info.split('==')[0].split('>=')[0].split('~=')[0].strip()

                # Check if this package matches any of our required packages
                if any(req.lower() == pkg_name.lower() for req in required_packages):
                    filtered_packages.add(pkg_info)

        # If we couldn't find matching packages, use our best guess
        if not filtered_packages and self.package_info:
            # Add the package itself as a dependency
            filtered_packages.add(self.package_info['possible_pypi_name'])

            # Also add any third-party packages we detected
            for pkg in required_packages:
                if pkg != self.package_name and pkg != self.package_info['possible_pypi_name']:
                    filtered_packages.add(pkg)

        # Construct the function definition path properly
        if class_name:
            function_def = f"{self.formatted_path}.{class_name}.{signature}"
        else:
            # Special handling for setup.py
            if file_path.endswith('setup.py') or '/setup.py' in file_path:
                function_def = f"{self.display_name}.setup.{signature}"
            else:
                function_def = f"{self.formatted_path}.{signature}"

        # Create a properly formatted URL to the function source
        prefix = self.git_url.rstrip(".git") if self.git_url else ""
        path = file_path.lstrip("../")

        function_url = f"{prefix}/blob/{self.branch}/{path}#L{start_line}" if prefix and self.branch else ""

        dependencies = set()
        # Collect function dependencies
        for call in node.nodes_of_class(astroid.Call):
            try:
                inferred = next(call.func.infer())
                if isinstance(inferred, nodes.FunctionDef):
                    # Get the module path where the called function is defined
                    inferred_module_path = os.path.relpath(
                        inferred.root().file,
                        self.package_path
                    ) if inferred.root().file else file_path

                    # Build the function path as stored in self.functions
                    func_path = f"{inferred_module_path}::{inferred.name}"
                    formatted_path_2 = f"{self.package_name}{format_path(inferred_module_path).split(self.package_name)[-1]}"
                    function_exe_cmd_2=f"{formatted_path_2}.{inferred.name}()"

                    if inferred.name.startswith("_"):
                        continue
                    if func_path in self.functions:
                        dependencies.add(function_exe_cmd_2)
            except astroid.InferenceError:
                continue

        modified_imports = set()

        for imp in imports:
            if imp.startswith("from"):
                # Apply regex to change 'from module' import to 'import module'
                modified_import = re.sub(r'^from (\w+)(\.\s*)?.*$', r'import \1', imp)
                modified_imports.add(modified_import)
            else:
                # If it's not a 'from' import, just add it as is
                modified_imports.add(imp)


        return FunctionInfo(
            id= str(uuid.uuid4()),
            function_name=node.name,
            module_path=file_path,
            params=args,
            returns=returns,
            function_exe_cmd=re.sub(r'\.{2,}', '.', function_exe_cmd),
            function_url=function_url,
            docstring=docstring,
            dependent_functions= dependencies,
            comments=comments,
            decorators=decorators,
            complexity=complexity,
            start_line=start_line,
            end_line=end_line,
            is_method=False,
            is_active=True,
            is_async=isinstance(node, nodes.AsyncFunctionDef),
            signature=signature,
            method_signature=re.sub(r'\.{2,}', '.', function_def),
            output_type="json",
            packages=filtered_packages,  # IMPROVED: Only include relevant packages
            imports=modified_imports,             # IMPROVED: Only include relevant imports
            runtime="python",
            is_updated=False,
            description="",
            description_embedding=""
        )

    def _extract_tests_info(self, node: nodes.FunctionDef, source: str) -> TestsInfo:
        """Extract detailed information about a test."""
        args = [arg.name for arg in node.args.args]
        returns = node.returns.as_string() if node.returns else None
        docstring = None
        if isinstance(node.doc_node, nodes.Const):
            docstring = node.doc_node.value.strip()
        complexity = _calculate_complexity(node)

        # Fix decorator handling
        decorators = []
        if node.decorators:
            for decorator in node.decorators.nodes:
                decorators.append(decorator.as_string())

        class_name = None
        if isinstance(node.parent, nodes.ClassDef):
            class_name = node.parent.name

        comments = []
        line_number=node.lineno
        end_line=node.end_lineno or node.lineno

        for line_num in range(line_number, end_line + 1):
            line = source.splitlines()[line_num - 1].strip()
            if '#' in line:
                comment_text = line.split('#', 1)[1].strip()
                comments.append(comment_text)

        signature=f"{node.name}({', '.join(arg.name for arg in node.args.args)})"
        formatted_path = format_path(self.relative_path)
        test_exe_cmd = f"{formatted_path}.{class_name}.{signature}" if class_name else f"{formatted_path}.{signature}"

        return TestsInfo(
            name=node.name,
            module_path="",
            args=args,
            returns=returns,
            test_exe_cmd=test_exe_cmd,
            docstring=docstring,
            comments=comments,
            decorators=decorators,  # Use the properly extracted decorators
            complexity=complexity,
            start_line=line_number,
            end_line=end_line,
            is_active=True,
            is_async=isinstance(node, nodes.AsyncFunctionDef),
            signature=signature,
            description="",
            description_embedding="",
            runtime="python"
        )

    def get_package_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics for the package."""
        return {
            'total_files': len(self.modules),
            'total_classes': len(self.classes),
            'total_functions': len(self.functions),
            'total_imports' : sum(len(imports) for imports in self.imports.values()),
            'average_complexity': sum(f.complexity for f in self.functions.values()) / len(self.functions) if self.functions else 0,
            'total_lines': sum(f.end_line - f.start_line for f in self.functions.values()),
            'errors': len(self.errors)
        }

    def convert_analyzer_to_knowledge_graph(self) -> str:
        """
        Convert analyzer results to RDF knowledge graph following the code ontology

        Args:
            analyzer: The package analyzer instance with analysis results
            output_path: Path to save the RDF data (optional)
            format: RDF serialization format (turtle/ttl, xml, n3, nt, jsonld)
                    Default is "turtle" which produces TTL format

        Returns:
            String containing the serialized RDF data
        """
        exporter = RDFExporter(self.git_url)

        exporter.add_functions(self.functions)
        exporter.add_classes(self.classes)
        exporter.add_apis(self.apis)
        exporter.add_modules(self.module_info)
        exporter.add_tests(self.tests)

        # Export in the specified format
        rdf_data = exporter.export_rdf()

        # Log or return the RDF data
        logger.info("RDF data generated successfully.")
        return rdf_data  # Ensure the RDF data is returned for further use

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
        ]

        # Add module information
        report.extend(["\nModules:"])
        if self.module_info:
            for file_path, module in sorted(self.module_info.items()):
                if not module.name.startswith("_"):
                    report.append(f"\n  {file_path}:")
                    report.extend([
                        f"    Name: {module.name}",
                        f"    Description: {module.description}",
                        f"    Description Embedding: {module.description_embedding}",
                    ])

        # Add class information
        report.extend(["\nClass Hierarchy:"])
        for class_path, class_info in sorted(self.classes.items()):
            report.extend([
                f"\n  {class_path}:",
                f"    Bases: {', '.join(class_info.bases) if class_info.bases else 'None'}",
                f"    Functions: {class_info.functions}",
                f"    Attributes: {class_info.attributes}",
                f"    Start Line: {class_info.start_line}",
                f"    End Line: {class_info.end_line}",
                f"    Decorators: {', '.join(class_info.decorators) if class_info.decorators else 'None'}",
                f"    Docstring: {class_info.docstring.strip() if class_info.docstring else 'None'}",
                f"    Description: {class_info.description}",
                f"    Description Embedding: {class_info.description_embedding}",
            ])

            # Add function information
        report.extend(["\nFunctions:"])
        for func_path, func_info in sorted(self.functions.items()):
            report.extend([
                f"\n  {func_path}:",
                f"    id: {func_info.id}",
                f"    name: {func_info.function_name}",
                f"    Module Path: {func_info.module_path}",
                f"    Signature: {func_info.signature}",
                f"    Packages: {func_info.packages}",
                f"    Imports: {func_info.imports}",
                f"    function_exe_cmd: {func_info.function_exe_cmd}",
                f"    Arguments: {func_info.params if func_info.params else 'None'}",
                f"    Returns: {func_info.returns if func_info.returns else 'None'}",
                f"    Start Line: {func_info.start_line}",
                f"    End Line: {func_info.end_line}",
                f"    Complexity: {func_info.complexity}",
                f"    Is Active: {func_info.is_active}",
                f"    Is Async: {func_info.is_async}",
                f"    Decorators: {', '.join(func_info.decorators) or 'None'}",
                f"    Docstring: {func_info.docstring if func_info.docstring else 'None'}",
                f"    Description: {func_info.description}",
                f"    Description Embedding: {func_info.description_embedding}",
                f"    Comments: {func_info.comments if func_info.comments else 'None'}",
                f"    Runtime: {func_info.runtime}"
            ])


        # Add apis information
        report.extend(["\nAPIs:"])
        for func_path, api_info in sorted(self.apis.items()):
            if api_info.http_method is not None:
                report.extend([
                    f"\n  {func_path}:",
                    f"    name: {api_info.name}",
                    f"    Module Path: {api_info.module_path if api_info.module_path else 'None'}",
                    f"    path: {api_info.path}",
                    f"    method: {api_info.http_method}",
                    f"    path_param: {api_info.path_params}",
                    f"    request_param: {api_info.request_params}",
                    f"    request_body: {api_info.request_body}",
                    f"    response_body: {api_info.response_body}",
                    f"    Description: {api_info.description}",
                    f"    Description Embedding: {api_info.description_embedding}",
                    f"    Is Active: {api_info.is_active}",
                    f"    Is Async: {api_info.is_async}",
                    f"    Runtime: {api_info.runtime}"
                ])

                # Add function information
        report.extend(["\nTests:"])
        for test_path, test_info in sorted(self.tests.items()):
            if not test_info.name.startswith("_"):
                report.extend([
                    f"\n  {test_path}:",
                    f"    name: {test_info.name}",
                    f"    Module Path: {test_info.module_path if test_info.module_path else 'None'}",
                    f"    Signature: {test_info.signature}",
                    f"    test_exe_cmd: {test_info.test_exe_cmd}",
                    f"    Arguments: {test_info.args if test_info.args else 'None'}",
                    f"    Returns: {test_info.returns if test_info.returns else 'None'}",
                    f"    Start Line: {test_info.start_line}",
                    f"    End Line: {test_info.end_line}",
                    f"    Complexity: {test_info.complexity}",
                    f"    Is Active: {test_info.is_active}",
                    f"    Is Async: {test_info.is_async}",
                    f"    Decorators: {', '.join(test_info.decorators) or 'None'}",
                    f"    Docstring: {test_info.docstring if test_info.docstring else 'None'}",
                    f"    Description: {test_info.description}",
                    f"    Description Embedding: {test_info.description_embedding}",
                    f"    Comments: {test_info.comments if test_info.comments else 'None'}",
                    f"    Runtime: {test_info.runtime}"
                ])


        # Add error information if any
        if self.errors:
            report.extend([
                "\nErrors:",
                *[f"- {path}: {error}" for path, error in self.errors]
            ])

        return '\n'.join(report)

    def get_functions_list(self) -> dict:
        result = []
        for func in self.functions.values():
            if func.signature.startswith("_"):
                continue
            # Convert the object to a dict automatically
            result.append(vars(func))  # or func.__dict__
        return { "library_name" : extract_repo_name(self.git_url), "functions" : result}

    def get_library_name(self, package_name):
        try:
            distribution = pkg_resources.get_distribution(package_name)
            return distribution.project_name  # This gives the correct library name
        except pkg_resources.DistributionNotFound:
            return package_name  # Fallback to the package name if not found
