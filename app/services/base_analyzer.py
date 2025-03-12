import os
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional
from typing import List, Set

import astroid
from astroid import nodes

from app import logger
from app.models.data_classes import ClassInfo, ModuleInfo, FunctionInfo, TestsInfo, ApiInfo
from app.services.rdf_exporter import RDFExporter


def extract_repo_name(github_url):
    # Use regular expression to extract the repo name from the URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if match:
        return match.group(2).replace('.git', '')  # The second group is the repo name
    return None


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
            logger.info(f"Analyzing file: {file_path}")
            self.relative_path = file_path

            self.requirement_info = requirement_info
            self.git_url = git_url
            self.branch = branch

            self.package_name=None
            if git_url:
                self.package_name = extract_repo_name(git_url)

            self.formatted_path = f"{self.package_name}{format_path(self.relative_path).split(self.package_name)[-1]}"

            with open(temp_file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the file using astroid
            module = astroid.parse(source, path=temp_file_path)
            self.modules[file_path] = module

            self._analyze_modules(module, file_path, source)
            self._analyze_classes(module, file_path, source)

            # Pass source to analysis methods
            if ("tests" in file_path.split("/")
                    or "test" in file_path.split("/")
                    or file_path.startswith("test")):
                self._analyze_tests(module, file_path, source)
            elif not file_path.split('/')[-1].__contains__("_"):
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
                        self.functions[f"{file_path}::{func_info.name}"] = func_info
            elif isinstance(node, nodes.FunctionDef):
                # If it's a function at the module level
                func_info = self._extract_function_info(node, source, file_path)
                self.functions[f"{file_path}::{func_info.name}"] = func_info


    def _analyze_api_endpoint(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all api endpoints in a module"""
        for node in module.body:
            if isinstance(node, nodes.ClassDef):
                # If it's a class, find all functions inside it
                for class_node in node.body:
                    if isinstance(class_node, nodes.FunctionDef):
                        api_info = _extract_api_endpoint_info(class_node, source)
                        self.apis[f"{file_path}::{api_info.name}"] = api_info
            elif isinstance(node, nodes.FunctionDef):
                # If it's a function at the module level
                api_info = _extract_api_endpoint_info(node, source)
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

                    if asname is not None:
                        package_name = name.split('.')[0]

                    if package_name not in self.packages[file_path]:
                        self.packages[file_path].append(package_name)  # Add package to the file_path list

                    self.imports[file_path].append(f"import {import_name}")
            elif isinstance(node, nodes.ImportFrom):
                module_name = node.modname
                for name, asname in node.names:  # Unpacking (name, alias) tuples
                    if not module_name:
                        import_name = f"{name}" if asname is None else f"{name} as {asname}"
                        self.imports[file_path].append(f"from {self.formatted_path} import {import_name}")
                    else:
                        import_name = f"{module_name}.{name}" if asname is None else f"{module_name}.{name} as {asname}"
                        package_name = import_name.split('.')[0]
                        if package_name not in self.packages[file_path]:
                            self.packages[file_path].append(package_name)  # Add package to the file_path list
                        self.imports[file_path].append(f"from {module_name} import {import_name}")

        self.module_info[file_path] = ModuleInfo(
            name=file_path.split('/')[-1],
            description="",
            description_embedding="",
            classes=[],
            functions=[])

    def get_used_imports(self, start_line, end_line, source_code, file_path):
        # Extract the relevant lines from the source code (lines are 1-indexed, so adjust accordingly)
        lines = source_code.splitlines()[start_line-1:end_line]

        # Define a list to store packages
        packages = []
        imports = []
        if file_path not in self.packages:
            self.packages[file_path] = []
            self.imports[file_path] = []

        for line in lines:
            for module_package in self.packages[file_path]:
                if module_package in line and module_package not in packages:
                    packages.append(module_package)
                    for module_import in self.imports[file_path]:
                        if module_package in module_import and module_import not in imports:
                            imports.append(module_import)

        return imports

    # Function to recursively collect dependencies
    def find_dependencies(self, node, file_path, package_path, package_name, functions, dependencies, all_deps):
        """
        Recursively find dependencies for the given node (function) and propagate dependencies.
        """
        # Iterate through all function calls in the node
        for call in node.nodes_of_class(astroid.Call):
            try:
                # Infer the function being called
                inferred = next(call.func.infer())

                if isinstance(inferred, nodes.FunctionDef):
                    # Get the module path where the called function is defined
                    inferred_module_path = os.path.relpath(
                        inferred.root().file,
                        package_path
                    ) if inferred.root().file else file_path

                    # Build the function path as stored in self.functions
                    func_path = f"{inferred_module_path}::{inferred.name}"
                    formatted_path_2 = f"{package_name}{format_path(inferred_module_path).split(package_name)[-1]}"
                    function_exe_cmd_2 = f"{formatted_path_2}.{inferred.name}()"

                    if inferred.name.startswith("_"):
                        continue

                    # If the function is in the functions set, add its execution command to the dependencies
                    if func_path in functions:
                        dependencies.add(function_exe_cmd_2)

                        # If the function is not already in all_deps, initialize its dependency set
                        if inferred.name not in all_deps:
                            all_deps[inferred.name] = set()

                        # Now recursively find dependencies in the called function
                        self.find_dependencies(inferred, file_path, package_path, package_name, functions, all_deps[inferred.name], all_deps)

                        # After finding dependencies, merge them with the current function's dependencies
                        dependencies.update(all_deps[inferred.name])

            except astroid.InferenceError:
                continue

    def _extract_function_info(self, node: nodes.FunctionDef, source: str,
                               file_path: str) -> FunctionInfo:
        """Extract detailed information about a function."""
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
        start_line=node.lineno
        end_line=node.end_lineno or node.lineno

        for line_num in range(start_line, end_line + 1):
            line = source.splitlines()[line_num - 1].strip()
            if '#' in line:
                comment_text = line.split('#', 1)[1].strip()
                comments.append(comment_text)

        function_exe_cmd=f"{self.formatted_path}.{class_name}.{node.name}()" if class_name else f"{self.formatted_path}.{node.name}()"
        signature=f"{node.name}({', '.join(arg.name for arg in node.args.args)})"
        imports = self.get_used_imports(start_line, end_line, source, file_path)

        function_def = f"{self.formatted_path}.{class_name}.{signature}" if class_name else f"{self.formatted_path}.{signature}"

        prefix = self.git_url.rstrip(".git")
        path = file_path.lstrip("../")

        function_url = f"{prefix}/blob/{self.branch}/{path}#L{start_line}"

        dependencies = set()
        # all_deps = {}
        # self.find_dependencies(node, file_path, self.package_path, self.package_name, self.functions, dependencies, all_deps)
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

        return FunctionInfo(
            id= str(uuid.uuid4()),
            name=node.name,
            module_path=file_path,
            args=args,
            returns=returns,
            function_exe_cmd=function_exe_cmd,
            function_url=function_url,
            docstring=docstring,
            description="",
            description_embedding="",
            dependent_functions= dependencies,
            comments=comments,
            decorators=decorators,  # Use the properly extracted decorators
            complexity=complexity,
            start_line=start_line,
            end_line=end_line,
            is_method=False,
            is_active=True,
            is_async=isinstance(node, nodes.AsyncFunctionDef),
            signature=signature,
            function_def=function_def,
            packages=self.requirement_info,
            imports=imports,
            runtime="python",
            is_updated=False
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
                f"    name: {func_info.name}",
                f"    Module Path: {func_info.module_path}",
                f"    Signature: {func_info.signature}",
                f"    Packages: {func_info.packages}",
                f"    Imports: {func_info.imports}",
                f"    function_exe_cmd: {func_info.function_exe_cmd}",
                f"    Arguments: {func_info.args if func_info.args else 'None'}",
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

    def get_functions_list(self) -> List[dict]:
        result = []
        for func in self.functions.values():
            if func.signature.startswith("_"):
                continue
            # Convert the object to a dict automatically
            result.append(vars(func))  # or func.__dict__
        return result


