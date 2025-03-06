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
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional
from typing import List, Set

import astroid
import requests
from astroid import nodes
from git import GitCommandError, Repo
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL
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
class FunctionInfo:
    """Store detailed information about a function or method."""
    name: str
    module_path: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    description: Optional[str]
    description_embedding: Optional[str]
    comments: List[str]
    decorators: List[str]
    complexity: int
    start_line: int
    end_line: int
    is_active: bool
    is_async: bool
    signature: str
    function_exe_cmd: str
    function_url: str
    packages: List[str]
    imports: List[str]
    runtime: str

@dataclass
class ApiInfo:
    """Store detailed information about a API endpoint"""
    path: str
    http_method: str
    name: str
    module_path: str
    path_params: List[str]
    request_params: List[str]
    request_body: Optional[str]
    response_body: Optional[str]
    description: Optional[str]
    description_embedding: Optional[str]
    is_active: bool
    is_async: bool
    # packages: List[str]
    # imports: List[str]
    # function_url: str
    runtime: str


@dataclass
class TestsInfo:
    """Store detailed information about tests."""
    name: str
    module_path: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    description: Optional[str]
    description_embedding: Optional[str]
    comments: List[str]
    decorators: List[str]
    complexity: int
    start_line: int
    end_line: int
    is_active: bool
    is_async: bool
    signature: str
    test_exe_cmd: str
    # packages: List[str]
    # imports: List[str]
    # function_url: str
    runtime: str

@dataclass
class ClassInfo:
    """Store detailed information about a class definition."""
    name: str
    bases: List[str]
    functions: List[str]
    attributes: List[str]
    docstring: Optional[str]
    decorators: List[str]
    start_line: int
    end_line: int
    description: Optional[str]
    description_embedding: Optional[str]

@dataclass
class ModuleInfo:
    """Store detailed information about each .py files required packages"""
    name: str
    functions: Dict[str, FunctionInfo]
    classes: Dict[str, ClassInfo]
    description: Optional[str]
    description_embedding: Optional[str]

@dataclass
class SubPackageInfo:
    name: str
    modules: Dict[str, ModuleInfo]
    description: Optional[str]
    description_embedding: Optional[str]

@dataclass
class PackageInfo:
    name: str
    modules: Dict[str, ModuleInfo]
    sub_package: Dict[str, SubPackageInfo]
    description: Optional[str]
    description_embedding: Optional[str]

@dataclass
class LibraryInfo:
    name: str
    package: Dict[str, PackageInfo]
    description: Optional[str]
    description_embedding: Optional[str]


class RDFExporter:
  """Class to export Python package analysis results to RDF following a code ontology"""

  def __init__(self, output_path: Optional[str] = None):
    """Initialize the RDF exporter with optional output path"""
    self.output_path = output_path
    self.graph = Graph()

    # Define namespaces
    self.CODE = Namespace("http://example.org/ontology/code-guru#")
    self.graph.bind("code", self.CODE)
    self.graph.bind("owl", OWL)
    self.graph.bind("rdfs", RDFS)

    # Add ontology declaration
    ontology_uri = URIRef("http://example.org/code-ontology")
    self.graph.add((ontology_uri, RDF.type, OWL.Ontology))
    self.graph.add((ontology_uri, URIRef("http://purl.org/dc/elements/1.1/title"),
                    Literal("Generated Code Component Knowledge Graph")))

  def _create_safe_uri(self, name: str) -> str:
    """Create a safe URI from a name, removing special characters"""
    return name.replace(" ", "_").replace(":", "_").replace("/", "_").replace(".", "_")

  def _get_component_uri(self, component_type: str, name: str) -> URIRef:
    """Create a URI for a code component"""
    safe_name = self._create_safe_uri(name)
    return URIRef(f"{self.CODE}{component_type}_{safe_name}")

  def add_functions(self, functions: Dict[str, Any]) -> None:
    """Add functions/methods to the RDF graph"""
    for func_path, func_info in functions.items():
      if func_info.name.startswith("_"):
        continue
      # Create URI for the function
      func_uri = self._get_component_uri("Function", f"{func_info.name}_{func_path}")

      # Add function type
      self.graph.add((func_uri, RDF.type, self.CODE.Function))

      # Add basic properties
      self.graph.add((func_uri, self.CODE.name, Literal(func_info.name)))
      if func_info.docstring:
        self.graph.add((func_uri, self.CODE.docstring, Literal(func_info.docstring)))
        self.graph.add((func_uri, self.CODE.isDocumented, Literal(True)))
      else:
        self.graph.add((func_uri, self.CODE.isDocumented, Literal(False)))

      # Add description
      self.graph.add((func_uri, self.CODE.description,
                      Literal(f"Description: {func_info.description}")))
      self.graph.add((func_uri, self.CODE.descriptionEmbedding,
                      Literal(f"DescriptionEmbedding: {func_info.description_embedding}")))

      # Add complexity
      self.graph.add((func_uri, self.CODE.complexity,
                      Literal(f"Complexity: {func_info.complexity}")))

      # Add parameters as inputs
      for arg in func_info.args:
        param_uri = self._get_component_uri("Parameter", f"{arg}_{func_info.name}")
        self.graph.add((param_uri, RDF.type, self.CODE.Parameter))
        self.graph.add((param_uri, self.CODE.name, Literal(arg)))
        self.graph.add((func_uri, self.CODE.hasInput, param_uri))

      # Add return type as output if available
      if func_info.returns:
        output_uri = self._get_component_uri("Parameter", f"output_{func_info.name}")
        self.graph.add((output_uri, RDF.type, self.CODE.Parameter))
        self.graph.add((output_uri, self.CODE.name, Literal("return")))

        # Map Python return types to ontology data types (simplified mapping)
        return_type = func_info.returns.lower()
        if "str" in return_type:
          datatype_uri = self.CODE.String
        elif "int" in return_type:
          datatype_uri = self.CODE.Integer
        elif "bool" in return_type:
          datatype_uri = self.CODE.Boolean
        elif "dict" in return_type or "json" in return_type:
          datatype_uri = self.CODE.JSON
        else:
          datatype_uri = self.CODE.Object

        self.graph.add((output_uri, self.CODE.hasDataType, datatype_uri))
        self.graph.add((func_uri, self.CODE.hasOutput, output_uri))

      # Add execution command if available
      if func_info.function_exe_cmd:
        self.graph.add((func_uri, self.CODE.function_exe_cmd, Literal(func_info.function_exe_cmd)))

      # Add comments if available
      if func_info.comments:
        comments_list = Literal(", ".join(func_info.comments))
        self.graph.add((func_uri, self.CODE.comments, comments_list))

  def add_apis(self, apis: Dict[str, Any]) -> None:
    """Add API endpoints to the RDF graph"""
    for api_path, api_info in apis.items():
      # Skip APIs without HTTP methods
      if api_info.http_method is None:
        continue

      # Create URI for the API
      api_uri = self._get_component_uri("API", f"{api_info.name}_{api_path}")

      # Add API type
      self.graph.add((api_uri, RDF.type, self.CODE.API))

      # Add basic properties
      self.graph.add((api_uri, self.CODE.name, Literal(api_info.name)))

      # Add path if available
      if api_info.path:
        self.graph.add((api_uri, self.CODE.path, Literal(api_info.path)))

      # Add HTTP method
      if api_info.http_method:
        method_uri = self._get_component_uri("HTTPMethod", api_info.http_method)
        self.graph.add((method_uri, RDF.type, self.CODE.HTTPMethod))
        self.graph.add((method_uri, RDFS.label, Literal(api_info.http_method)))
        self.graph.add((api_uri, self.CODE.hasMethod, method_uri))

        # Add data modification flag based on HTTP method
        is_modifying = api_info.http_method in ['POST', 'PUT', 'PATCH', 'DELETE']
        self.graph.add((api_uri, self.CODE.isDataModifying, Literal(is_modifying)))

      # Add path parameters
      if api_info.path_params:
        for param in api_info.path_params:
          param_uri = self._get_component_uri("PathParameter", f"{param}_{api_info.name}")
          self.graph.add((param_uri, RDF.type, self.CODE.PathParameter))
          self.graph.add((param_uri, self.CODE.name, Literal(param)))
          self.graph.add((param_uri, self.CODE.isRequired, Literal(True)))
          self.graph.add((param_uri, self.CODE.hasDataType, self.CODE.String))
          self.graph.add((api_uri, self.CODE.hasInput, param_uri))

      # Add request parameters
      if api_info.request_params:
        for param in api_info.request_params:
          param_uri = self._get_component_uri("RequestParameter", f"{param}_{api_info.name}")
          self.graph.add((param_uri, RDF.type, self.CODE.RequestParameter))
          self.graph.add((param_uri, self.CODE.name, Literal(param)))
          self.graph.add((api_uri, self.CODE.hasInput, param_uri))

      # Add description
      self.graph.add((api_uri, self.CODE.description,
                      Literal(f"Description: {api_info.description}")))
      self.graph.add((api_uri, self.CODE.descriptionEmbedding,
                      Literal(f"DescriptionEmbedding: {api_info.description_embedding}")))

      # Add request body if available
      if api_info.request_body:
        body_uri = self._get_component_uri("RequestBody", f"request_{api_info.name}")
        self.graph.add((body_uri, RDF.type, self.CODE.RequestBody))
        self.graph.add((body_uri, self.CODE.name, Literal(f"{api_info.name} Request")))
        # Assuming JSON for request bodies
        self.graph.add((body_uri, self.CODE.contentType, Literal("application/json")))
        self.graph.add((api_uri, self.CODE.hasRequestBody, body_uri))

      # Add response body if available
      if api_info.response_body:
        body_uri = self._get_component_uri("ResponseBody", f"response_{api_info.name}")
        self.graph.add((body_uri, RDF.type, self.CODE.ResponseBody))
        self.graph.add((body_uri, self.CODE.name, Literal(f"{api_info.name} Response")))
        # Assuming JSON for response bodies
        self.graph.add((body_uri, self.CODE.contentType, Literal("application/json")))
        self.graph.add((api_uri, self.CODE.hasResponseBody, body_uri))

  def add_tests(self, tests: Dict[str, Any]) -> None:
      """Add functions/methods to the RDF graph"""
      for func_path, test_info in tests.items():
          if test_info.name.startswith("_"):
              continue
          # Create URI for the function
          func_uri = self._get_component_uri("Tests", f"{test_info.name}_{func_path}")

          # Add function type
          self.graph.add((func_uri, RDF.type, self.CODE.Function))

          # Add basic properties
          self.graph.add((func_uri, self.CODE.name, Literal(test_info.name)))
          if test_info.docstring:
              self.graph.add((func_uri, self.CODE.docstring, Literal(test_info.docstring)))
              self.graph.add((func_uri, self.CODE.isDocumented, Literal(True)))
          else:
              self.graph.add((func_uri, self.CODE.isDocumented, Literal(False)))

          # Add description
          self.graph.add((func_uri, self.CODE.description,
                          Literal(f"Description: {test_info.description}")))
          self.graph.add((func_uri, self.CODE.descriptionEmbedding,
                          Literal(f"DescriptionEmbedding: {test_info.description_embedding}")))

          # Add complexity
          self.graph.add((func_uri, self.CODE.complexity,
                          Literal(f"Complexity: {test_info.complexity}")))

          # Add parameters as inputs
          for arg in test_info.args:
              param_uri = self._get_component_uri("Parameter", f"{arg}_{test_info.name}")
              self.graph.add((param_uri, RDF.type, self.CODE.Parameter))
              self.graph.add((param_uri, self.CODE.name, Literal(arg)))
              self.graph.add((func_uri, self.CODE.hasInput, param_uri))

          # Add description
          self.graph.add((func_uri, self.CODE.description,
                          Literal(f"Description: {test_info.description}")))
          self.graph.add((func_uri, self.CODE.descriptionEmbedding,
                          Literal(f"DescriptionEmbedding: {test_info.description_embedding}")))

          # Add return type as output if available
          if test_info.returns:
              output_uri = self._get_component_uri("Parameter", f"output_{test_info.name}")
              self.graph.add((output_uri, RDF.type, self.CODE.Parameter))
              self.graph.add((output_uri, self.CODE.name, Literal("return")))

              # Map Python return types to ontology data types (simplified mapping)
              return_type = test_info.returns.lower()
              if "str" in return_type:
                  datatype_uri = self.CODE.String
              elif "int" in return_type:
                  datatype_uri = self.CODE.Integer
              elif "bool" in return_type:
                  datatype_uri = self.CODE.Boolean
              elif "dict" in return_type or "json" in return_type:
                  datatype_uri = self.CODE.JSON
              else:
                  datatype_uri = self.CODE.Object

              self.graph.add((output_uri, self.CODE.hasDataType, datatype_uri))
              self.graph.add((func_uri, self.CODE.hasOutput, output_uri))

          # Add execution command if available
          if test_info.test_exe_cmd:
              self.graph.add((func_uri, self.CODE.function_exe_cmd, Literal(test_info.test_exe_cmd)))

          # Add comments if available
          if test_info.comments:
              comments_list = Literal(", ".join(test_info.comments))
              self.graph.add((func_uri, self.CODE.comments, comments_list))

  def add_classes(self, classes: Dict[str, Any]) -> None:
      """Add classes to the RDF graph"""
      for class_path, class_info in classes.items():
          # Create URI for the class
          formatted_class_path = str(class_path).replace("py::","")
          class_uri = self._get_component_uri("Class", f"{formatted_class_path}")

          # Add class
          self.graph.add((class_uri, RDF.type, self.CODE.Class))

          # Add basic properties
          self.graph.add((class_uri, self.CODE.name, Literal(class_info.name)))
          if class_info.docstring:
              self.graph.add((class_uri, self.CODE.docstring, Literal(class_info.docstring)))
              self.graph.add((class_uri, self.CODE.isDocumented, Literal(True)))
          else:
              self.graph.add((class_uri, self.CODE.isDocumented, Literal(False)))

          # Add base classes as dependencies
          for base in class_info.bases:
              if base != "object":  # Skip Python's base object
                  base_uri = self._get_component_uri("Class", base)
                  self.graph.add((class_uri, self.CODE.dependsOn, base_uri))

          # Add description
          self.graph.add((class_uri, self.CODE.description,
                          Literal(f"Description: {class_info.description}")))
          self.graph.add((class_uri, self.CODE.descriptionEmbedding,
                          Literal(f"DescriptionEmbedding: {class_info.description_embedding}")))

          # Connect methods to the class
          for func_name in class_info.functions:
              if func_name.startswith("_"):
                  continue
              func_uri = self._get_component_uri("Function", f"{formatted_class_path}_{func_name}")
              self.graph.add((class_uri, self.CODE.connectsTo, func_uri))

  def add_modules(self, components: Dict[str, Any]) -> None:
    """Add code components and their dependencies to the RDF graph"""
    for component_path, component_info in components.items():
      if component_info.name.startswith("_"):
        continue
      # Create URI for the component
      component_uri = self._get_component_uri("CodeComponent", f"{component_info.name}_{component_path}")

      # Add component type
      self.graph.add((component_uri, RDF.type, self.CODE.CodeComponent))

      # Add basic properties
      self.graph.add((component_uri, self.CODE.name, Literal(component_info.name)))

      # Add description
      self.graph.add((component_uri, self.CODE.description,
                      Literal(f"Description: {component_info.description}")))
      self.graph.add((component_uri, self.CODE.descriptionEmbedding,
                      Literal(f"DescriptionEmbedding: {component_info.description_embedding}")))

      # # Add packages as a list
      # if component_info.packages:
      #   packages_list = Literal(", ".join(component_info.packages))
      #   self.graph.add((component_uri, self.CODE.Packages, packages_list))

      # # Add imports as dependencies
      # for imp in component_info.imports:
      #   # Create a simpler name for the import
      #   import_name = imp.split(" as ")[0] if " as " in imp else imp
      #   import_uri = self._get_component_uri("CodeComponent", import_name)
      #   self.graph.add((component_uri, self.CODE.dependsOn, import_uri))

  def export_rdf(self, output_path: Optional[str] = None, format: str = "turtle") -> str:
    """
    Export the RDF graph in the specified format

    Args:
        output_path: Path to save the output file
        format: RDF serialization format (turtle, xml, n3, nt, jsonld)
                Default is "turtle" which produces TTL format

    Returns:
        String of the serialized RDF data
    """
    path = output_path or self.output_path

    # Map format names to rdflib formats
    format_map = {
      "turtle": "turtle",  # TTL format
      "ttl": "turtle",
      "xml": "xml",        # RDF/XML
      "rdfxml": "xml",
      "n3": "n3",          # Notation3
      "nt": "nt",          # N-Triples
      "ntriples": "nt",
      "jsonld": "json-ld", # JSON-LD
      "json-ld": "json-ld",
    }

    # Get the correct format string for rdflib
    rdf_format = format_map.get(format.lower(), "turtle")

    # Serialize to the specified format
    rdf_data = self.graph.serialize(format=rdf_format)

    # Write to file if path is provided
    if path:
      # Ensure directory exists
      os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

      with open(path, "w", encoding="utf-8") as f:
        f.write(rdf_data)
      logger.info(f"Knowledge graph exported to {path} in {format} format")

    return rdf_data

  def export_to_turtle(self, output_path: Optional[str] = None) -> str:
    """Export the RDF graph to Turtle format (TTL)"""
    return self.export_rdf(output_path, format="turtle")

  # Function to convert analyzer results to knowledge graph


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

    def _analyze_file(self, file_path: str, git_url: str = None) -> None:
        """Analyze a single Python file and extract its AST information."""
        try:
            relative_path = os.path.relpath(file_path, self.package_path)
            logger.info(f"Analyzing file: {relative_path}")
            self.source_path = relative_path

            package_name=None
            if git_url:
                package_name = extract_repo_name(git_url)


            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse the file using astroid
            module = astroid.parse(source, path=file_path)
            self.modules[relative_path] = module

            self._analyze_modules(module, relative_path, source)
            self._analyze_classes(module, relative_path, source, package_name)

            # Pass source to analysis methods
            if ("tests" in relative_path.split("/")
                or "test" in relative_path.split("/")
                or relative_path.startswith("test")):
              self._analyze_tests(module, relative_path, source)
            elif not relative_path.split('/')[-1].__contains__("_"):
              self._analyze_functions(module, relative_path, source, package_name)
              self._analyze_api_endpoint(module, relative_path, source)

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            self.errors.append((file_path, str(e)))

    def _analyze_functions(self, module: nodes.Module, file_path: str, source: str, package_name:str = None) -> None:
        """Analyze all functions in a module, including methods."""
        for node in module.body:
          if isinstance(node, nodes.ClassDef):
            # If it's a class, find all functions inside it
            for class_node in node.body:
              if isinstance(class_node, nodes.FunctionDef):
                func_info = self._extract_function_info(class_node, source, file_path, package_name)
                self.functions[f"{file_path}::{func_info.name}"] = func_info
          elif isinstance(node, nodes.FunctionDef):
            # If it's a function at the module level
              func_info = self._extract_function_info(node, source, file_path, package_name)
              self.functions[f"{file_path}::{func_info.name}"] = func_info


    def _analyze_api_endpoint(self, module: nodes.Module, file_path: str, source: str) -> None:
        """Analyze all api endpoints in a module"""

        for node in module.body:
          if isinstance(node, nodes.ClassDef):
            # If it's a class, find all functions inside it
            for class_node in node.body:
              if isinstance(class_node, nodes.FunctionDef):
                api_info = self._extract_api_endpoint_info(class_node, source)
                self.apis[f"{file_path}::{api_info.name}"] = api_info
          elif isinstance(node, nodes.FunctionDef):
            # If it's a function at the module level
            api_info = self._extract_api_endpoint_info(node, source)
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

    def _analyze_classes(self, module: nodes.Module, file_path: str, source: str, package_name:str = None) -> None:
        """Analyze all classes in a module."""
        for node in module.nodes_of_class(nodes.ClassDef):
            class_info = self._extract_class_info(node, source, package_name)
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
                    import_name = name if asname is None else asname
                    package_name = import_name.split('.')[0]
                    if package_name not in self.packages[file_path]:
                        self.packages[file_path].append(package_name)  # Add package to the file_path list
                    self.imports[file_path].append(import_name)
            elif isinstance(node, nodes.ImportFrom):
                module_name = node.modname
                for name, asname in node.names:  # Unpacking (name, alias) tuples
                    import_name = f"{module_name}.{name}" if asname is None else f"{module_name}.{name} as {asname}"
                    package_name = import_name.split('.')[0]
                    if package_name not in self.packages[file_path]:
                        self.packages[file_path].append(package_name)  # Add package to the file_path list
                    self.imports[file_path].append(import_name)

        self.module_info[file_path] = ModuleInfo(
            name=file_path.split('/')[-1],
            description="",
            description_embedding="",
            classes=[],
            functions=[])



    def format_path(self, path):
      # Split the path by "/"
      split_path = path.split('/')

      # Remove the first element ("..")
      split_path = split_path[1:]

      # Join the remaining parts with "."
      joined_path = '.'.join(split_path)

      # Remove the ".py" extension using slicing
      joined_path = joined_path[:-3]  # Removes last 3 characters (".py")

      return joined_path


    def get_used_packages(self, start_line, end_line, source_code, file_path):
        # Extract the relevant lines from the source code (lines are 1-indexed, so adjust accordingly)
        lines = source_code.splitlines()[start_line-1:end_line]

        # Define a list to store packages
        packages = []
        if file_path not in self.packages:
            self.packages[file_path] = []

        for module_package in self.packages[file_path]:
            for line in lines:
                if module_package in line and module_package not in packages:
                    packages.append(module_package)
        return packages


    def _extract_function_info(self, node: nodes.FunctionDef, source: str,
                               file_path: str, package_name:str = None) -> FunctionInfo:
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

        class_name = None
        if isinstance(node.parent, nodes.ClassDef):
          class_name = node.parent.name

        comments = []
        start_line=node.lineno
        end_line=node.end_lineno or node.lineno

        signature=None
        for line_num in range(start_line, end_line + 1):
            line = source.splitlines()[line_num - 1].strip()
            if line_num == start_line:
                signature = line.strip().replace("def ", "")
            if '#' in line:
                comment_text = line.split('#', 1)[1].strip()
                comments.append(comment_text)

        formatted_path = f"{package_name}{self.format_path(self.source_path).split(package_name)[-1]}"
        function_exe_cmd=f"{formatted_path}.{class_name}.{node.name}()" if class_name else f"{formatted_path}.{node.name}()"

        imports=[]
        packages = self.get_used_packages(start_line, end_line, source, file_path)

        return FunctionInfo(
            name=node.name,
            module_path=file_path,
            args=args,
            returns=returns,
            function_exe_cmd=function_exe_cmd,
            function_url="",
            docstring=docstring,
            description="",
            description_embedding="",
            comments=comments,
            decorators=decorators,  # Use the properly extracted decorators
            complexity=complexity,
            start_line=start_line,
            end_line=end_line,
            is_active=True,
            is_async=isinstance(node, nodes.AsyncFunctionDef),
            signature=signature,
            packages=packages,
            imports=imports,
            runtime="python"
        )

    def _extract_api_endpoint_info(self, node: nodes.FunctionDef, source: str) -> ApiInfo | None:
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

    def _extract_tests_info(self, node: nodes.FunctionDef, source: str) -> TestsInfo:
        """Extract detailed information about a test."""
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
        formatted_path = self.format_path(self.source_path)
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

    def _extract_class_info(self, node: nodes.ClassDef, source: str, package_name:str = None) -> ClassInfo:
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
        exporter = RDFExporter("output.ttl")

        exporter.add_functions(self.functions)
        exporter.add_classes(self.classes)
        exporter.add_apis(self.apis)
        exporter.add_modules(self.module_info)
        exporter.add_tests(self.tests)

        # Export in the specified format
        rdf_data = exporter.export_rdf("output.ttl", "turtle")

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
        if not func_info.name.startswith("_"):
          report.extend([
            f"\n  {func_path}:",
            f"    name: {func_info.name}",
            f"    Module Path: {func_info.module_path if func_info.module_path else 'None'}",
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
                    self._analyze_file(os.path.join(root, file), self.git_url)

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

def extract_repo_name(github_url):
    # Use regular expression to extract the repo name from the URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if match:
        return match.group(2).replace('.git', '')  # The second group is the repo name
    return None

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
            txt_output = analyzer.generate_report()
            ttl_output = analyzer.convert_analyzer_to_knowledge_graph()


            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(txt_output)
                    logger.info(f"Analysis report saved to: {output_file}")

            return ttl_output

    except Exception as e:
        error_msg = f"Error analyzing package: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

class ApiException(Exception):
    """ Custom exception for API-related errors """
    pass


def upload_to_content_service(content):
    file_name = f"ttl_output_{uuid.uuid1().hex}.ttl"

    url = "https://ig.aidtaas.com/mobius-content-service/v1.0/content/upload?filePath=python"
    bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3Ny1NUVdFRTNHZE5adGlsWU5IYmpsa2dVSkpaWUJWVmN1UmFZdHl5ejFjIn0.eyJleHAiOjE3NDExOTM2NjAsImlhdCI6MTc0MTE1NzY2MCwianRpIjoiZjI5ZmRhMWUtZTQ5OC00OTU0LWI5MTItMGZlNzdjOWZhYzUxIiwiaXNzIjoiaHR0cDovL2tleWNsb2FrLXNlcnZpY2Uua2V5Y2xvYWsuc3ZjLmNsdXN0ZXIubG9jYWw6ODA4MC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIkJPTFRaTUFOTl9CT1RfbW9iaXVzIiwiUEFTQ0FMX0lOVEVMTElHRU5DRV9tb2JpdXMiLCJNT05FVF9tb2JpdXMiLCJWSU5DSV9tb2JpdXMiLCJhY2NvdW50Il0sInN1YiI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInR5cCI6IkJlYXJlciIsImF6cCI6IkhPTEFDUkFDWV9tb2JpdXMiLCJzaWQiOiIzZmFlYWE4OS0wMjUwLTQ0OTQtYjE3NC04MjA1YTIxMzMyZDkiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIkhDWV9URU5BTlRfQ1VTVE9NIiwib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsIkJPQl9URU5BTlRfQ1VTVE9NIiwiUElfVEVOQU5UX0NVU1RPTSIsIk1PTkVUX1RFTkFOVF9DVVNUT00iXX0sInJlc291cmNlX2FjY2VzcyI6eyJIT0xBQ1JBQ1lfbW9iaXVzIjp7InJvbGVzIjpbIkFHRU5UX1dSSVRFIiwiSE9MQUNSQUNZX0FQUFJPVkUiLCJIT0xBQ1JBQ1lfV1JJVEUiLCJIT0xBQ1JBQ1lfUkVBRCIsIkhPTEFDUkFDWV9FWEVDVVRFIiwiSE9MQUNSQUNZX1VTRVIiLCJBR0VOVF9FWEVDVVRFIiwiQUdFTlRfUkVBRCJdfSwiQk9MVFpNQU5OX0JPVF9tb2JpdXMiOnsicm9sZXMiOlsiQk9MVFpNQU5OX0JPVF9VU0VSIiwiQk9CX0VYRUNVVEUiLCJCT0JfQVBQUk9WRSIsIkJPQl9SRUFEIiwiQk9CX1dSSVRFIl19LCJQQVNDQUxfSU5URUxMSUdFTkNFX21vYml1cyI6eyJyb2xlcyI6WyJQQVNDQUxfSU5URUxMSUdFTkNFX1VTRVIiLCJJTkdFU1RJT05fUkVBRCIsIkNPSE9SVFNfQVBQUk9WRSIsIkNPSE9SVFNfV1JJVEUiLCJTQ0hFTUFfV1JJVEUiLCJJTkdFU1RJT05fQVBQUk9WRSIsIlVOSVZFUlNFX1JFQUQiLCJJTkdFU1RJT05fV1JJVEUiLCJJTkdFU1RJT05fRVhFQ1VURSIsIlNDSEVNQV9FWEVDVVRFIiwiU0NIRU1BX1JFQUQiLCJCSUdRVUVSWV9SRUFEIiwiQklHUVVFUllfV1JJVEUiLCJDT0hPUlRTX1JFQUQiLCJTQ0hFTUFfQVBQUk9WRSIsIlVOSVZFUlNFX0FQUFJPVkUiLCJCSUdRVUVSWV9FWEVDVVRFIiwiVU5JVkVSU0VfV1JJVEUiLCJDT0hPUlRTX0VYRUNVVEUiLCJCSUdRVUVSWV9BUFBST1ZFIiwiVU5JVkVSU0VfRVhFQ1VURSJdfSwiTU9ORVRfbW9iaXVzIjp7InJvbGVzIjpbIk1PTkVUX0FQUFJPVkUiLCJNT05FVF9VU0VSIiwiTU9ORVRfRVhFQ1VURSIsIk1PTkVUX1JFQUQiLCJNT05FVF9XUklURSJdfSwiVklOQ0lfbW9iaXVzIjp7InJvbGVzIjpbIlZJTkNJX1VTRVIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInJlcXVlc3RlclR5cGUiOiJURU5BTlQiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6IkFpZHRhYXMgQWlkdGFhcyIsInRlbmFudElkIjoiMmNmNzZlNWYtMjZhZC00ZjJjLWJjY2MtZjRiYzFlN2JmYjY0IiwicHJlZmVycmVkX3VzZXJuYW1lIjoicGFzc3dvcmRfdGVuYW50X2FpZHRhYXNAZ2FpYW5zb2x1dGlvbnMuY29tIiwiZ2l2ZW5fbmFtZSI6IkFpZHRhYXMiLCJmYW1pbHlfbmFtZSI6IkFpZHRhYXMiLCJlbWFpbCI6InBhc3N3b3JkX3RlbmFudF9haWR0YWFzQGdhaWFuc29sdXRpb25zLmNvbSJ9.X_i9DMNytrYhDpFNjNWQnWjwK2qZCr12jmPE8D7DaTeCZHwTW-y27Pf-pbLZ_HpA5usK8p8h_S3LLOUjmoGHFrHTRPB0oGPYTiIt80p07wqM2aeyGc_418Q5neGdn22gSjtK_z3_RJY1JlFm4d1DHg9uQlCaffknbBv3uH8XFg-yZqVCkU31FmQY6jbTUNHJsU8srgyJ5NCnqVb-1DLkz0sXf1IhxjZyKD9GY5EAP17OhMPV0JYRZ75W9bmcsnou6RNFmJxdtZMl8kP4h1G_oFGLQKodxnWUbKq-ZdcfNxkcage-mQLBeE8JOIm8-85SwNk_NsfzXdw1f46OcjaTPQ"

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
        return json.loads(json.dumps(response.json())).get("cdnUrl")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        raise ApiException("Failed to make API call")
    except ValueError as e:
        logging.error(f"JSON parsing error: {e}")
        raise ApiException("Object mapping failure")


def import_ontology(cdn_url):

    url = 'https://ig.aidtaas.com/pi-ontology-service/ontology/v1.0/patch?graphDb=NEO4J'
    bearer_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI3Ny1NUVdFRTNHZE5adGlsWU5IYmpsa2dVSkpaWUJWVmN1UmFZdHl5ejFjIn0.eyJleHAiOjE3NDExOTM2NjAsImlhdCI6MTc0MTE1NzY2MCwianRpIjoiZjI5ZmRhMWUtZTQ5OC00OTU0LWI5MTItMGZlNzdjOWZhYzUxIiwiaXNzIjoiaHR0cDovL2tleWNsb2FrLXNlcnZpY2Uua2V5Y2xvYWsuc3ZjLmNsdXN0ZXIubG9jYWw6ODA4MC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIkJPTFRaTUFOTl9CT1RfbW9iaXVzIiwiUEFTQ0FMX0lOVEVMTElHRU5DRV9tb2JpdXMiLCJNT05FVF9tb2JpdXMiLCJWSU5DSV9tb2JpdXMiLCJhY2NvdW50Il0sInN1YiI6IjJjZjc2ZTVmLTI2YWQtNGYyYy1iY2NjLWY0YmMxZTdiZmI2NCIsInR5cCI6IkJlYXJlciIsImF6cCI6IkhPTEFDUkFDWV9tb2JpdXMiLCJzaWQiOiIzZmFlYWE4OS0wMjUwLTQ0OTQtYjE3NC04MjA1YTIxMzMyZDkiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIi8qIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIkhDWV9URU5BTlRfQ1VTVE9NIiwib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsIkJPQl9URU5BTlRfQ1VTVE9NIiwiUElfVEVOQU5UX0NVU1RPTSIsIk1PTkVUX1RFTkFOVF9DVVNUT00iXX0sInJlc291cmNlX2FjY2VzcyI6eyJIT0xBQ1JBQ1lfbW9iaXVzIjp7InJvbGVzIjpbIkFHRU5UX1dSSVRFIiwiSE9MQUNSQUNZX0FQUFJPVkUiLCJIT0xBQ1JBQ1lfV1JJVEUiLCJIT0xBQ1JBQ1lfUkVBRCIsIkhPTEFDUkFDWV9FWEVDVVRFIiwiSE9MQUNSQUNZX1VTRVIiLCJBR0VOVF9FWEVDVVRFIiwiQUdFTlRfUkVBRCJdfSwiQk9MVFpNQU5OX0JPVF9tb2JpdXMiOnsicm9sZXMiOlsiQk9MVFpNQU5OX0JPVF9VU0VSIiwiQk9CX0VYRUNVVEUiLCJCT0JfQVBQUk9WRSIsIkJPQl9SRUFEIiwiQk9CX1dSSVRFIl19LCJQQVNDQUxfSU5URUxMSUdFTkNFX21vYml1cyI6eyJyb2xlcyI6WyJQQVNDQUxfSU5URUxMSUdFTkNFX1VTRVIiLCJJTkdFU1RJT05fUkVBRCIsIkNPSE9SVFNfQVBQUk9WRSIsIkNPSE9SVFNfV1JJVEUiLCJTQ0hFTUFfV1JJVEUiLCJJTkdFU1RJT05fQVBQUk9WRSIsIlVOSVZFUlNFX1JFQUQiLCJJTkdFU1RJT05fV1JJVEUiLCJJTkdFU1RJT05fRVhFQ1VURSIsIlNDSEVNQV9FWEVDVVRFIiwiU0NIRU1BX1JFQUQiLCJCSUdRVUVSWV9SRUFEIiwiQklHUVVFUllfV1JJVEUiLCJDT0hPUlRTX1JFQUQiLCJTQ0hFTUFfQVBQUk9WRSIsIlVOSVZFUlNFX0FQUFJPVkUiLCJCSUdRVUVSWV9FWEVDVVRFIiwiVU5JVkVSU0VfV1JJVEUiLCJDT0hPUlRTX0VYRUNVVEUiLCJCSUdRVUVSWV9BUFBST1ZFIiwiVU5JVkVSU0VfRVhFQ1VURSJdfSwiTU9ORVRfbW9iaXVzIjp7InJvbGVzIjpbIk1PTkVUX0FQUFJPVkUiLCJNT05FVF9VU0VSIiwiTU9ORVRfRVhFQ1VURSIsIk1PTkVUX1JFQUQiLCJNT05FVF9XUklURSJdfSwiVklOQ0lfbW9iaXVzIjp7InJvbGVzIjpbIlZJTkNJX1VTRVIiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInJlcXVlc3RlclR5cGUiOiJURU5BTlQiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6IkFpZHRhYXMgQWlkdGFhcyIsInRlbmFudElkIjoiMmNmNzZlNWYtMjZhZC00ZjJjLWJjY2MtZjRiYzFlN2JmYjY0IiwicHJlZmVycmVkX3VzZXJuYW1lIjoicGFzc3dvcmRfdGVuYW50X2FpZHRhYXNAZ2FpYW5zb2x1dGlvbnMuY29tIiwiZ2l2ZW5fbmFtZSI6IkFpZHRhYXMiLCJmYW1pbHlfbmFtZSI6IkFpZHRhYXMiLCJlbWFpbCI6InBhc3N3b3JkX3RlbmFudF9haWR0YWFzQGdhaWFuc29sdXRpb25zLmNvbSJ9.X_i9DMNytrYhDpFNjNWQnWjwK2qZCr12jmPE8D7DaTeCZHwTW-y27Pf-pbLZ_HpA5usK8p8h_S3LLOUjmoGHFrHTRPB0oGPYTiIt80p07wqM2aeyGc_418Q5neGdn22gSjtK_z3_RJY1JlFm4d1DHg9uQlCaffknbBv3uH8XFg-yZqVCkU31FmQY6jbTUNHJsU8srgyJ5NCnqVb-1DLkz0sXf1IhxjZyKD9GY5EAP17OhMPV0JYRZ75W9bmcsnou6RNFmJxdtZMl8kP4h1G_oFGLQKodxnWUbKq-ZdcfNxkcage-mQLBeE8JOIm8-85SwNk_NsfzXdw1f46OcjaTPQ"
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    data = {
        "ontologyId": "67c7f94236726044b0df97a8",
        "universes": [
            "623ebf349279af04cdd709dc"
        ],
        "url": cdn_url,
        "ontologyName": "API",
        "semanticStructures": "DATA",
        "fileType": "Turtle",
        "tenantID": "2cf76e5f-26ad-4f2c-bccc-f4bc1e7bfb64"
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
        ttl_output = analyze_package(package_path, "report.txt")

        # cdn_url = upload_to_content_service(ttl_output)
        # logging.info(f"------- cdn_url:  {cdn_url} --------")
        # import_ontology(cdn_url)

        # print(report)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
