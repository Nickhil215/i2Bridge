from typing import Any, Dict, Optional

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL


class RDFExporter:
    """Class to export Python package analysis results to RDF following a code ontology"""

    def __init__(self, git_url: Optional[str] = None):
        """Initialize the RDF exporter with optional output path"""
        self.git_url = git_url.rstrip(".git")
        self.graph = Graph()

        # Define namespaces
        self.CODE = Namespace(self.git_url)
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

            # Add complexity
            self.graph.add((func_uri, self.CODE.complexity,
                            Literal(func_info.complexity)))

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

            # Add function definition if available
            if func_info.function_def:
                self.graph.add((func_uri, self.CODE.function_def, Literal(func_info.function_def)))

            # Add function url if available
            if func_info.function_url:
                self.graph.add((func_uri, self.CODE.function_url, Literal(func_info.function_url)))

            # Add comments if available
            if func_info.comments:
                comments_list = Literal(", ".join(func_info.comments))
                self.graph.add((func_uri, self.CODE.comments, comments_list))

            #Add packages as a list
            if func_info.packages:
              packages_list = Literal(", ".join(func_info.packages))
              self.graph.add((func_uri, self.CODE.package, packages_list))

            # Add imports as dependencies
            for imp in func_info.imports:
              import_uri = self._get_component_uri("Function", imp)
              self.graph.add((func_uri, self.CODE.hasImport, import_uri))

            # Add dependent functions
            for dep_func in func_info.dependent_functions:
                dep_func_uri = self._get_component_uri("Function", dep_func)
                self.graph.add((func_uri, self.CODE.dependsOn, dep_func_uri))

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


    def export_rdf(self) -> str:

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
        rdf_format = format_map.get("turtle", "turtle")

        # Serialize to the specified format
        rdf_data = self.graph.serialize(format=rdf_format)

        return rdf_data
