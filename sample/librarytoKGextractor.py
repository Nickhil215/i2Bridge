import astroid
from astroid import nodes
import os
import sys
import inspect
from typing import List, Dict, Optional, Set, Any, Union, Tuple
import json
from dataclasses import dataclass, field, asdict
import importlib
import logging
import time
import re
from radon.visitors import ComplexityVisitor
import pkg_resources

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FunctionInfo:
    name: str
    module: str
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    complexity: int = 0
    line_number: int = 0
    end_line: int = 0
    is_method: bool = False
    is_async: bool = False
    signature: str = ""
    is_c_function: bool = False  # For C extensions
    is_imported: bool = False    # Is this imported from another module?
    imported_from: str = ""      # Which module it was imported from

@dataclass
class ClassInfo:
    name: str
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, FunctionInfo] = field(default_factory=dict)
    attributes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0
    end_line: int = 0
    is_c_class: bool = False     # For C extensions
    is_imported: bool = False    # Is this imported from another module?
    imported_from: str = ""      # Which module it was imported from

@dataclass
class ConstantInfo:
    name: str
    value_type: str              # Type of the constant (str, int, etc.)
    is_imported: bool = False    # Is this imported from another module?
    imported_from: str = ""      # Which module it was imported from

@dataclass
class ModuleInfo:
    name: str
    docstring: Optional[str] = None
    version: str = "Unknown"
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    constants: Dict[str, ConstantInfo] = field(default_factory=dict)
    submodules: List[str] = field(default_factory=list)
    is_c_module: bool = False    # For C extensions
    imports: Dict[str, str] = field(default_factory=dict)  # name -> module imported from

class HierarchicalExtractor:
    """
    Hierarchical library extractor that properly handles module hierarchy,
    distinguishes between library-defined and imported objects, and properly
    categorizes submodules vs functions/classes.
    """
    def __init__(self):
        self.modules = {}  # module_path -> ModuleInfo
        self.processed_modules = set()
        self.library_prefix = ""  # Will be set to the main library name
        self.excluded_prefixes = {
            # Standard library modules to exclude
            "abc", "array", "ast", "asyncio", "audioop", "base64", "binascii", 
            "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", 
            "cmd", "code", "codecs", "codeop", "collections", "colorsys", 
            "compileall", "concurrent", "configparser", "contextlib", "contextvars", 
            "copy", "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses", 
            "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest", 
            "email", "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", 
            "fileinput", "fnmatch", "formatter", "fractions", "ftplib", "functools", 
            "gc", "getopt", "getpass", "gettext", "glob", "grp", "gzip", "hashlib", 
            "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr", 
            "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json", 
            "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", 
            "macpath", "mailbox", "mailcap", "marshal", "math", "mimetypes", 
            "mmap", "modulefinder", "msilib", "msvcrt", "multiprocessing", 
            "netrc", "nis", "nntplib", "numbers", "operator", "optparse", 
            "os", "parser", "pathlib", "pdb", "pickle", "pickletools", "pipes", 
            "pkgutil", "platform", "plistlib", "poplib", "posix", "posixpath", 
            "pprint", "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr", 
            "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib", 
            "resource", "rlcompleter", "runpy", "sched", "secrets", "select", 
            "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd", 
            "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3", 
            "ssl", "stat", "statistics", "string", "stringprep", "struct", 
            "subprocess", "sunau", "symbol", "symtable", "sys", "sysconfig", 
            "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", 
            "test", "textwrap", "threading", "time", "timeit", "tkinter", 
            "token", "tokenize", "trace", "traceback", "tracemalloc", "tty", 
            "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest", 
            "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref", 
            "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", 
            "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
            "sre_compile", "sre_constants", "sre_parse", "ntpath", "genericpath"
        }
        self.start_time = time.time()
        self.cycles_detected = 0
        self.processed_paths = set()  # Track directories we've processed for physical submodules
        
    def is_public_name(self, name: str) -> bool:
        """Check if a name follows public naming convention (doesn't start with underscore)"""
        return not name.startswith('_')
    
    def extract_library(self, library_name: str, max_depth: int = 15):
        """Extract the public namespace members from a Python library with deeper recursion"""
        self.start_time = time.time()
        self.library_prefix = library_name
        logger.info(f"Extracting namespace from: {library_name}")
        
        # Get library version
        version = get_library_version(library_name)
        logger.info(f"Detected {library_name} version: {version}")
        
        # Import the library
        try:
            library = importlib.import_module(library_name)
            
            # Create module info if not exists
            if library_name not in self.modules:
                self.modules[library_name] = ModuleInfo(
                    name=library_name,
                    docstring=library.__doc__,
                    version=version,
                    is_c_module=not hasattr(library, '__file__') or library.__file__ is None
                )
            
            # Process the root module first - which recursively processes submodules
            self._process_module(library, library_name, depth=0, max_depth=max_depth)
            
            # Second pass: process any discovered modules that weren't processed yet
            # But only if they're part of the target library
            second_pass_count = 0
            while True:
                unprocessed = [
                    module_name for module_name in self.modules.keys()
                    if module_name not in self.processed_modules and
                    module_name.startswith(f"{library_name}.")
                ]
                
                if not unprocessed:
                    break
                    
                logger.info(f"Starting second pass, found {len(unprocessed)} unprocessed modules")
                second_pass_count += 1
                
                for module_name in unprocessed:
                    try:
                        module = importlib.import_module(module_name)
                        self._process_module(module, module_name, depth=0, max_depth=max_depth)
                    except ImportError as e:
                        logger.warning(f"Could not import {module_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing {module_name}: {e}")
                
                # Avoid infinite loops
                if second_pass_count >= 3:
                    logger.warning("Reached maximum number of second passes, stopping")
                    break
            
            # Filter out modules that aren't part of the target library
            self._filter_non_library_modules()
            
            # Print summary
            elapsed = time.time() - self.start_time
            logger.info(f"Extraction completed in {elapsed:.2f} seconds")
            logger.info(f"Total library modules: {len(self.modules)}")
            logger.info(f"Processed modules: {len(self.processed_modules)}")
            logger.info(f"Cycles detected: {self.cycles_detected}")
            
            return self.modules
            
        except ImportError as e:
            logger.error(f"Error importing {library_name}: {e}")
            return {}
    
    def _filter_non_library_modules(self):
        """Filter out modules that aren't part of the target library"""
        to_remove = []
        for module_name in self.modules:
            # Keep the main library module and its submodules
            if not module_name.startswith(self.library_prefix):
                to_remove.append(module_name)
                
        for module_name in to_remove:
            del self.modules[module_name]
            
        # Update submodule references to only include existing modules
        for module_info in self.modules.values():
            module_info.submodules = [m for m in module_info.submodules if m in self.modules]
    
    def _is_library_module(self, module_name: str) -> bool:
        """Check if a module is part of the target library or should be excluded"""
        # Main check: does it start with the library name?
        if module_name.startswith(f"{self.library_prefix}.") or module_name == self.library_prefix:
            return True
            
        # Check against excluded prefixes (standard library)
        for prefix in self.excluded_prefixes:
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                return False
                
        # Default: include it if it's not in the standard library
        # This is more permissive but can be adjusted if needed
        return True
    
    def _analyze_imports(self, module_name: str, source_file: str) -> Dict[str, str]:
        """Analyze imports in a module using AST parsing"""
        imports = {}
        try:
            # Try to use astroid to parse the file
            with open(source_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Parse with astroid
            module = astroid.parse(file_content)
            
            # Look for import statements
            for node in module.get_children():
                if isinstance(node, nodes.Import):
                    for name, alias in node.names:
                        imports[alias or name] = name
                        
                elif isinstance(node, nodes.ImportFrom):
                    module_src = node.modname
                    for name, alias in node.names:
                        imports[alias or name] = f"{module_src}.{name}"
                        
                # Handle special cases like "from ... import *"
                elif isinstance(node, nodes.ImportFrom) and '*' in [name for name, alias in node.names]:
                    # This is a wildcard import, more complex to track
                    module_src = node.modname
                    imports[f"_wildcard_import_{len(imports)}"] = f"{module_src}.*"
                    
        except Exception as e:
            logger.debug(f"Error analyzing imports for {module_name}: {e}")
            
        return imports
    
    def _is_imported(self, name: str, module_info: ModuleInfo) -> Tuple[bool, str]:
        """Check if a name is imported from another module"""
        if name in module_info.imports:
            return True, module_info.imports[name]
        return False, ""
    
    def _process_module(self, module, module_name: str, depth: int = 0, max_depth: int = 15, parent_chain=None):
        """Process a module and extract its public API with deep recursion"""
        # Initialize parent chain to track cycles
        if parent_chain is None:
            parent_chain = set()
            
        # Skip if already processed or if it's a standard library module we want to ignore
        if module_name in self.processed_modules:
            return
            
        # Check if this module is part of a processing chain (cycle detection)
        if module_name in parent_chain:
            self.cycles_detected += 1
            logger.debug(f"Cycle detected: {module_name} in {parent_chain}")
            return
            
        # Skip if it's not a library module
        if not self._is_library_module(module_name) and depth > 0:
            return
            
        # Too deep?
        if depth > max_depth:
            logger.debug(f"Reached maximum depth ({max_depth}) for {module_name}")
            return
        
        # Print depth information for debugging
        indent = "  " * depth
        logger.info(f"{indent}Processing module: {module_name} (depth {depth})")
        
        # Add to processed modules
        self.processed_modules.add(module_name)
        
        # Create module info if not exists
        is_c_module = not hasattr(module, '__file__') or module.__file__ is None or module.__file__.endswith(('.so', '.pyd'))
        
        if module_name not in self.modules:
            self.modules[module_name] = ModuleInfo(
                name=module_name,
                docstring=module.__doc__,
                is_c_module=is_c_module
            )
        
        # Analyze imports if it's not a C module
        if not is_c_module and hasattr(module, '__file__') and module.__file__ is not None:
            imports = self._analyze_imports(module_name, module.__file__)
            self.modules[module_name].imports = imports
        
        # Update parent chain for cycle detection
        new_parent_chain = parent_chain.copy()
        new_parent_chain.add(module_name)
        
        # Process the module's content
        self._process_module_contents(module, module_name, depth, max_depth, new_parent_chain)
    
    def _process_module_contents(self, module, module_name: str, depth: int, max_depth: int, parent_chain):
        """Process a module's contents (attributes, submodules, etc.)"""
        # First, discover all potential submodules to ensure we capture the full structure
        self._discover_submodules(module, module_name, depth, max_depth, parent_chain)
        
        # Process attributes in __all__ if available
        module_all = getattr(module, '__all__', None)
        if module_all is not None:
            # Use __all__ to determine exported names
            for name in module_all:
                if not self.is_public_name(name):
                    continue
                self._process_attribute(module, name, module_name)
        else:
            # Fallback: use non-underscore names as public
            for name in dir(module):
                if not self.is_public_name(name):
                    continue
                self._process_attribute(module, name, module_name)
    
    def _process_attribute(self, module, name: str, module_name: str):
        """Process a single module attribute (function, class, constant, etc.)"""
        # Skip if already processed in this module
        module_info = self.modules[module_name]
        if (name in module_info.functions or name in module_info.classes or 
            name in module_info.constants or name in module_info.submodules):
            return
            
        try:
            obj = getattr(module, name)
            
            # Skip if it's None
            if obj is None:
                return
            
            # Check if it's imported
            is_imported, imported_from = self._is_imported(name, module_info)
            
            # Check type and process accordingly
            if inspect.ismodule(obj):
                # It's a submodule - already handled by discover_submodules
                return
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                # It's a function
                self._process_function(obj, name, module_name, is_imported, imported_from)
            elif inspect.isclass(obj):
                # It's a class
                self._process_class(obj, name, module_name, is_imported, imported_from)
            elif callable(obj):
                # It's some other callable object
                self._process_callable_object(obj, name, module_name, is_imported, imported_from)
            else:
                # It's a constant or other value
                self._process_constant(name, obj, module_name, is_imported, imported_from)
        except Exception as e:
            logger.debug(f"Error processing attribute {module_name}.{name}: {e}")
    
    def _discover_submodules(self, module, module_name: str, depth: int, max_depth: int, parent_chain):
        """Discover all potential submodules using multiple methods"""
        # Method 1: Check explicitly defined submodules from __all__ if available
        module_all = getattr(module, '__all__', None)
        if module_all is not None:
            for name in module_all:
                try:
                    obj = getattr(module, name)
                    if inspect.ismodule(obj):
                        self._process_submodule(obj, name, module_name, depth, max_depth, parent_chain)
                except Exception as e:
                    logger.debug(f"Error checking __all__ item {name} in {module_name}: {e}")
        
        # Method 2: Check all attributes for modules
        for name in dir(module):
            # Skip private attributes, but process public ones
            if not self.is_public_name(name):
                continue
                
            try:
                obj = getattr(module, name)
                if inspect.ismodule(obj):
                    self._process_submodule(obj, name, module_name, depth, max_depth, parent_chain)
            except Exception as e:
                logger.debug(f"Error checking attribute {name} in {module_name}: {e}")
        
        # Method 3: If it's a package, check its __path__ for physical subpackages
        if hasattr(module, '__path__') and hasattr(module, '__file__') and module.__file__:
            pkg_dir = os.path.dirname(module.__file__)
            
            # Skip if we already processed this path
            if pkg_dir in self.processed_paths:
                return
                
            self.processed_paths.add(pkg_dir)
                
            try:
                # List directories that might be subpackages
                for item in os.listdir(pkg_dir):
                    # Skip __pycache__ and private packages
                    if item.startswith('_') or not os.path.isdir(os.path.join(pkg_dir, item)):
                        continue
                        
                    # Check if it has an __init__.py file (making it a package)
                    init_path = os.path.join(pkg_dir, item, '__init__.py')
                    if os.path.exists(init_path):
                        # Construct the module name
                        submodule_name = f"{module_name}.{item}"
                        
                        # Try to import it
                        try:
                            submodule = importlib.import_module(submodule_name)
                            self._process_submodule(submodule, item, module_name, depth, max_depth, parent_chain)
                        except ImportError as e:
                            logger.debug(f"Could not import {submodule_name}: {e}")
                        except Exception as e:
                            logger.debug(f"Error processing {submodule_name}: {e}")
            except Exception as e:
                logger.debug(f"Error examining files in {pkg_dir}: {e}")
    
    def _process_submodule(self, submodule, name: str, parent_module: str, depth: int, max_depth: int, parent_chain):
        """Process a single submodule and schedule it for full extraction"""
        submodule_name = submodule.__name__
        
        # Check if it's a standard library module we want to ignore
        if not self._is_library_module(submodule_name):
            return
        
        # Add to parent's submodules list if not already there
        if submodule_name not in self.modules[parent_module].submodules:
            self.modules[parent_module].submodules.append(submodule_name)
        
        # Add to modules dict if not exists
        if submodule_name not in self.modules:
            self.modules[submodule_name] = ModuleInfo(
                name=submodule_name,
                docstring=submodule.__doc__,
                is_c_module=not hasattr(submodule, '__file__') or submodule.__file__ is None
            )
        
        # Process the submodule if not already processed
        if submodule_name not in self.processed_modules:
            try:
                self._process_module(submodule, submodule_name, depth + 1, max_depth, parent_chain)
            except Exception as e:
                logger.warning(f"Error recursively processing {submodule_name}: {e}")
    
    def _process_function(self, func, name: str, module_name: str, is_imported: bool = False, imported_from: str = ""):
        """Process a function object"""
        # Skip private functions
        if not self.is_public_name(name):
            return
            
        # Basic info
        is_c_func = inspect.isbuiltin(func) or not inspect.isfunction(func)
        
        func_info = FunctionInfo(
            name=name,
            module=module_name,
            docstring=func.__doc__,
            is_c_function=is_c_func,
            is_imported=is_imported,
            imported_from=imported_from
        )
        
        # Try to get signature
        try:
            signature = inspect.signature(func)
            func_info.args = [param for param in signature.parameters]
            func_info.signature = f"def {name}{signature}"
        except (TypeError, ValueError):
            # Can't get signature for some built-in functions
            func_info.signature = f"def {name}(*args, **kwargs)"
        
        # Try to get source info
        if not is_c_func:
            try:
                source_file = inspect.getsourcefile(func)
                source_lines, start_line = inspect.getsourcelines(func)
                func_info.line_number = start_line
                func_info.end_line = start_line + len(source_lines) - 1
                
                # Try to get more detailed info using astroid
                self._enhance_function_with_astroid(func, func_info, source_file)
            except (TypeError, OSError, IOError):
                pass
        
        # Store the function
        self.modules[module_name].functions[name] = func_info
    
    def _process_class(self, cls, name: str, module_name: str, is_imported: bool = False, imported_from: str = ""):
        """Process a class object"""
        # Skip private classes
        if not self.is_public_name(name):
            return
            
        # Basic info
        is_c_class = not inspect.isclass(cls) or not hasattr(cls, '__module__')
        
        class_info = ClassInfo(
            name=name,
            docstring=cls.__doc__,
            bases=[base.__name__ for base in cls.__bases__ if base.__name__ != 'object'],
            is_c_class=is_c_class,
            is_imported=is_imported,
            imported_from=imported_from
        )
        
        # Get methods and attributes
        for member_name, member in inspect.getmembers(cls):
            # Skip private members (including special methods)
            if not self.is_public_name(member_name):
                continue
            
            if inspect.isfunction(member) or inspect.ismethod(member) or inspect.ismethoddescriptor(member):
                # Process method
                method_info = FunctionInfo(
                    name=member_name,
                    module=module_name,
                    docstring=member.__doc__,
                    is_method=True,
                    is_c_function=not inspect.isfunction(member),
                    is_imported=is_imported,
                    imported_from=imported_from
                )
                
                # Try to get signature
                try:
                    signature = inspect.signature(member)
                    method_info.args = [param for param in signature.parameters]
                    method_info.signature = f"def {member_name}{signature}"
                except (TypeError, ValueError):
                    # Can't get signature
                    method_info.signature = f"def {member_name}(self, *args, **kwargs)"
                
                class_info.methods[member_name] = method_info
            
            elif not callable(member) and not inspect.isclass(member) and not inspect.ismodule(member):
                # Process attribute (if it's not callable and not a class/module)
                class_info.attributes.append(member_name)
        
        # Try to get source info
        if not is_c_class:
            try:
                source_file = inspect.getsourcefile(cls)
                source_lines, start_line = inspect.getsourcelines(cls)
                class_info.line_number = start_line
                class_info.end_line = start_line + len(source_lines) - 1
                
                # Try to get more detailed info using astroid
                self._enhance_class_with_astroid(cls, class_info, source_file)
            except (TypeError, OSError, IOError):
                pass
        
        # Store the class
        self.modules[module_name].classes[name] = class_info
    
    def _process_constant(self, name: str, value: Any, module_name: str, is_imported: bool = False, imported_from: str = ""):
        """Process a constant"""
        # Skip private constants
        if not self.is_public_name(name):
            return
            
        # Store the constant
        self.modules[module_name].constants[name] = ConstantInfo(
            name=name,
            value_type=type(value).__name__,
            is_imported=is_imported,
            imported_from=imported_from
        )
    
    def _process_callable_object(self, obj, name: str, module_name: str, is_imported: bool = False, imported_from: str = ""):
        """Process callable object that's not a function or class"""
        # Skip private objects
        if not self.is_public_name(name):
            return
            
        # Treat as a function
        func_info = FunctionInfo(
            name=name,
            module=module_name,
            docstring=getattr(obj, '__doc__', None),
            is_c_function=True,
            is_imported=is_imported,
            imported_from=imported_from
        )
        
        # Try to infer signature from docstring
        if hasattr(obj, '__doc__') and obj.__doc__:
            # Simple heuristic: look for "Parameters" section
            params_section = obj.__doc__.split('Parameters\n----------\n')
            if len(params_section) > 1:
                params_lines = params_section[1].split('\n\n')[0].strip().split('\n')
                args = []
                for line in params_lines:
                    parts = line.strip().split(':', 1)
                    if len(parts) > 0:
                        args.append(parts[0].strip())
                
                func_info.args = args
                func_info.signature = f"def {name}({', '.join(args)})"
            else:
                func_info.signature = f"def {name}(*args, **kwargs)"
        else:
            func_info.signature = f"def {name}(*args, **kwargs)"
        
        # Store the function
        self.modules[module_name].functions[name] = func_info
    
    def _enhance_function_with_astroid(self, func, func_info: FunctionInfo, source_file: str):
        """Try to enhance function info using astroid"""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Calculate complexity
            complexity_visitor = ComplexityVisitor.from_code(file_content)
            complexity_dict = {item.name: item.complexity for item in complexity_visitor.functions}
            
            # Parse with astroid
            module = astroid.parse(file_content)
            
            # Find the function node
            func_node = None
            for node in module.get_children():
                if isinstance(node, nodes.FunctionDef) and node.name == func_info.name:
                    func_node = node
                    break
            
            if func_node:
                # Update complexity
                func_info.complexity = complexity_dict.get(func_info.name, 0)
                
                # Check if it's an async function
                func_info.is_async = isinstance(func_node, nodes.AsyncFunctionDef)
                
                # Get return annotation if available
                if func_node.returns:
                    if isinstance(func_node.returns, nodes.Name):
                        func_info.returns = func_node.returns.name
                    elif isinstance(func_node.returns, nodes.Attribute):
                        func_info.returns = func_node.returns.as_string()
                    else:
                        func_info.returns = str(func_node.returns)
                
                # Get decorators
                if func_node.decorators:
                    for decorator in func_node.decorators.nodes:
                        if isinstance(decorator, nodes.Name):
                            func_info.decorators.append(decorator.name)
                        elif isinstance(decorator, nodes.Call):
                            if isinstance(decorator.func, nodes.Name):
                                func_info.decorators.append(decorator.func.name)
                            else:
                                func_info.decorators.append(decorator.func.as_string())
                        else:
                            func_info.decorators.append(decorator.as_string())
        except Exception as e:
            # Silently fail if we can't enhance
            pass
    
    def _enhance_class_with_astroid(self, cls, class_info: ClassInfo, source_file: str):
        """Try to enhance class info using astroid"""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse with astroid
            module = astroid.parse(file_content)
            
            # Find the class node
            class_node = None
            for node in module.get_children():
                if isinstance(node, nodes.ClassDef) and node.name == class_info.name:
                    class_node = node
                    break
            
            if class_node:
                # Get decorators
                if class_node.decorators:
                    for decorator in class_node.decorators.nodes:
                        if isinstance(decorator, nodes.Name):
                            class_info.decorators.append(decorator.name)
                        elif isinstance(decorator, nodes.Call):
                            if isinstance(decorator.func, nodes.Name):
                                class_info.decorators.append(decorator.func.name)
                            else:
                                class_info.decorators.append(decorator.func.as_string())
                        else:
                            class_info.decorators.append(decorator.as_string())
        except Exception as e:
            # Silently fail if we can't enhance
            pass

    def print_extraction_tree(self, root_module: str = None, max_depth: int = None, current_depth: int = 0):
        """Print the extraction tree to visualize the hierarchy"""
        if current_depth == 0:
            logger.info("Module hierarchy:")
            
        if max_depth is not None and current_depth > max_depth:
            return
            
        if root_module is None:
            # Find the top-level modules (those without dots)
            top_modules = [name for name in self.modules.keys() if '.' not in name]
            for module in sorted(top_modules):
                self.print_extraction_tree(module, max_depth, 0)
            return
        
        indent = "  " * current_depth
        if root_module not in self.modules:
            logger.info(f"{indent}Module {root_module} not found in extraction results")
            return
        
        module_info = self.modules[root_module]
        
        # Print module info
        func_count = len(module_info.functions)
        class_count = len(module_info.classes)
        const_count = len(module_info.constants)
        submodule_count = len(module_info.submodules)
        
        logger.info(f"{indent}Module: {root_module} ({func_count} functions, {class_count} classes, {const_count} constants, {submodule_count} submodules)")
        
        # Count library-defined vs imported functions
        library_funcs = sum(1 for f in module_info.functions.values() if not f.is_imported)
        imported_funcs = func_count - library_funcs
        
        # Print function names for this module
        if library_funcs > 0:
            # Filter out imported functions
            lib_func_names = [name for name, func in module_info.functions.items() if not func.is_imported]
            if len(lib_func_names) <= 10:
                logger.info(f"{indent}  Library Functions: {', '.join(sorted(lib_func_names))}")
            else:
                logger.info(f"{indent}  Library Functions: {', '.join(sorted(lib_func_names)[:10])} and {len(lib_func_names) - 10} more...")
        
        # Recursively print submodules, but avoid cycles
        visited_submodules = set()
        for submodule in sorted(module_info.submodules):
            if submodule in visited_submodules or submodule == root_module:
                continue
                
            visited_submodules.add(submodule)
            self.print_extraction_tree(submodule, max_depth, current_depth + 1)

    def export_json(self, output_file: str):
        """Export the extracted information to a JSON file with safe serialization"""
        import json
        
        # Define a custom JSON encoder to handle non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    # First try the default encoder
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    # If that fails, return a string representation
                    return str(obj)
        
        # Create a serializable dictionary manually instead of using asdict
        data = {}
        for module_name, module_info in self.modules.items():
            module_dict = {
                "name": module_info.name,
                "docstring": module_info.docstring,
                "version": module_info.version,
                "is_c_module": module_info.is_c_module,
                "submodules": module_info.submodules,
                "imports": module_info.imports,
                
                # Functions
                "functions": {
                    name: {
                        "name": func.name,
                        "module": func.module,
                        "args": func.args,
                        "returns": func.returns,
                        "docstring": func.docstring,
                        "decorators": func.decorators,
                        "complexity": func.complexity,
                        "line_number": func.line_number,
                        "end_line": func.end_line,
                        "is_method": func.is_method,
                        "is_async": func.is_async,
                        "signature": func.signature,
                        "is_c_function": func.is_c_function,
                        "is_imported": func.is_imported,
                        "imported_from": func.imported_from
                    }
                    for name, func in module_info.functions.items()
                },
                
                # Classes
                "classes": {
                    name: {
                        "name": cls.name,
                        "bases": cls.bases,
                        "docstring": cls.docstring,
                        "decorators": cls.decorators,
                        "line_number": cls.line_number,
                        "end_line": cls.end_line,
                        "is_c_class": cls.is_c_class,
                        "is_imported": cls.is_imported,
                        "imported_from": cls.imported_from,
                        "attributes": cls.attributes,
                        
                        # Methods
                        "methods": {
                            method_name: {
                                "name": method.name,
                                "module": method.module,
                                "args": method.args,
                                "returns": method.returns,
                                "docstring": method.docstring,
                                "decorators": method.decorators,
                                "complexity": method.complexity,
                                "line_number": method.line_number,
                                "end_line": method.end_line,
                                "is_method": method.is_method,
                                "is_async": method.is_async,
                                "signature": method.signature,
                                "is_c_function": method.is_c_function,
                                "is_imported": method.is_imported,
                                "imported_from": method.imported_from
                            }
                            for method_name, method in cls.methods.items()
                        }
                    }
                    for name, cls in module_info.classes.items()
                },
                
                # Constants
                "constants": {
                    name: {
                        "name": const.name,
                        "value_type": const.value_type,
                        "is_imported": const.is_imported,
                        "imported_from": const.imported_from
                    }
                    for name, const in module_info.constants.items()
                }
            }
            
            data[module_name] = module_dict
        
        # Write to file with the custom encoder
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=CustomEncoder)
        
        logger.info(f"Data exported to {output_file}")

    def export_summary(self, output_file: str):
        """Export a summary of the extraction to a text file without recursion issues"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Library Extraction Summary\n")
            f.write(f"=======================\n\n")
            
            # Total counts
            total_modules = len(self.modules)
            total_functions = sum(len(m.functions) for m in self.modules.values())
            library_functions = sum(
                sum(1 for func in m.functions.values() if not func.is_imported)
                for m in self.modules.values()
            )
            imported_functions = total_functions - library_functions
            
            total_classes = sum(len(m.classes) for m in self.modules.values())
            library_classes = sum(
                sum(1 for cls in m.classes.values() if not cls.is_imported)
                for m in self.modules.values()
            )
            imported_classes = total_classes - library_classes
            
            f.write(f"Total modules: {total_modules}\n")
            f.write(f"Total functions: {total_functions} (Library: {library_functions}, Imported: {imported_functions})\n")
            f.write(f"Total classes: {total_classes} (Library: {library_classes}, Imported: {imported_classes})\n\n")
            
            # Module listing with counts
            f.write(f"Module Hierarchy\n")
            f.write(f"===============\n\n")
            
            # Non-recursive implementation to avoid max recursion depth issues
            module_stack = [(name, 0) for name in sorted(self.modules.keys()) if '.' not in name]
            visited = set()
            
            while module_stack:
                module_name, depth = module_stack.pop(0)
                
                if module_name in visited:
                    continue
                    
                visited.add(module_name)
                
                # Get module info
                if module_name not in self.modules:
                    continue
                    
                module_info = self.modules[module_name]
                indent = "  " * depth
                
                # Count library-defined vs imported functions
                library_funcs = [
                    name for name, func in module_info.functions.items() 
                    if not func.is_imported and self.is_public_name(name)
                ]
                imported_funcs = [
                    name for name, func in module_info.functions.items() 
                    if func.is_imported and self.is_public_name(name)
                ]
                
                # Write module info
                f.write(f"{indent}{module_name} ({len(library_funcs)} library functions, {len(imported_funcs)} imported functions)\n")
                
                # Write function names if there are any
                if library_funcs:
                    f.write(f"{indent}  Library Functions: {', '.join(sorted(library_funcs))}\n")
                
                # Write submodule list
                if module_info.submodules:
                    f.write(f"{indent}  Submodules: {', '.join(sorted(module_info.submodules))}\n")
                
                # Add submodules to stack
                submodules = [(m, depth + 1) for m in sorted(module_info.submodules) 
                              if m != module_name and m not in visited]
                module_stack = submodules + module_stack
            
        logger.info(f"Summary exported to {output_file}")

    def export_rdf(self, output_file: str, format: str = 'turtle', include_imported=False):
        """
        Export the extracted information to RDF format
        
        Args:
            output_file: Path to output file
            format: RDF format (turtle, xml, json-ld, etc.)
            include_imported: Whether to include imported functions/classes in the output
        """
        try:
            import rdflib
            from rdflib import Graph, Namespace, Literal, BNode, URIRef
            from rdflib.namespace import RDF, RDFS, XSD
            
            # Initialize RDF graph
            g = Graph()
            
            # Define namespaces
            PYLIB = Namespace("http://example.org/pylib/")
            g.bind("pylib", PYLIB)
            g.bind("rdfs", RDFS)
            
            # Add extracted information to the graph
            for module_name, module_info in self.modules.items():
                # Create proper URI for the module
                module_uri = PYLIB[module_name.replace('.', '_')]
                g.add((module_uri, RDF.type, PYLIB.Module))
                g.add((module_uri, RDFS.label, Literal(module_name)))
                
                if module_info.docstring:
                    g.add((module_uri, RDFS.comment, Literal(module_info.docstring)))
                
                # Add version information
                g.add((module_uri, PYLIB.hasVersion, Literal(module_info.version)))
                g.add((module_uri, PYLIB.isCModule, Literal(module_info.is_c_module, datatype=XSD.boolean)))
                
                # Add submodules - IMPORTANT: Use hasSubmodule, not defines
                for submodule in module_info.submodules:
                    if submodule in self.modules:  # Only reference modules we actually extracted
                        submodule_uri = PYLIB[submodule.replace('.', '_')]
                        g.add((submodule_uri, RDF.type, PYLIB.Module))
                        g.add((submodule_uri, RDFS.label, Literal(submodule)))
                        g.add((module_uri, PYLIB.hasSubmodule, submodule_uri))
                
                # Add functions - only public ones and optionally filter out imported ones
                for name, func_info in module_info.functions.items():
                    # Skip if it's a private function or imported (if exclude_imported is True)
                    if not self.is_public_name(name) or (func_info.is_imported and not include_imported):
                        continue
                        
                    func_uri = PYLIB[f"{module_name.replace('.', '_')}_{name}"]
                    g.add((func_uri, RDF.type, PYLIB.Function))
                    g.add((func_uri, RDFS.label, Literal(name)))
                    
                    # Add function properties
                    if func_info.docstring:
                        g.add((func_uri, RDFS.comment, Literal(func_info.docstring)))
                    
                    g.add((func_uri, PYLIB.hasComplexity, Literal(func_info.complexity, datatype=XSD.integer)))
                    g.add((func_uri, PYLIB.lineNumber, Literal(func_info.line_number, datatype=XSD.integer)))
                    g.add((func_uri, PYLIB.endLine, Literal(func_info.end_line, datatype=XSD.integer)))
                    g.add((func_uri, PYLIB.isMethod, Literal(func_info.is_method, datatype=XSD.boolean)))
                    g.add((func_uri, PYLIB.isAsync, Literal(func_info.is_async, datatype=XSD.boolean)))
                    g.add((func_uri, PYLIB.isCFunction, Literal(func_info.is_c_function, datatype=XSD.boolean)))
                    g.add((func_uri, PYLIB.signature, Literal(func_info.signature)))
                    
                    # Add import information
                    g.add((func_uri, PYLIB.isImported, Literal(func_info.is_imported, datatype=XSD.boolean)))
                    if func_info.is_imported:
                        g.add((func_uri, PYLIB.importedFrom, Literal(func_info.imported_from)))
                    
                    # Connect to module using 'defines', not 'hasSubmodule'
                    g.add((module_uri, PYLIB.defines, func_uri))
                    
                    # Add arguments
                    for arg in func_info.args:
                        arg_uri = BNode()
                        g.add((arg_uri, RDF.type, PYLIB.Parameter))
                        g.add((arg_uri, RDFS.label, Literal(arg)))
                        g.add((func_uri, PYLIB.hasParameter, arg_uri))
                    
                    # Add decorators
                    for decorator in func_info.decorators:
                        dec_uri = BNode()
                        g.add((dec_uri, RDF.type, PYLIB.Decorator))
                        g.add((dec_uri, RDFS.label, Literal(decorator)))
                        g.add((func_uri, PYLIB.hasDecorator, dec_uri))
                
                # Add classes - only public ones and optionally filter out imported ones
                for name, class_info in module_info.classes.items():
                    # Skip if it's a private class or imported (if exclude_imported is True)
                    if not self.is_public_name(name) or (class_info.is_imported and not include_imported):
                        continue
                        
                    class_uri = PYLIB[f"{module_name.replace('.', '_')}_{name}"]
                    g.add((class_uri, RDF.type, PYLIB.Class))
                    g.add((class_uri, RDFS.label, Literal(name)))
                    
                    # Add class properties
                    if class_info.docstring:
                        g.add((class_uri, RDFS.comment, Literal(class_info.docstring)))
                    
                    g.add((class_uri, PYLIB.lineNumber, Literal(class_info.line_number, datatype=XSD.integer)))
                    g.add((class_uri, PYLIB.endLine, Literal(class_info.end_line, datatype=XSD.integer)))
                    g.add((class_uri, PYLIB.isCClass, Literal(class_info.is_c_class, datatype=XSD.boolean)))
                    
                    # Add import information
                    g.add((class_uri, PYLIB.isImported, Literal(class_info.is_imported, datatype=XSD.boolean)))
                    if class_info.is_imported:
                        g.add((class_uri, PYLIB.importedFrom, Literal(class_info.imported_from)))
                    
                    # Connect to module using 'defines', not 'hasSubmodule'
                    g.add((module_uri, PYLIB.defines, class_uri))
                    
                    # Add base classes
                    for base in class_info.bases:
                        base_uri = PYLIB[base.replace('.', '_')]
                        g.add((base_uri, RDF.type, PYLIB.Class))
                        g.add((base_uri, RDFS.label, Literal(base)))
                        g.add((class_uri, PYLIB.inheritsFrom, base_uri))
                    
                    # Add attributes - only public ones
                    for attr in class_info.attributes:
                        # Skip if it's a private attribute
                        if not self.is_public_name(attr):
                            continue
                            
                        attr_uri = BNode()
                        g.add((attr_uri, RDF.type, PYLIB.Attribute))
                        g.add((attr_uri, RDFS.label, Literal(attr)))
                        g.add((class_uri, PYLIB.hasAttribute, attr_uri))
                    
                    # Add methods - only public ones
                    for method_name, method_info in class_info.methods.items():
                        # Skip if it's a private method
                        if not self.is_public_name(method_name):
                            continue
                            
                        method_uri = PYLIB[f"{module_name.replace('.', '_')}_{name}_{method_name}"]
                        g.add((method_uri, RDF.type, PYLIB.Method))
                        g.add((method_uri, RDFS.label, Literal(method_name)))
                        
                        if method_info.docstring:
                            g.add((method_uri, RDFS.comment, Literal(method_info.docstring)))
                        
                        g.add((method_uri, PYLIB.signature, Literal(method_info.signature)))
                        g.add((method_uri, PYLIB.isCFunction, Literal(method_info.is_c_function, datatype=XSD.boolean)))
                        
                        g.add((class_uri, PYLIB.hasMethod, method_uri))
                        
                        # Add method arguments
                        for arg in method_info.args:
                            arg_uri = BNode()
                            g.add((arg_uri, RDF.type, PYLIB.Parameter))
                            g.add((arg_uri, RDFS.label, Literal(arg)))
                            g.add((method_uri, PYLIB.hasParameter, arg_uri))
                
                # Add constants - only public ones and optionally filter out imported ones
                for name, const_info in module_info.constants.items():
                    # Skip if it's a private constant or imported (if exclude_imported is True)
                    if not self.is_public_name(name) or (const_info.is_imported and not include_imported):
                        continue
                        
                    const_uri = PYLIB[f"{module_name.replace('.', '_')}_{name}"]
                    g.add((const_uri, RDF.type, PYLIB.Constant))
                    g.add((const_uri, RDFS.label, Literal(name)))
                    g.add((const_uri, PYLIB.hasType, Literal(const_info.value_type)))
                    
                    # Add import information
                    g.add((const_uri, PYLIB.isImported, Literal(const_info.is_imported, datatype=XSD.boolean)))
                    if const_info.is_imported:
                        g.add((const_uri, PYLIB.importedFrom, Literal(const_info.imported_from)))
                    
                    # Connect to module using 'defines', not 'hasSubmodule'
                    g.add((module_uri, PYLIB.defines, const_uri))
            
            # Export the graph
            g.serialize(destination=output_file, format=format)
            logger.info(f"RDF graph exported to {output_file} in {format} format")
            
        except ImportError:
            logger.error("RDFLib is required for RDF export. Please install it using 'pip install rdflib'")

# Helper function from original code
def get_library_version(library_name: str) -> str:
    """
    Get the version of a library using multiple methods.
    Tries different approaches silently and returns "Unknown" if all fail.
    
    Args:
        library_name: The name of the library to get the version for
        
    Returns:
        The version string or "Unknown" if version can't be determined
    """
    # Method 1: Use importlib.metadata (Python 3.8+)
    try:
        return str(importlib.metadata.version(library_name))
    except Exception:
        pass

    # Method 2: Use pkg_resources
    try:
        return str(pkg_resources.get_distribution(library_name).version)
    except Exception:
        pass

    # Method 3: Try importing and checking various version attributes
    try:
        module = importlib.import_module(library_name)
        for attr in ['__version__', 'VERSION', 'version', '_version_', 'Version']:
            try:
                version = getattr(module, attr)
                if version and isinstance(version, (str, int, float)):
                    return str(version)
            except Exception:
                continue
            
        # Method 4: Try checking package metadata if available
        try:
            if hasattr(module, '__package__') and module.__package__:
                return str(importlib.metadata.version(module.__package__))
        except Exception:
            pass
            
    except Exception:
        pass

    return "Unknown"

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hierarchical_extractor.py <library_name>")
        sys.exit(1)
    
    library_name = sys.argv[1]
    extractor = HierarchicalExtractor()
    extractor.extract_library(library_name)
    
    # Print the extraction tree
    extractor.print_extraction_tree(max_depth=3)  # limit depth for readability
    
    # Export as JSON
    extractor.export_json(f"{library_name}_api334.json")
    
    # Export a human-readable summary
    extractor.export_summary(f"{library_name}_summary334.txt")
    
    # Export as RDF (optional, requires rdflib)
    try:
        # Export without imported objects (only true library API)
        extractor.export_rdf(f"{library_name}_api334.ttl", format="turtle", include_imported=False)
    except ImportError:
        print("RDFLib not installed, skipping RDF export")
