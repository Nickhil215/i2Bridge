from dataclasses import dataclass
from typing import Dict, Optional
from typing import List


@dataclass
class FunctionInfo:
    """Store detailed information about a function or method."""
    id: str
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