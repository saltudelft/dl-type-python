from importlab import environment, fs, graph, import_finder, output, parsepy, resolve, utils
from argparse import Namespace

import argparse
import sys, inspect, os

from functools import reduce
from typing import NewType
from queue import Queue


class ModuleExtractor():
    def __init__(self):
        pass

    def get_file_base_name(self, file):
        """
        Retrieves the name of the specified file, dropping the path information
        for the file as well as the extension of the file.
        :param: file  Full file path
        :return: file base name (without extension)
        """
        # Drop path from string to get the basename of the file
        base_name = os.path.basename(file)

        # Drop extension from the file
        # TODO: This might be a bit problematic with files which only start with a dot
        # TODO: e.g. .file (which could still be a valid python script, just with the wrong extension)
        base_name = base_name.split(".")[0]

        return base_name

    def get_imports(self, file):
        """
        Returns the import statements for the specified file, which contains information
        of the import name, the alias of the import, whether it is an import from or not,
        whether the import is a star import and the full path to the import.
        The method checks all transitive imports as well, and returns the import statements
        of the transitive imports as well.

        :param: file  Name of the file
        :return: List of ImportStatement objects containing information about imports
        """
        # The commented out code returns tuples instead; ImportStatement objects are easier to work with,
        # so we use the parsepy alternative.
        #imports = import_finder.get_imports(file)

        # Get imports from the specified file and the Python version of the system
        # imports = parsepy.get_imports(file, sys.version_info[:2])

        # The commented out code also adds the file that we are examining as an import
        # to retrieve the type definitions here. However, this is commented out as this
        # is done in the AST parsing step.
        # file_import_name = self.get_file_base_name(file)
        # file_import = parsepy.ImportStatement(file_import_name, None, False, False, file)
        # imports.append(file_import)

        # TODO: This could (and should) potentially be optimized to (perhaps) generate the
        # TODO: dependency tree only once and cache it somewhere.

        # TODO: Make the try-catch less generic.
        try:
            # Create queue to perform BFS; This will hold file paths
            # We also create a set of files that we already checked so that we do not
            # re-visit the same file more than once.
            file_queue = Queue()
            checked_files = set()

            # Add root file that we are analyzing; We use abspath to convert the potentially
            # relative path to an absolute one.
            root_file = os.path.abspath(file)
            file_queue.put(root_file)
            checked_files.add(root_file)

            # Final import statements to return
            import_statements = []

            # Iterate while queue is not empty (we will break in loop when needed)
            while not file_queue.empty():
                current_file = file_queue.get()

                # Get import statements for current file
                current_imports = self.get_file_import_statements(current_file)

                # Add all current import statements to final result
                import_statements += current_imports

                for stmt in current_imports:
                    # Get base name of import (the import name before the final dot)
                    base_name = self.get_file_base_name(stmt.name)

                    # Keep track whether we should add this file for recursive analysis
                    # of imports. If the base name of the import is equal to the import name,
                    # this means that we imported a full module, and should indeed process it.
                    # Otherwise, we resolve this to False.
                    add_path = base_name == stmt.name

                    if (not add_path):
                        try:
                            # Resolve base import (without the import portion after the last dot)
                            base_import = import_finder.resolve_import(base_name, False, False)

                            # Paths un-equal; this means we imported a module.
                            # As such, we should analyze this file recursively.
                            add_path = base_import.path != stmt.source
                        except:
                            continue
                    
                    # Add file for further analysis if marked as such, if the statement source is resolved,
                    # and if the current file is not visited.
                    if (add_path and stmt.source is not None and stmt.source not in checked_files):
                        file_queue.put(stmt.source)    # Add file to BFS queue
                        checked_files.add(stmt.source) # Mark file as 'visited'

            # Return final import statements list
            return import_statements

            # ImportLab graph approach; Issue with this is that the graph built
            # does not distinguish from imports, so we have to do it manually.
            # Create environment & arguments used for creating graph
            # default_version = '%d.%d' % sys.version_info[:2]
            # args = Namespace(inputs=[file], python_version=default_version, pythonpath='')
            # env = environment.create_from_args(args)

            # # Create import graph
            # importGraph = graph.ImportGraph.create(env, args.inputs, True)

            # # Prints unresolved imports
            # #print(importGraph.get_all_unresolved())
            
            # # Get a topologicgally sorted list of files
            # import_statements = importGraph.sorted_source_files()

            # # Since the sorted source files returns a list of lists, we concatenate them
            # # to form a coherent list of paths.
            # import_statements = reduce(lambda a,b: a + b, import_statements)
            
            # # Finally, we want to get the import statements for each of the dependency files.
            # # We do this by reducing the list of paths to first get the imports for that file,
            # # and then we concatenate each of the import statement lists together to finally form
            # # a coherent list of import statements.

            # # Helper function to combine two paths (or one list of ImportStatements and one path)
            # # to a coherent list of ImportStatements
            # def reduce_paths(p1, p2):
            #     p2_statements = self.get_file_import_statements(p2)
                
            #     # Combine previous import statements with path 2 import statements
            #     return p1 + p2_statements


            # # Convert first import statement to file import statements so that
            # # the reduce operation below works as expected.
            # if (len(import_statements) > 0):
            #     import_statements[0] = self.get_file_import_statements(import_statements[0])

            # import_statements = reduce(reduce_paths, import_statements)

            # return import_statements
        except:
            return []



    def get_module(self, import_name):
        """
        Resolves the specified import to a module/package, or returns None
        if that is not possible.

        Example import name: scikit.linalg

        :param: import_name  Name of the import (full import name) to resolve
        :return: Module object or None
        """
        # Old code to load module from system
        # Module not in system; We load the module.
        # if (name not in sys.modules):
        #     spec = importlib.util.spec_from_file_location(name, path)
        #     module = importlib.util.module_from_spec(spec)
        #     spec.loader.exec_module(module)
        # else:
        #     # Module in system; Retrieve directly
        #     module = sys.modules[name]

        try:
            # Check whether the import entry is a module. It is possible that the
            # import statement refers to an import of a module member (e.g. function or class),
            # in which case the import will be resolved to 'None'. Otherwise, the import
            # is resolved to a non-None path. Note that the path will be resolved correctly
            # if the import is a star statement (i.e. it will return a non-None path)
            is_module = import_finder._resolve_import(import_name) is not None
            
            # Import is a module if it has a path
            if (is_module):
                # Retrieve the package given by the import name. We have to split by the '.'
                # to conform to the function call's signature.
                # Note that the package will be found, since the _resolve_import method above
                # will load the module as well, and if a path is returned, we will get the package.
                i, module = import_finder._find_package(import_name.split("."))
                return module
            else:
                return None
        except:
            # Resolving imports/finding package may fail in certain scenarios
            return None

    def get_from_import_type(self, import_name, import_from):
        """
        Checks whether the specified import_from substring refers to a type in
        the fully qualified import name string.
        Returns the inspect member if that is the case, and None otherwise.

        Example:
        typing.Dict: typing.Dict
        None: scikit.linalg

        :param: import_name  Full import name
        :param: import_from  From import portion of the full import name
        :return: member if import_from refers to type, None otherwise
        """

        # Get the length offset of the import_from name + 1 (to account for the dot before the new name)
        offset = len(import_from)+1

        # Import cannot be resolved, as new_name == name
        if (len(import_name) < offset):
            return None

        # Remove the new_name from the name
        # Attempt to resolve a module from the base name
        base_name = import_name[0:-offset]
        module = self.get_module(base_name)

        try:
            # Check whether the import from name is actually an attribute
            # in the module that we just imported (provided that the importing was successful).
            # If new_name is found as an attribute, that means it is a member (function, class, etc.)
            # in the module.
            if (module is not None and hasattr(module, import_from)):
                attribute = getattr(module, import_from)
                is_valid_type = self.is_type_member(attribute)

                if (is_valid_type):
                    # The name of the type is the 'from' portion of the
                    # import (the last substring of the import right after
                    # the final dot)
                    return (import_from, attribute.__module__)
        except:
            # Checking for attribute in module might result in errors
            pass

        # No 'additional' import type determined
        return None

    def resolve_modules(self, file):
        """
        The method resolves modules imported by the file by loading the modules and returning
        the loaded modules in the form of a dictionary.

        The method also takes care of handling special cases of modules where a direct type is imported,
        for example: "from typing import Dict". To account for this, the method also returns a set of types
        that were resolved to be actual types rather than packages.

        The method thus returns a tuple of two elements, the first element being a dictionary of imports with
        key as the full module name (e.g. numpy if numpy is imported, or scikit.linalg if the submodule is imported
        instead), and the value is the loaded module object itself. The second element is a set of resolved types described earlier.


        :param:   file  File to resolve modules for (provided as a path)
        :return:  Tuple in the form: (imports, resolved_types)
        """
        # Retrieve the imports for the provided file in the form of a list of ImportStatements.
        imports = self.get_imports(file)

        # Keep track of the modules in the form (module_name, module_object)
        modules = dict()

        # Keep track of extra imported types that originate from the import from statements.
        imported_types = set()

        for import_entry in imports:
            name = import_entry.name
            module = self.get_module(name)

            # Import successfully resolved to a module; We store this module in our created modules dictionary.
            if (module is not None):
                modules[name] = module
            else:
                # Import not a module; Check whether the from import refers to a type.
                # If it does, add it to the extra imported types set.
                from_type = self.get_from_import_type(import_entry.name, import_entry.new_name)

                if (from_type is not None):
                    imported_types.add(from_type)

        # TODO: Return a special object/class instead of a tuple
        return (modules, imported_types)

    def get_types(self, file):
        """
        Returns a set of types visible by the file.

        :param: file  File path to get visible types for
        :return: Set of types (provided as names) visible by the Python file.
        """
        # Retrieve the resolved modules and additional types
        modules, additional_types = self.resolve_modules(file)

        # Construct fully qualified module strings from additional types
        additional_type_set = set([m[1] + "." + m[0] for m in additional_types])

        # Keep a set to ignore duplicates; Add initial additional types
        # resolved from the module imports.
        types = set() | additional_type_set

        for module_name in modules:
            # Get the module from the dictionary, and extract the types from the module.
            module = modules[module_name]
            module_types = self.get_types_module(module)

            # Merge the types set with the retrieved module types
            types.update(module_types)
        
        return types

    def get_file_import_statements(self, file):
        """
        Gets the import statements for the specified file. Does not check imports of imports.

        :param: file  filename
        :return: List of ImportStatement objects
        """
        
        if (file.endswith(".py")):
            try:
                return parsepy.get_imports(file, sys.version_info[:2])
            except:
                # There are cases where templating is used in python in the form of {%- ... %}, which
                # causes the importlab import parser to fail. This can either be alleviated by
                # building another static analyzer that will check for import statements, or simply
                # opting to not return visible types.
                # See example: aio-libs/create-aio-app/create_aio_app/template/{{cookiecutter.project_name}}/conftest.py
                return []
        else:
            # There are cases where we might be analyzing a .pyd, .pyc or .pyo import.
            # In such cases, we want to return empty, since otherwise the behavior
            # above causes crashes.
            # TODO: Could we tailor to these cases as well somehow?
            return []

    def is_type_member(self, member):
        """
        Predicate function to filter a specified inspected member based
        on whether it is a type or not. If the member is not a type, False
        is returned, and otherwise True is returned.

        Can be used in conjunction with inspect.getmembers to retrieve types in a module.

        The types detected include classes, NewType definitions (typing) and Type Aliases (typing)

        :param: member  Member to check
        :return: True if member is a type
        """
        # If the attribute has an attribute for subclass checking, that means either the attribute
        # is a class, or it is a typing type (or a derivation).
        # Alternatively, the attribute has the attribute supertype, it means that the attribute has
        # been created using the NewType annotation.
        # Moreover, we ignore data descriptors (e.g. properties) as they are not type definitions.
        # TODO: This could possibly use major improvements; One concern is that this might break in newer Python versions.
        try:
            return (hasattr(member, "__subclasscheck__") or hasattr(member, "__supertype__")) \
                    and not inspect.isdatadescriptor(member)
        except:
            # There are cases where hasattr results in passing the call around, which ultimately might
            # result in a crash. Thus, we must be mindful of catching an exception, and returning False
            # in such a case.
            # Example: Analyzing werkzeug/locals.py will pass it on to Flask, which will tell us that
            # we are working outside of application context, ultimately resulting in a crash.
            return False

        # A bit more of a naive solution; Seems to overlook some types.
        # We only care about classes and type alises.
        # For classes, we simply check if the object is a class.
        # For type aliases, we check whether the object is an instance of a generic alias.
        # if (inspect.isclass(member) or isinstance(member, _GenericAlias)):
        #     return True
        # elif (inspect.isfunction(member)):
        #     # Convert the function representation into string format
        #     func_string = str(member)

        #     # If the function name starts with function NewType.<locals>.,
        #     # that means we have a new type declared by NewType.
        #     # TODO: This can perhaps be a bit improved to be less error prone.
        #     # TODO: Although if the typing definitions remain the same,
        #     # TODO: this should not be breaking.
        #     return func_string.startswith("<function NewType.<locals>.")
        
        # return False
    
    def get_types_module(self, module):
        """
        Returns the defined types in the specified module as a List of strings

        :param: module  Module to retrieve types from
        :return: List of types
        """
        try:
            members = inspect.getmembers(module, predicate=self.is_type_member)
            return self.get_types_from_members(members)
        except:
            # In very rare cases, getmembers might crash due to using the hasattr/getattr method call.
            return []

    def get_types_from_members(self, members):
        """
        Returns types from the specified members in the form of a set of strings.
        The types returned are qualified with their origin module name.
        The members are expected to be a container of tuples in the form (name, member),
        where name is a string, and member is an inspect member.

        :param: members  Container of member tuples in form (name, member)
        :return: Set of (qualified) types as strings
        """
        # Ignore types for which we cannot resolve the name for.
        # TODO: Maybe we can also handle edge cases here, if there are any remaining?
        type_strings = set([m[1].__module__ + "." + m[0] for m in members])
        
        return type_strings

extractor = ModuleExtractor()
fname = "module_test.py"

#import_entries = extractor.get_imports(fname)
#modules = extractor.resolve_modules(fname)
types = extractor.get_types(fname)

print(types)
#print(modules)
#print(import_entries)