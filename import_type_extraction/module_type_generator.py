import sys
import os
import pandas as pd

# Append directory before current one to import DLTPy & gh_query dependencies
sys.path.append('../')
from dltpy import config
from dltpy.preprocessing.pipeline import project_filter, list_files
from dltpy.input_preparation.generate_df import parse_df
from dltpy.preprocessing.utils import ParallelExecutor

from joblib import delayed
from module_type_extractor import ModuleExtractor

class ModuleGenerator():
    def __init__(self, repos_dir, output_dir):
        self.repos_dir = repos_dir
        self.output_dir = output_dir
        self.type_extractor = ModuleExtractor()

    def process_repos_for_types(self, repos_list, jobs, batch_size, start=0):
        """
        Processes the specified list of repositories for import types, by extracting the import
        types for each project, and saving the corresponding CSV files.
        Runs the processing in parallel for each repo.

        :param: repos_list List of repositories
        :param: jobs       Number of jobs to use
        :param: start      Starting index for ID
        """
        # Create missing dirs if needed
        os.makedirs(self.output_dir, exist_ok=True)

        ParallelExecutor(n_jobs=jobs, batch_size=batch_size)(total=len(repos_list))(
            delayed(self.process_project_for_import)(i, project) for i, project in enumerate(repos_list, start=start))

    def process_project_for_import(self, i, project):
        """
        Processes a single project for import type analysis.
        The function extracts the visible import types for a project,
        creates a dataframe for the project with columns: [author, repo, file, types],
        and saves it in the output directory.
        
        TODO: Code adapted from pipeline.py. In the future, it would
        probably be better to extract the common functionality to reduce
        code redundancy in terms of changes.

        :param: i        ID of the project
        :param: project  Project dictionary (expected to contain at least the author, repo fields)
        """
        project_id = f'{project["author"]}/{project["repo"]}'
        print(f'Running pipeline for project {i} {project_id}')

        # TODO: Caching check could be added here
        
        # Get directory
        print(f'Filtering for {project_id}...')
        filtered_project_directory = project_filter.filter_directory(os.path.join(self.repos_dir, project["author"],
                                                                                project["repo"]))
        
        print(f'Extracting import typesfor {project_id}...')
        
        # Get files recursively from project directory
        file_list = list_files(filtered_project_directory)
        
        # Store extracted types as dictionary {'filename' -> [type_list]}
        extracted_types = {}
        
        for filename in file_list:
            # Get import types & add to dictionary
            extracted_types[filename] = list(self.type_extractor.get_types(filename))

        # Add entry for 'files' in project to contain dicts of filename and types
        project['files'] = [{'filename': filename, 'types': extracted_types[filename] }
                        for filename in file_list]
        
        # Write the project as a CSV file
        self.write_project(project)

    def get_project_filename(self, project) -> str:
        """
        Return the filename at which a project import type datafile should be stored.
        :param project: the project dict
        :return: return filename
        """
        return os.path.join(self.output_dir, f"{project['author']}{project['repo']}-import-types.csv")
        
    def write_project(self, project) -> None:
        """
        Writes the project to a CSV file.
        Assumes the project already has a 'files' field which contains
        nested dictionary entries of {'filename', 'types'} for each file.
        If there is no 'files' field, nothing is written; the project is simply skipped.

        :param: project  Project to write
        """
        import_types = []
        columns = ['author', 'repo', 'file', 'types']

        if 'files' in project:
            for file in project['files']:
                type_data = (
                    project['author'],
                    project['repo'],
                    file['filename'],
                    file['types']
                )

                import_types.append(type_data)

        if len(import_types) == 0:
            print("Skipped...")
            return
        
        type_df = pd.DataFrame(import_types, columns=columns)
        type_df.to_csv(self.get_project_filename(project), index=False)

        # Delete dataframe after writing to CSV to eventually let the garbage collector
        # free memory.
        del type_df


    def filter_type(self, type_string: str, prefixes: list) -> bool:
        """
        Filters the specified type string with the given list of prefixes.
        Returns false if the type string should be filtered out, and true
        otherwise (if it should be kept)

        :param: type_string  String representation of a type
        :param: prefixes     List of prefixes (as strings) to omit
        :return: True if type should be kept, False if it should be filtered out
        """

        for prefix in prefixes:
            if (type_string.startswith(prefix)):
                return False
        
        return True

    def filter_types(self, types: list, prefixes: list) -> list:
        """
        Filters the specified list of types with the given list of prefixes.
        Omits all types from types list that start with any string in prefixes.

        :param: types   List of type strings (as list of strings)
        :param: prefixes List of prefixes to use for filtering types out
        """

        return [t.strip('\'"') for t in types if self.filter_type(t.strip('\'"'), prefixes)]

    def string_to_list(self, list_string: str) -> list:
        """
        Converts a Python string representation of a list into a Python list of strings.

        :param: list_string  String representation of a list
        :return: List of strings
        """

        return list_string.strip(']["\'').split(', ')

    def concatenate_dataframes(self, out_dir: str, prefix_filters = []) -> None:
        """
        Concatenates the CSV files present in the 'output_dir' location,
        and saves them to a single CSV file.

        :param: out_dir  Location to output CSV to (including filename)
        :param: prefix_filters - List of prefixes to filter out for resulting types in final CSV.
        """
        # Create missing dirs if needed
        os.makedirs(out_dir, exist_ok=True)

        DATA_FILES = list_files(self.output_dir)
        df = parse_df(DATA_FILES, batch_size=128)

        # Filter types with specified prefix filters
        df['types'] = df['types'].apply(lambda t : self.filter_types(self.string_to_list(t), prefix_filters))

        df.to_csv(out_dir, index=False)