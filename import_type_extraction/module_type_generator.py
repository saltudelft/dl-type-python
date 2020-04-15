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

    def process_repos_for_members(self, repos_list, jobs, batch_size, start=0):
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
        
        print(f'Extracting import types for {project_id}...')
        
        # Get files recursively from project directory
        file_list = list_files(filtered_project_directory)
        
        # Store extracted types as dictionary {'filename' -> [type_list]}
        extracted_types = {}
        
        for filename in file_list:
            # Get import members & add to dictionary
            members = self.type_extractor.get_members(filename)
            extracted_types[filename] = {}

            for m in members:
                extracted_types[filename][m] = list(members[m])

        # Add entry for 'files' in project to contain dicts of filename, types and functions
        project['files'] = [{'filename': filename, \
                             'types': extracted_types[filename]['types'], \
                             'functions': extracted_types[filename]['functions']}
                        for filename in file_list]
        
        # Write the project as a CSV file
        self.write_project(project)

    def get_project_filename(self, project) -> str:
        """
        Return the filename at which a project import type datafile should be stored.
        :param project: the project dict
        :return: return filename
        """
        return os.path.join(self.output_dir, f"{project['author']}{project['repo']}-import-members.csv")
        
    def write_project(self, project) -> None:
        """
        Writes the project to a CSV file.
        Assumes the project already has a 'files' field which contains
        nested dictionary entries of {'filename', 'types', 'functions'} for each file.
        If there is no 'files' field, nothing is written; the project is simply skipped.

        :param: project  Project to write
        """
        import_types = []
        columns = ['author', 'repo', 'file', 'types', 'functions']

        if 'files' in project:
            for file in project['files']:
                type_data = (
                    project['author'],
                    project['repo'],
                    file['filename'],
                    file['types'],
                    file['functions']
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


    def filter_member(self, member_string: str, prefixes: list) -> bool:
        """
        Filters the specified member string with the given list of prefixes.
        Returns false if the member string should be filtered out, and true
        otherwise (if it should be kept)

        :param: member_string  String representation of a member
        :param: prefixes     List of prefixes (as strings) to omit
        :return: True if member should be kept, False if it should be filtered out
        """

        # Check for empty string (can be obtained after trimming quotes)
        if (len(member_string) == 0):
            return False

        for prefix in prefixes:
            if (member_string.startswith(prefix)):
                return False
        
        return True

    def filter_members(self, members: list, prefixes: list) -> list:
        """
        Filters the specified list of members with the given list of prefixes.
        Omits all members from members list that start with any string in prefixes.

        :param: members   List of member strings (as list of strings)
        :param: prefixes List of prefixes to use for filtering members out
        """

        return [t.strip('\'"') for t in members if self.filter_member(t.strip('\'"'), prefixes)]

    def string_to_list(self, list_string: str) -> list:
        """
        Converts a Python string representation of a list into a Python list of strings.

        :param: list_string  String representation of a list
        :return: List of strings
        """

        return list_string.strip(']["\'').split(', ')

    def concatenate_dataframes(self, out_dir: str, filename: str, prefix_filters = []) -> None:
        """
        Concatenates the CSV files present in the 'output_dir' location,
        and saves them to a single CSV file.

        :param: out_dir  Location to output CSV to (including filename)
        :param: prefix_filters - List of prefixes to filter out for resulting members in final CSV.
        """
        # Create missing dirs if needed
        os.makedirs(out_dir, exist_ok=True)

        DATA_FILES = list_files(self.output_dir)
        df = parse_df(DATA_FILES, batch_size=128)

        # Filter dataframe
        df = self.filter_dataframe(df, prefix_filters)

        # Write dataframe to CSV
        df.to_csv(os.path.join(out_dir, filename), index=False)

    def filter_dataframe(self, df: pd.DataFrame, prefix_filters = []):
        """
        Filters a dataframe encoding visible members with the specified prefix filters,
        and with additional criteria (filters out files ending in the _test.py suffix)

        :param: df  Dataframe to filter
        :param: prefix_filters List of prefixes to filter ouit for resulting members
        :return: filtered dataframe
        """
        # Filter members with specified prefix filters
        df['types'] = df['types'].apply(lambda t : self.filter_members(self.string_to_list(t), prefix_filters))
        df['functions'] = df['functions'].apply(lambda t : self.filter_members(self.string_to_list(t), prefix_filters))

        # Filter dataframe from files that have a *_test.py extension
        df = df[~df['file'].str.endswith('_test.py')]

        return df