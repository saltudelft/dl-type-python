# Here the code checks which repositories can be accessed.


#from githubcloner import getReposURLs
from os.path import join, isdir, exists
from os import listdir
from distutils.dir_util import copy_tree
from shutil import copytree
import sys
import json
import collections

api_prefix = "https://api.github.com"


def find_author_repos(repos_path):
    """
    Creates a dictionary for all the repos of the authors
    :param repos_path:
    :return:
    """

    author_repos = {}

    for author in listdir(repos_path):
        author_repos[author] = []
        for repo in listdir(join(repos_path, author)):
            author_repos[author].append(repo)

    return author_repos


def load_json(f_path):
    """
    Loads a JSON file.
    """

    with open(f_path, 'r') as json_file:
        return json.load(json_file)


def list_file(l, f_name):
    """
    Writes the elements of a list to a file.
    """

    with open(f_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)


def find_duplicate_repos(repos_list):
    return [(repo, count) for repo, count in collections.Counter(repos_list).items() if count > 1]


def listdir_nohidden(path):

    for f in listdir(path):
        if not f.startswith("."):
            yield f


def find_current_repos(path_current_repos, author_repo=False):

    if author_repo:
        return ["%s/%s" % (user, p) for user in listdir_nohidden(path_current_repos) for p in listdir_nohidden(join(path_current_repos,
                                                                                user)) if isdir(join(path_current_repos,
                                                                                                     user, p))]
    else:
        return [p for user in listdir_nohidden(path_current_repos) for p in listdir_nohidden(join(path_current_repos,
                                                                                     user)) if isdir(join(path_current_repos,
                                                                                                          user, p))]

def remained_gh_repos(repos_list, path_current_repos):
    """
    Calculates the GitHub repos that failed to be fetched.
    """

    current_repos = find_current_repos(path_current_repos)
    repos_list = [p["repo"] for p in repos_list]
    return list(set(repos_list) - set(current_repos))

    # print("Intersection:", list(set(repos_list) & set(current_repos)))
    # print("All repos:", len(set(repos_list)))
    #
    # print("Current repos:", len(current_repos))
    # print("Remained repos:", len(remained_repos), remained_repos)


def select_repos(top_n, json_file_repos, current_repos_path):
    """
    It selects top n starred repos to another directory
    """

    all_repos = load_json(json_file_repos)
    current_repos = find_author_repos(current_repos_path)
    selected_repos = []

    for i, repo in enumerate(all_repos):

        if i <= top_n:
            if repo['author'] in current_repos and repo['repo'] in current_repos[repo['author']]:
                selected_repos.append("%s/%s" % (repo['author'], repo['repo']))
        else:
            break

    return selected_repos


def gen_json_file(f_name, all_repos, curr_repos):
    """
    Generates a JSON file from current projects and all the projects
    """

    found_repos = []
    for r in all_repos:
        if "%s/%s" % (r["author"], r["repo"]) in curr_repos:
            found_repos.append(r)

    with open(f_name, 'w') as f:
        json.dump(found_repos, f)


def cp_selected_repos(selected_repos, repos_path, dest_path):
    """
    It copies the selected repos into the destination directory.
    """

    for repo in selected_repos:

        author_name, repo_name = tuple(repo.split("/"))

        if not exists(join(dest_path, author_name, repo_name)):


#            copy_tree(join(repos_path, repo), join(dest_path, author_name, repo_name), symlinks=False,
#                     igonre_dangling_symlinks=True)
            copytree(join(repos_path, repo), join(dest_path, author_name, repo_name), symlinks=True,
                     ignore_dangling_symlinks=True)

        print("copied %s" % repo) 



if __name__ == '__main__':

    # gh_token = sys.argv[1]
    #
    # g = getReposURLs(api_prefix)
    # urls = g.fromAuthenticatedUser("mir-am", gh_token)
    #
    # print(urls)

    # repos = load_json('./data/mypy-dependents-by-stars.json')
    # remain_repos = remained_gh_repos(repos, './data/py_gh_repos/')
    # list_file(remain_repos, "./data/remian_repos.txt")

    #sel_repos = select_repos(500, './data/mypy-dependents-by-stars.json', './data/py_gh_repos/')
    #cp_selected_repos(sel_repos, './data/py_gh_repos/', './data/training_repos/')

    gen_json_file("./data/py_projects_all.json", load_json('./data/mypy-dependents-by-stars.json'),
                  find_current_repos('./data/py_gh_repos/', True))

    #print(find_author_repos('./data/py_gh_repos/'))

