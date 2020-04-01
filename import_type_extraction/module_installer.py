import subprocess
import sys
import urllib.request, json

def get_json_from_url(url):
    """
    Retrieves a JSON resource from the specified URL.

    :param: url  URL to load data from
    :return: Dictionary corresponding to the JSON data
    """

    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
        return data
    
    return None

def extract_packages_from_url(url):
    """
    Extracts a list of packages from the specified URL.

    :param: url  URL to laod packages from
    :return: list of package names extracted from the URL.
    """

    popular_packages = get_json_from_url(url)
    projects = [x['project'] for x in popular_packages['rows']]

    return projects

def install_packages(packages):
    """
    Installs the list of specified packages.

    :param: packages  List of packages (as list of package names)
    """

    # Install all specified modules
    for p in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])
        except:
            # Ignore package if we cannot install it
            continue

# Source: https://hugovk.github.io/top-pypi-packages/
url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-365-days.json"

packages = extract_packages_from_url(url)
print(packages)
install_packages(packages)
