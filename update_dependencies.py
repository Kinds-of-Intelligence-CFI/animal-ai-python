import subprocess

# Defining the allowed versions of packages that are used by Animal-AI PyPi package.
# This script is called by the build process to ensure that the correct versions of packages are installed.
allowed_versions = {
    "mlagents_envs": "0.30.*",
    "numpy": "1.21.*",  # For now, we're using this version of numpy but it may need to be updated in the future if transitive dependencies require it (e.g. ml-agents 1.0.0).
    "protobuf": "3.20.*",
}


def update_packages():
    for package, version in allowed_versions.items():
        subprocess.run(["pip", "install", f"{package}=={version}"])


if __name__ == "__main__":
    print("Updating specified packages to allowed versions...")
    update_packages()
    print("Udate complete.")
