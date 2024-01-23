import subprocess

# Defining the allowed versions of packages that are used by Animal-AI PyPi package.
allowed_versions = {
    "mlagents_envs": "0.30.*",
    "numpy": "1.21.*",  # Version may need updates in the future based on dependencies.
    "protobuf": "3.20.*", # Need to update this accordingly if ml-agents version is updated.
}


def run_command(command):
    """Run a shell command and return its output, error and exit status."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, error = process.communicate()
    return output, error, process.returncode


def update_packages():
    """Update the specified packages to the allowed versions."""
    for package, version in allowed_versions.items():
        print(f"Updating {package} to version {version}...")
        output, error, status = run_command(f"pip install {package}=={version}")

        if status != 0:
            print(f"Failed to install {package}. Error: {error.decode().strip()}")
            continue

        print(f"{package} updated successfully.")


if __name__ == "__main__":
    print("Updating specified packages to allowed versions...")
    update_packages()
    print("Update complete.")
