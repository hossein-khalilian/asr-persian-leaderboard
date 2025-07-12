import re

import pkg_resources

# Input and output file (same in this case)
filename = "../requirements.txt"


def get_installed_version(pkg_name):
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistributionNotFound:
        return None


def extract_base_package(package):
    # Handles extras like `datasets[audio]` and avoids version pinning issues
    return re.split(r"[<=>\[ ]", package)[0]


updated_lines = []
with open(filename, "r") as file:
    for line in file:
        stripped = line.strip()

        # Skip empty lines or comments
        if not stripped or stripped.startswith("#"):
            updated_lines.append(line)
            continue

        # Already versioned
        if any(op in stripped for op in ["==", ">=", "<="]):
            updated_lines.append(line)
            continue

        # Extract package name (handle extras like datasets[audio])
        base_pkg = extract_base_package(stripped)
        version = get_installed_version(base_pkg)

        if version:
            # Insert version with original formatting
            if "[" in stripped:
                extras = stripped[stripped.index("[") :]
                updated_line = f"{base_pkg}{extras}=={version}\n"
            else:
                updated_line = f"{base_pkg}=={version}\n"
            updated_lines.append(updated_line)
        else:
            print(f"⚠️ Package '{base_pkg}' not found in environment. Keeping as-is.")
            updated_lines.append(line)

# Save updated requirements
with open(filename, "w") as file:
    file.writelines(updated_lines)

print("✅ requirements.txt updated with installed versions.")
