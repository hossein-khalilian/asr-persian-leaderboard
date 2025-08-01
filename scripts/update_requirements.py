import re

import pkg_resources

# Input and output file (same in this case)
filename = "./requirements.txt"


def get_installed_version(pkg_name):
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistributionNotFound:
        return None
    except pkg_resources.RequirementParseError:
        return None


def extract_base_package(package):
    # Handles extras like `datasets[audio]` and avoids version pinning issues
    return re.split(r"[<=>\[ ]", package)[0]


def should_skip_line(line):
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("--")
        or stripped.startswith("git+")
        or "://" in stripped  # skip URLs
    )


updated_lines = []
with open(filename, "r") as file:
    for line in file:
        stripped = line.strip()

        if should_skip_line(stripped):
            updated_lines.append(line)
            continue

        # Already versioned
        if any(op in stripped for op in ["==", ">=", "<="]):
            updated_lines.append(line)
            continue

        base_pkg = extract_base_package(stripped)
        version = get_installed_version(base_pkg)

        if version:
            # Keep extras if present
            if "[" in stripped:
                extras = stripped[stripped.index("[") :]
                updated_line = f"{base_pkg}{extras}=={version}\n"
            else:
                updated_line = f"{base_pkg}=={version}\n"
            updated_lines.append(updated_line)
        else:
            print(f"⚠️ Package '{base_pkg}' not found in environment. Keeping as-is.")
            updated_lines.append(line)

with open(filename, "w") as file:
    file.writelines(updated_lines)

print("✅ requirements.txt updated with installed versions.")
