# README:
# This script modifies the priority of Scoop shims on Windows.
#
# By default, in scoop, later installed software will overwrite the shims of previously installed software.
# There is no way to change the priority except `scoop shim alter` or `scoop reset`.
# This is troublesome when you have many packages. So I wrote this script to make it more user friendly.
# Just copy and run it in Python. Enjoy!
#
# Usage:
# 1. Interactive mode (default): `python modify_scoop_shims_priority.py`
# 2. Auto mode: `python modify_scoop_shims_priority.py --auto`
#
# Known issue:
# - If there are spaces in the 'Name', may causing unexpected error


import argparse
import os
import subprocess

shims = {}


def get_shim_list() -> str:
    result = subprocess.run(
        ["pwsh", "-Command", "scoop shim list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def parse_shim_list(output):
    lines = output.strip().split("\n")[3:]  # Skip header lines

    index = 1
    print("Index  Name             Source                              Alternatives")
    for line in lines:
        parts = line.split()[:-2]
        if len(parts) < 3:
            continue
        if parts[0] == "office" and parts[1] == "tool" and parts[2] == "plus.console":
            continue  # FIXME: office-tool-plus shim named 'office tool plus.console' breaks the parsing
        name: str = parts[0]
        source: str = parts[1]
        alternatives: list[str] = parts[2:]
        print(
            f"{index:>3}. {name:<12} from {source:<16} have alternatives: {", ".join(alternatives)}"
        )
        shims[name] = {"source": source, "alternatives": alternatives}
        index += 1


def change_shim_priority(name, source, alter_name):
    cur_shim_path = os.path.join(os.environ["SCOOP"], "shims", f"{name}.shim")
    bak_shim_path = os.path.join(
        os.environ["SCOOP"], "shims", f"{name}.shim.{shims[name]['source']}"
    )
    new_shim_path = os.path.join(
        os.environ["SCOOP"], "shims", f"{name}.shim.{alter_name}"
    )

    os.rename(cur_shim_path, bak_shim_path)
    os.rename(new_shim_path, cur_shim_path)

    print(f"Modified shim priority for '{name}' to use '{alter_name}'.")


def auto_subpriority(subalter_name):
    for name, details in shims.items():
        source = details["source"]
        alternatives = details["alternatives"]
        if source == subalter_name and len(alternatives) == 2:
            print(
                f"Auto-modifying '{name}' priority '{alternatives[0]}' to '{alternatives[1]}'"
            )
            change_shim_priority(name, source, alternatives[1])


def interactive_subpriority():
    while True:
        index = input("Enter index to modify (or 'q' to quit): ").strip()
        if index.isdigit() and 1 <= int(index) <= len(shims):
            name = list(shims.keys())[int(index) - 1]
            details = shims[name]
            source = details["source"]
            alternatives = details["alternatives"]
            print(f"Modifying shim priority for '{name}' (current source: '{source}')")
            choice = input(
                "Select the alternative to use (0 or 'q' cancel): "
                + ", ".join(f"{i}. {a}" for i, a in enumerate(alternatives))
                + " "
            ).strip()
            if choice.isdigit() and 0 < int(choice) < len(alternatives):
                alter_name = alternatives[int(choice)]
                change_shim_priority(name, source, alter_name)
            elif choice.lower() == "0" or choice.lower() == "q":
                print("No changes made.")
            else:
                print("Invalid selection. Ignoring...")
        elif index.lower() == "q":
            break
        else:
            print("Invalid selection. Ignoring...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto", action="store_true", help="Auto modify sub-priority shims"
    )
    print("Scanning for shims, it may take a while...")
    print("-" * 80)
    parse_shim_list(get_shim_list())
    for name, details in shims.items():
        assert (
            details["source"] == details["alternatives"][0]
        ), "Scoop may change 'shim list' output format, please check."
    print("-" * 80)
    args = parser.parse_args()
    if args.auto:
        auto_subpriority("busybox")
    else:
        interactive_subpriority()


if __name__ == "__main__":
    main()
