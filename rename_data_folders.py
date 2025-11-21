"""
Script to rename all DATA folders and files to simplified tool ID naming convention.
Removes suffixes like 'gain10paperBG', 'gain10', '_NN', etc. from folder and file names.
"""
import os
import re
from pathlib import Path

# Path to the DATA folder
DATA_ROOT = Path(r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\PhD\Dataset\CCD_DATA\DATA")

def extract_tool_id(name):
    """
    Extract the base tool ID (e.g., 'tool070') from a full name like 'tool070gain10paperBG'.
    """
    match = re.match(r'(tool\d{3})', name)
    if match:
        return match.group(1)
    return None

def rename_folders(directory, suffix_to_remove):
    """
    Rename all folders in the specified directory by removing the suffix pattern.
    """
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    folders = [f for f in directory.iterdir() if f.is_dir()]
    renamed_count = 0
    
    for folder in sorted(folders):
        folder_name = folder.name
        tool_id = extract_tool_id(folder_name)
        
        if tool_id and folder_name != f"{tool_id}{suffix_to_remove}":
            new_name = folder_name.replace(folder_name, f"{tool_id}{suffix_to_remove}")
            # More robust: just use the tool_id + expected suffix
            new_name = f"{tool_id}{suffix_to_remove}"
            new_path = folder.parent / new_name
            
            if new_path.exists():
                print(f"⚠️  Skipped (target exists): {folder_name} -> {new_name}")
            else:
                try:
                    folder.rename(new_path)
                    print(f"✓ Renamed: {folder_name} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"✗ Failed to rename {folder_name}: {e}")
        else:
            if tool_id:
                print(f"  Already correct: {folder_name}")
    
    print(f"\nRenamed {renamed_count} folders in {directory.name}/\n")

def rename_files(directory, pattern, replacement_func):
    """
    Rename all files matching the pattern in the specified directory.
    """
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    files = [f for f in directory.iterdir() if f.is_file()]
    renamed_count = 0
    
    for file in sorted(files):
        file_name = file.name
        tool_id = extract_tool_id(file_name)
        
        if tool_id:
            new_name = replacement_func(file_name, tool_id)
            
            if new_name != file_name:
                new_path = file.parent / new_name
                
                if new_path.exists():
                    print(f"⚠️  Skipped (target exists): {file_name} -> {new_name}")
                else:
                    try:
                        file.rename(new_path)
                        print(f"✓ Renamed: {file_name} -> {new_name}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"✗ Failed to rename {file_name}: {e}")
            else:
                print(f"  Already correct: {file_name}")
    
    print(f"\nRenamed {renamed_count} files in {directory.name}/\n")

def simplify_profile_name(file_name, tool_id):
    """Simplify profile file names to tool###_area_vs_angle.csv/svg"""
    if '_area_vs_angle.csv' in file_name:
        return f"{tool_id}_area_vs_angle.csv"
    elif '_area_vs_angle_plot.svg' in file_name:
        return f"{tool_id}_area_vs_angle_plot.svg"
    return file_name

def main():
    print("=" * 70)
    print("DATA FOLDER RENAMING SCRIPT")
    print("=" * 70)
    print("\nThis script will rename folders and files to use simplified tool IDs.")
    print("Example: tool070gain10paperBG -> tool070\n")
    
    # Confirm before proceeding
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        return
    
    print("\n" + "=" * 70)
    print("STEP 1: Renaming folders in DATA/tools/")
    print("=" * 70)
    rename_folders(DATA_ROOT / "tools", "")
    
    print("=" * 70)
    print("STEP 2: Renaming folders in DATA/blurred/")
    print("=" * 70)
    rename_folders(DATA_ROOT / "blurred", "_blurred")
    
    print("=" * 70)
    print("STEP 3: Renaming folders in DATA/masks/")
    print("=" * 70)
    rename_folders(DATA_ROOT / "masks", "_final_masks")
    
    print("=" * 70)
    print("STEP 4: Renaming files in DATA/1d_profiles/")
    print("=" * 70)
    rename_files(DATA_ROOT / "1d_profiles", "_area_vs_angle", simplify_profile_name)
    
    print("\n" + "=" * 70)
    print("RENAMING COMPLETE!")
    print("=" * 70)
    print("\nYou can now use the simplified tool IDs in your configuration.")
    print("Example: TOOL_ID = 'tool070'")

if __name__ == "__main__":
    main()


#HAHA