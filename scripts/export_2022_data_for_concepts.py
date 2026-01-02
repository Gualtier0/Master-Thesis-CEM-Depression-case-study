#!/usr/bin/env python3
"""
Export 2022 Dataset for Concept Generation

This script processes the 2022 dataset and exports it in the format requested
: a list of lists where each inner list contains all posts
for one user.

Usage:
    python scripts/export_2022_data_for_concepts.py

Output:
    - data/raw/2022/2022_posts_for_concepts.pkl (recommended format)
    - data/raw/2022/2022_posts_for_concepts.npy (alternative format)
    - data/raw/2022/2022_subject_ids.json (subject ID mapping)
    - data/raw/2022/2022_data_stats.txt (statistics summary)
"""

import os
import pickle
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
import re


def find_all_xml_files(datos_dir: Path) -> List[Path]:
    """
    Find all subject XML files in the datos directory.

    Args:
        datos_dir: Path to the datos directory

    Returns:
        Sorted list of XML file paths
    """
    xml_files = list(datos_dir.glob("subject*.xml"))

    # Sort by numeric subject ID for consistent ordering
    def extract_subject_number(path):
        match = re.search(r'subject(\d+)', path.name)
        return int(match.group(1)) if match else 0

    return sorted(xml_files, key=extract_subject_number)


def parse_subject_xml(xml_file: Path) -> Tuple[str, List[str]]:
    """
    Extract subject ID and all posts from an XML file.

    Args:
        xml_file: Path to the subject's XML file

    Returns:
        Tuple of (subject_id, list_of_posts)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get subject ID
    subject_id_elem = root.find('ID')
    subject_id = subject_id_elem.text if subject_id_elem is not None else f"unknown_{xml_file.stem}"

    # Extract all posts
    posts = []
    for writing in root.findall('WRITING'):
        title_elem = writing.find('TITLE')
        text_elem = writing.find('TEXT')

        title = title_elem.text if title_elem is not None and title_elem.text else ""
        text = text_elem.text if text_elem is not None and text_elem.text else ""

        # Combine title and text, normalize whitespace
        post = f"{title} {text}".strip()
        post = re.sub(r'\s+', ' ', post)  # Normalize multiple spaces/newlines

        if post:  # Only add non-empty posts
            posts.append(post)

    return subject_id, posts


def load_labels(labels_file: Path) -> dict:
    """
    Load labels from risk_golden_truth.txt.

    Args:
        labels_file: Path to the labels file

    Returns:
        Dictionary mapping subject_id to label (0=control, 1=patient)
    """
    labels = {}
    if not labels_file.exists():
        print(f"  ⚠ Warning: Labels file not found: {labels_file}")
        return labels

    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subject_id, label = parts
                labels[subject_id] = int(label)

    return labels


def build_posts_list(xml_files: List[Path], labels: dict = None) -> Tuple[List[List[str]], List[str], List[int]]:
    """
    Build list of lists structure from XML files.

    Args:
        xml_files: List of XML file paths
        labels: Optional dictionary of subject_id -> label

    Returns:
        Tuple of (all_posts, subject_ids, label_list)
        - all_posts: List of lists, each inner list contains posts for one subject
        - subject_ids: List of subject IDs corresponding to each inner list
        - label_list: List of labels (0 or 1) corresponding to each subject
    """
    all_posts = []
    subject_ids = []
    label_list = []

    print(f"Parsing {len(xml_files)} XML files...")

    for i, xml_file in enumerate(xml_files):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(xml_files)} files processed")

        subject_id, posts = parse_subject_xml(xml_file)
        all_posts.append(posts)
        subject_ids.append(subject_id)

        # Get label if available
        if labels:
            label = labels.get(subject_id, -1)  # -1 if not found
            label_list.append(label)
        else:
            label_list.append(-1)

    return all_posts, subject_ids, label_list


def sort_by_labels(all_posts: List[List[str]], subject_ids: List[str], labels: List[int]) -> Tuple[List[List[str]], List[str], List[int]]:
    """
    Sort subjects by label (controls first, then patients).

    Args:
        all_posts: List of lists of posts
        subject_ids: List of subject IDs
        labels: List of labels

    Returns:
        Sorted (all_posts, subject_ids, labels) with controls (0) first, then patients (1)
    """
    # Create list of tuples (posts, subject_id, label)
    combined = list(zip(all_posts, subject_ids, labels))

    # Sort by label (0 first, then 1)
    combined_sorted = sorted(combined, key=lambda x: x[2])

    # Unzip back into separate lists
    all_posts_sorted = [item[0] for item in combined_sorted]
    subject_ids_sorted = [item[1] for item in combined_sorted]
    labels_sorted = [item[2] for item in combined_sorted]

    return all_posts_sorted, subject_ids_sorted, labels_sorted


def generate_stats(all_posts: List[List[str]], subject_ids: List[str], labels: List[int] = None) -> str:
    """
    Generate summary statistics for the dataset.

    Args:
        all_posts: List of lists of posts
        subject_ids: List of subject IDs
        labels: Optional list of labels

    Returns:
        Formatted statistics string
    """
    total_subjects = len(all_posts)
    total_posts = sum(len(posts) for posts in all_posts)
    avg_posts = total_posts / total_subjects if total_subjects > 0 else 0
    min_posts = min(len(posts) for posts in all_posts) if all_posts else 0
    max_posts = max(len(posts) for posts in all_posts) if all_posts else 0

    # Count posts distribution
    post_counts = [len(posts) for posts in all_posts]
    median_posts = sorted(post_counts)[len(post_counts) // 2] if post_counts else 0

    # Count labels if available
    label_info = ""
    if labels:
        n_controls = labels.count(0)
        n_patients = labels.count(1)
        label_info = f"""
Label Distribution:
- Controls (label 0): {n_controls} subjects (indices 0-{n_controls-1})
- Patients (label 1): {n_patients} subjects (indices {n_controls}-{total_subjects-1})

IMPORTANT: Subjects are sorted by label!
- First {n_controls} subjects = controls
- Last {n_patients} subjects = patients
"""

    stats = f"""2022 Dataset Statistics
========================

Total Subjects: {total_subjects}
Total Posts: {total_posts}
Average Posts per Subject: {avg_posts:.2f}
Median Posts per Subject: {median_posts}
Min Posts per Subject: {min_posts}
Max Posts per Subject: {max_posts}
{label_info}
Data Structure:
- Format: List of lists
- Each list represents one subject
- Each inner list contains all posts for that subject

Files Generated:
1. 2022_posts_for_concepts.pkl - Pickle format (recommended)
2. 2022_posts_for_concepts.npy - Numpy format (alternative)
3. 2022_subject_ids.json - Subject ID mapping
4. 2022_data_stats.txt - This file

Usage:
------
# Load pickle format (recommended)
import pickle
with open('2022_posts_for_concepts.pkl', 'rb') as f:
    posts = pickle.load(f)

# Load numpy format
import numpy as np
posts = np.load('2022_posts_for_concepts.npy', allow_pickle=True)

Structure:
----------
posts[0] = [post1, post2, post3, ...]  # All posts for subject {subject_ids[0] if subject_ids else '0'}
posts[1] = [post1, post2, ...]         # All posts for subject {subject_ids[1] if len(subject_ids) > 1 else '1'}
...
posts[{total_subjects-1}] = [...]      # All posts for subject {subject_ids[-1] if subject_ids else 'N'}

Total: {total_subjects} subjects, {total_posts} posts
    
"""
    return stats


def save_data(all_posts: List[List[str]], subject_ids: List[str], labels: List[int], output_dir: Path):
    """
    Save data in multiple formats.

    Args:
        all_posts: List of lists of posts
        subject_ids: List of subject IDs
        labels: List of labels
        output_dir: Directory to save output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pickle format (recommended)
    print("\nSaving outputs...")
    pkl_path = output_dir / "2022_posts_for_concepts.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_posts, f)
    print(f"  ✓ Saved: {pkl_path.name}")

    # 2. Numpy format (alternative)
    npy_path = output_dir / "2022_posts_for_concepts.npy"
    np.save(npy_path, np.array(all_posts, dtype=object))
    print(f"  ✓ Saved: {npy_path.name}")

    # 3. Subject IDs mapping (with labels)
    json_path = output_dir / "2022_subject_ids.json"
    id_mapping = {
        i: {
            "subject_id": subject_id,
            "label": label,
            "label_name": "control" if label == 0 else "patient"
        }
        for i, (subject_id, label) in enumerate(zip(subject_ids, labels))
    }
    with open(json_path, 'w') as f:
        json.dump(id_mapping, f, indent=2)
    print(f"  ✓ Saved: {json_path.name}")

    # 4. Statistics
    stats = generate_stats(all_posts, subject_ids, labels)
    stats_path = output_dir / "2022_data_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(stats)
    print(f"  ✓ Saved: {stats_path.name}")

    return pkl_path, npy_path, json_path, stats_path


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    datos_dir = project_root / "data" / "raw" / "2022" / "datos"
    output_dir = project_root / "data" / "raw" / "2022"

    print("="*70)
    print("Export 2022 Dataset for Concept Generation")
    print("="*70)
    print(f"\nData source: {datos_dir}")
    print(f"Output directory: {output_dir}")

    # Check if datos directory exists
    if not datos_dir.exists():
        print(f"\n❌ ERROR: Directory not found: {datos_dir}")
        print("Please ensure the 2022 dataset is in data/raw/2022/datos/")
        return 1

    # Load labels
    print("\nStep 1: Loading labels...")
    labels_file = output_dir / "risk_golden_truth.txt"
    labels_dict = load_labels(labels_file)
    print(f"  ✓ Loaded {len(labels_dict)} labels")

    # Find XML files
    print("\nStep 2: Finding XML files...")
    xml_files = find_all_xml_files(datos_dir)
    print(f"  ✓ Found {len(xml_files)} XML files")

    if len(xml_files) == 0:
        print("\n❌ ERROR: No XML files found!")
        return 1

    # Parse XML files and build data structure
    print("\nStep 3: Parsing XML files and extracting posts...")
    all_posts, subject_ids, labels = build_posts_list(xml_files, labels_dict)
    print(f"  ✓ Parsed {len(subject_ids)} subjects")

    total_posts = sum(len(posts) for posts in all_posts)
    print(f"  ✓ Extracted {total_posts} total posts")

    # Sort by labels (controls first, then patients)
    print("\nStep 4: Sorting by labels (controls first, then patients)...")
    all_posts, subject_ids, labels = sort_by_labels(all_posts, subject_ids, labels)
    n_controls = labels.count(0)
    n_patients = labels.count(1)
    print(f"  ✓ Sorted: {n_controls} controls (indices 0-{n_controls-1}), {n_patients} patients (indices {n_controls}-{len(labels)-1})")

    # Save data
    print("\nStep 5: Saving data...")
    pkl_path, npy_path, json_path, stats_path = save_data(all_posts, subject_ids, labels, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  1. {pkl_path.name} - Main output (send to supervisor)")
    print(f"  2. {npy_path.name} - Alternative format")
    print(f"  3. {json_path.name} - Subject ID and label mapping")
    print(f"  4. {stats_path.name} - Statistics summary")

    print(f"\nData structure:")
    print(f"  - {len(all_posts)} subjects total")
    print(f"  - First {n_controls} = controls (label 0)")
    print(f"  - Last {n_patients} = patients (label 1)")
    print(f"  - {total_posts} total posts")
    print(f"  - Format: List of lists [[user1_posts], [user2_posts], ...]")

    print(f"\nNext step:")
    print(f"  Send '{pkl_path.name}' to your supervisor for concept generation.")
    print(f"  Tell them: First {n_controls} are controls, last {n_patients} are patients.")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
