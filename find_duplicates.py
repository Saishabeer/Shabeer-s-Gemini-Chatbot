import sys
from collections import Counter
from typing import List, Any


def find_all_duplicates(data: List[Any]) -> List[Any]:
    """
    Finds all items that appear more than once in a list.

    This function uses collections.Counter for efficiency, making it suitable
    for lists with various data types (numbers, strings, etc.).

    Args:
        data: A list of items to check for duplicates.

    Returns:
        A list containing only the items that were duplicated.
        Returns an empty list if no duplicates are found.
    """
    if not isinstance(data, list) or not data:
        return []

    # Count occurrences of each item in the list
    counts = Counter(data)

    # Use a list comprehension to filter for items that appear more than once
    duplicates = [item for item, count in counts.items() if count > 1]
    return duplicates


if __name__ == "__main__":
    # Example list with mixed data types
    sample_list = [1, 2, 3, 2, 4, 1, 5, 6, 3, "a", "a", "hello", "world", "hello"]

    print(f"Original list: {sample_list}")

    duplicate_items = find_all_duplicates(sample_list)

    if duplicate_items:
        print(f"\n✅ Found duplicates: {duplicate_items}")
    else:
        print("\nℹ️ No duplicates were found in the list.")