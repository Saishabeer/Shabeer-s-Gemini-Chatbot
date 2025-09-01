def find_duplicates(input_list):
    """
    Finds and returns a list of duplicate elements from the input list.

    Args:
        input_list: A list of elements (can be numbers, strings, or any hashable type).

    Returns:
        A list containing only the duplicate elements.  Returns an empty list if no duplicates are found.
    """
    seen = set()
    duplicates = []
    for item in input_list:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates

# Example usage:
my_list = [1, 2, 3, 2, 4, 1, 5, 6, 3, "apple", "banana", "apple"]
duplicate_elements = find_duplicates(my_list)
print(f"The duplicate elements are: {duplicate_elements}")
