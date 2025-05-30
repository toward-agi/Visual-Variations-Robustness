import re
import math

def coordinate_match(gt, model_output):
    # Define a function to clean and convert string to float and handle the small decimal tolerance
    def parse_coordinate(coord_str):
        # Remove parentheses, spaces, and filter out any non-numeric values
        coord_str = coord_str.strip('()').strip()
        # Split by commas and only process numeric values
        values = coord_str.split(',')
        numeric_values = []
        
        for val in values:
            val = val.strip()
            if val:  # Ignore empty strings
                try:
                    # Try to convert each value to a float (will handle both integers and floats)
                    numeric_values.append(float(val))
                except ValueError:
                    # If the value cannot be converted, skip it
                    continue
        
        return numeric_values

    # Extract the coordinates from the model output using regular expressions
    coord_pattern = r"\(([^)]+)\)"  # Pattern to match the coordinates inside parentheses
    model_coords = re.findall(coord_pattern, model_output)
    
    if not model_coords:
        return False
    
    # Parse the model coordinates and ground truth into numerical lists
    model_coords = [parse_coordinate(coord) for coord in model_coords]
    gt_coords = parse_coordinate(gt)

    # Check dimensionality and allow for the tolerance in comparison
    for model_coord in model_coords:
        # Ensure the number of dimensions match between model output and GT
        if len(model_coord) != len(gt_coords):
            continue
        
        # Check if the values are approximately equal (tolerance for small floating-point differences)
        if all(math.isclose(m, g, abs_tol=1e-6) for m, g in zip(model_coord, gt_coords)):
            return True
    
    return False
