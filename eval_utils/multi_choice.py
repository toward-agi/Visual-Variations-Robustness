# 对于direction的任务，让模型从up,down,left,right,up-left,up-right,down-left,down-right中选择一个
# 对于object的任务，让模型直接输出，然后和我们给定的所有的这些candidate object进行比对，只要有匹配上的就算正确
# 之前shunian跟我说过这部分，就是，这种任务里面，用llm-as-a-judge的效果并不好。
def check_answer(gt, pred):
    gt = gt.split(',')
    gt = [x.lower() for x in gt]
    pred = pred.lower()
    return any(g in pred for g in gt)


def evaluate_direction(gt, pred):
    # List of valid directions
    directions = ["up", "down", "left", "right", "top-left", "bottom-left", "top-right", "bottom-right"]
    assert gt in directions, f"Invalid ground truth direction: {gt}"
    
    # Normalize pred and check if gt appears as a word in pred
    normalized_pred = pred.lower()  # Convert to lowercase for comparison
    if gt in normalized_pred:  # Check if gt is a substring of pred
        return True
    else:
        return False
    
def rotate_direction(initial_direction, angle):
    directions = ["up", "top-right", "right", "bottom-right", "down", "bottom-left", "left", "top-left"]
    direction_to_index = {
        "up": 0,
        "top-right": 1,
        "right": 2,
        "bottom-right": 3,
        "down": 4,
        "bottom-left": 5,
        "left": 6,
        "top-left": 7
    }
    
    # Ensure the initial direction is valid
    if initial_direction not in direction_to_index:
        raise ValueError(f"Invalid initial direction: {initial_direction}")
    
    # Calculate the number of 45-degree steps to rotate (negative for counterclockwise)
    steps = -angle // 45
    
    # Find the new direction index
    initial_index = direction_to_index[initial_direction]
    new_index = (initial_index + steps) % len(directions)
    
    # Return the new direction
    return directions[new_index]
