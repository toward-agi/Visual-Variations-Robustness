# path这里，应该得有两种指标，一种是完全匹配，一种是部分匹配，就是将其拆成多个部分，然后看有多少部分是匹配的
# example: gt: [(1,1),(1,2),(3,-1)], pred: [(1,1),(1,3),(3,-1)]，按照位置逐个匹配，那就得返回[1,0,1]
import re

#总共三种指标
# 1. 完全匹配，就是模型可以识别，且可以follow顺序；
# 2. 只要匹配的上就行，模型可以识别，允许顺序不一样；
# 3. 按照顺序逐个匹配，和1类似，但是更加犀利度。 视觉能力以及follow能力。

def path_match_all(gt, pred):
    gt = gt.replace(' ', '')
    pred = pred.replace('（', '(').replace('）', ')').replace('，', ',')
    pred = pred.replace(' ', '')
    pred = pred.replace('[[', '[').replace(']]', ']')
    pred = pred.replace('【', '[').replace('】', ']')
    return gt in pred

def parse_coordinates(coord_str):
    pattern = r"\((-?\d+),\s*(-?\d+)\)"
    matches = re.findall(pattern, coord_str)
    return [(int(x), int(y)) for x, y in matches]


def path_match_partial_order(gt, pred):
    gt = gt.replace(' ', '')
    pred = pred.replace('（', '(').replace('）', ')').replace('，', ',')
    pred = pred.replace(' ', '')
    pred = pred.replace('[[', '[').replace(']]', ']')
    pred = pred.replace('【', '[').replace('】', ']')

    gt = parse_coordinates(gt)
    try:
        pred = parse_coordinates(pred)
    except:
        return [0] * len(gt)
    
    match = [0] * len(gt)

    if len(pred) < len(gt):
        pred = pred + [(999, 999)] * (len(gt) - len(pred))

    for i, (g, p) in enumerate(zip(gt, pred)):
        if g == p:
            match[i] = 1
    return match
        
def path_match_partial_non_order(gt, pred):
    gt = gt.replace(' ', '')
    pred = pred.replace('（', '(').replace('）', ')').replace('，', ',')
    pred = pred.replace(' ', '')
    pred = pred.replace('[[', '[').replace(']]', ']')
    pred = pred.replace('【', '[').replace('】', ']')

    gt = parse_coordinates(gt)
    try:
        pred = parse_coordinates(pred)
    except:
        return [0] * len(gt)

    match = [0] * len(gt)
    for i, g in enumerate(gt):
        for p in pred:
            if g == p:
                match[i] = 1
                break
    return match

def test():
    gt = '[(1,1),(1,2),(3,-1)]'
    pred = '[(1,1),(1,3),(1,2)]'
    print(path_match_all(gt, pred))
    print(path_match_partial_order(gt, pred))
    print(path_match_partial_non_order(gt, pred))

    gt = '[(1,1),(1,2),(3,-1)]'
    pred = '[(1,2)]'
    print(path_match_all(gt, pred))
    print(path_match_partial_order(gt, pred))
    print(path_match_partial_non_order(gt, pred))

    gt = '[(1,1),(1,2),(3,-1)]'
    pred = 'the answer is [(1,2),(1,1),(3,-1)]'
    print(path_match_all(gt, pred))
    print(path_match_partial_order(gt, pred))
    print(path_match_partial_non_order(gt, pred))


if __name__ == '__main__':
    test()