def get_mean_score(output):
    pred_mean = 0
    for i, elem in enumerate(output, 1):
        pred_mean += i * elem
    return pred_mean

def get_std_score(output):
    pred_std = 0
    pred_mean = get_mean_score(output)
    for j, elem in enumerate(output, 1):
        pred_std += elem * (j - pred_mean) ** 2
    return pred_std