def smooth(Y,weight=0.6): #weight是平滑度，tensorboard 默认0.6
    scalar = Y
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed