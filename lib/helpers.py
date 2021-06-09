import os


def get_trial_dir(dir_format, i0=0):
    """Return directory for a particular trial

    Example: get_trial_dir('results/trial%d') will iterate through values of results/trial%d until it finds one that is
    not create. It then creates this directory and return the string. This is useful when you have parallelizing
    multiple trials so that each outputs files in a separate directory"""
    i = i0
    while True:
        results_dir_i = dir_format % i
        if os.path.isdir(results_dir_i):
            i += 1
        else:
            try:
                os.makedirs(results_dir_i)
                break
            except FileExistsError:
                pass
    return results_dir_i, i



