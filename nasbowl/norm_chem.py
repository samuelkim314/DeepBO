def get_objective_y(y, objective='gap'):
    # Columns: A, B, C, mu, alpha, homo, lumo, gap, r2
    y1 = y.loc[:, 'A':'r2'].values
    y1[:, 0] = y1[:, 0] / 6.2e5
    y1[:, 1] = y1[:, 1] / 4.4e2
    y1[:, 2] = y1[:, 2] / 2.8e2
    y1[:, 3] = y1[:, 3] / 30
    y1[:, 4] = (y1[:, 4] - 6) / 191     # alpha
    y1[:, 5] = (y1[:, 5] + 0.4) / 0.5
    y1[:, 6] = (y1[:, 6] + 0.2) / 0.4
    y1[:, 7] = (y1[:, 7] - 0.02) / 0.6  # gap, eps_lumo - eps_homo
    y1[:, 8] = (y1[:, 8] - 19) / 3.4e3

    # Thermodynamic quantities, normalized. Columns: u298, h298, g298, cv
    y2 = y.loc[:, 'u298':'cv'].values
    y2[:, 0:3] = (y2[:, 0:3] + 700) / 750
    y2[:, 3] = (y2[:, 3] - 6) / 52

    # The objective function picks out the desired feature and un-normalizes it
    obj_fun = None
    if objective == 'gap':
        def obj_fun(z):
            return z[:, 7] * 0.6 + 0.02
    elif objective == 'mingap':
        def obj_fun(z):
            return -(z[:, 7] * 0.6 + 0.02)
    elif objective == 'alpha':
        def obj_fun(z):
            return z[:, 4] * 191 + 6
    elif objective == 'minalpha':
        def obj_fun(z):
            return -(z[:, 4] * 191 + 6)
    elif objective == 'cv':     # Heat capacity at 298.15K
        def obj_fun(z):
            return z[:, 3] * 52 + 6
    elif objective == 'mincv':
        def obj_fun(z):
            return - (z[:, 3] * 52 + 6)
    elif objective == 'gap0.1':
        def obj_fun(z):
            gap = z[:, 7] * 0.6 + 0.02
            return -(gap - 0.1)**2
    elif objective == 'gapalpha':
        def obj_fun(z):
            gap = z[:, 7]
            alpha = z[:, 4]
            return gap + alpha

    elif objective == 'mingapalpha':
        def obj_fun(z):
            gap = z[:, 7]
            alpha = z[:, 4]
            return -gap + alpha

    if objective == 'gap' or objective == 'mingap' or objective == 'alpha' or objective == 'minalpha' or \
            objective == 'gap0.1' or objective == 'gapalpha' or objective == 'mingapalpha':
        return y1, obj_fun
    else:
        return y2, obj_fun
