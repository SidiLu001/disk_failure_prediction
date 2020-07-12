import numpy as np

def read_data(xpath, ypath, group):
    X = np.load(xpath)
    y = np.load(ypath)
    L = X.shape[0]
    # **********************
    # smart: 0 - 12
    # perf: 13 - 92
    # loc: 93
    Loc = [93]
    Sgroup = [i for i in range(13)]
    Pgroup = [i for i in range(13,93)]
    groups = {
        'S' : Sgroup,
        'P' : Pgroup,
        'SL' : Sgroup + Loc,
        'PL' : Pgroup + Loc,
        'SP' : Sgroup + Pgroup,
        'SPL' : Sgroup + Pgroup + Loc,
    }
    assert group in groups
    useGroup = groups[group]
    X = X[:, :, useGroup]
    m,n,p = X.shape
    X = X.reshape(m,n,p,1)
    y = y.reshape(L, -1)

    from sklearn.model_selection import KFold
    seed = 15
    np.random.seed(seed)
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print('KFold = %d :'%n_splits)
    for train_index, test_index in kfold.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        yield train_X, train_y, test_X, test_y

    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # return train_X, train_y, test_X, test_y
    
def read_data1(xpath, ypath, group):
    X = np.load(xpath)
    y = np.load(ypath)
    L = X.shape[0]
    # **********************
    # smart: 0 - 12
    # perf: 13 - 92
    # loc: 93
    Loc = [93]
    Sgroup = [i for i in range(13)]
    Pgroup = [i for i in range(13,93)]
    groups = {
        'S' : Sgroup,
        'P' : Pgroup,
        'SL' : Sgroup + Loc,
        'PL' : Pgroup + Loc,
        'SP' : Sgroup + Pgroup,
        'SPL' : Sgroup + Pgroup + Loc,
    }
    assert group in groups
    useGroup = groups[group]
    X = X[:,:,useGroup]
    # **********************
    X = X.reshape(L, -1)
    y = y.reshape(L,)
    from sklearn.model_selection import KFold
    seed = 15
    np.random.seed(seed)
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print('KFold = %d :'%n_splits)
    for train_index, test_index in kfold.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        yield train_X, train_y, test_X, test_y
