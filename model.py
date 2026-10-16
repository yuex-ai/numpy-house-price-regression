"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).
    Args:
        X: (N, F) array-like of floats, may contain NaN.
    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    N,F=X.shape
    X=np.asarray(X,dtype=float)
    mask=np.isnan(X)
    col_mean=np.nanmean(X,axis=0)
    global_mean=np.nanmean(X)
    if np.isnan(global_mean):
        return 0.0
    col_mean=np.where(np.isnan(col_mean),0.0,col_mean)
    row,col=np.where(mask)
    X[row,col]=col_mean[col]
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.asarray(X, dtype=float)
    q1=np.percentile(X,25,axis=0)
    q3=np.percentile(X,75,axis=0)
    iqr=q3-q1
    lower=q1-k*iqr
    upper=q3+k*iqr
    return lower,upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X=np.asarray(X,dtype=float)
    X=np.clip(X,lower,upper)
    return X

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    numerator=np.asarray(numerator,dtype=float)
    denominator=np.asarray(denominator,dtype=float)
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    N,F=X.shape
    col=col.reshape(N,1)
    #output=np.concatenate([X,col],axis=1)
    output=np.column_stack([X,col])
    return output

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    N = labels.shape[0]

    uniques, inverse = np.unique(labels, return_inverse=True)
    C = uniques.shape[0]

    output = np.zeros((N, C), dtype=float)
    output[np.arange(N), inverse] = 1.0
    return output

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X=np.asarray(X,dtype=float)
    x_mean=np.mean(X,axis=0)
    x_std=np.std(X,axis=0)
    x_std=np.where(x_std==0,1.0,x_std)
    return (x_mean,x_std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X=np.asarray(X,dtype=float)
    output=(X-mean)/std
    return output

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N,F=X.shape
    bias=np.ones((N,1))
    output=np.hstack([bias,X])
    return output

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng=np.random.RandomState(seed)
    idx=rng.permutation(n_samples)
    return idx

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    indices=np.asarray(indices)
    n=len(indices)
    test_ratio=1-train_ratio-val_ratio
    train_n=int(n*train_ratio)
    val_n=int(n*val_ratio)
    test_n=int(n*test_ratio)
    train_idx=indices[:train_n]
    val_idx=indices[train_n:train_n+val_n]
    test_idx=indices[train_n+val_n:]
    return (train_idx,val_idx,test_idx)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.asarray(indices)

    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A=X.T@X
    b=X.T@y
    theta=np.linalg.inv(A)@b
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    n=len(y_true)
    output=(1/n)*sum(np.abs(y_true-y_pred))
    return float(output)

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    n=len(y_true)
    output=np.sqrt((1/n)*sum((y_true-y_pred)**2))
    return float(output)

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    avg=np.mean(y_true)
    res=sum((y_true-y_pred)**2)
    tol=sum((y_true-avg)**2)
    if tol==0:
        return 0.0
    else:
        return float(1-(res/tol))

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r=y_true-y_pred
    mean=np.mean(r)
    std=np.std(r)
    median_abs=np.median(np.abs(r))
    dict={
        'mean':mean,
        'std':std,
        'median_abs':median_abs
    }
    return dict

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X=impute_nan_with_mean(X)
    lower,upper=compute_iqr_bounds(X,iqr_k)
    X=clip_columns(X,lower,upper)
    return X

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    X_num = np.asarray(X_num, dtype=float)

    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio_col = make_ratio_feature(numerator, denominator)

    output = append_column(X_num, ratio_col)

    if cat_labels is not None:
        label_block = one_hot_encode(cat_labels)
        output = np.hstack([output, label_block])

    return output

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples=X.shape[0]
    idx=make_shuffled_indices(n_samples,seed)
    train_idx,val_idx,test_idx=partition_indices(idx,train_ratio,val_ratio)
    X_train,y_train=subset_xy(X,y,train_idx)
    X_val,y_val=subset_xy(X,y,val_idx)
    X_test,y_test=subset_xy(X,y,test_idx)
    dict={
        'X_train':X_train,'y_train':y_train,
        'X_val':X_val,'y_val':y_val,
        'X_test':X_test,'y_test':y_test
    }
    return dict

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X_train=splits['X_train']
    y_train=splits['y_train']
    X_val=splits['X_val']
    y_val=splits['y_val']
    X_test=splits['X_test']
    y_test=splits['y_test']
    train_mean,train_std=fit_standardizer(X_train)
    scaled_train=apply_standardizer(X_train,train_mean,train_std)
    scaled_val=apply_standardizer(X_val,train_mean,train_std)
    scaled_test=apply_standardizer(X_test,train_mean,train_std)
    scaled_train=add_bias_column(scaled_train)
    scaled_val=add_bias_column(scaled_val)
    scaled_test=add_bias_column(scaled_test)
    std_splits={
        'X_train':scaled_train,
        'X_val':scaled_val,
        'X_test':scaled_test,
        'y_train':y_train,
        'y_val':y_val,
        'y_test':y_test
    }
    return (std_splits,train_mean,train_std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    MAE=mean_absolute_error(y_true,y_pred)
    RMSE=root_mean_squared_error(y_true,y_pred)
    r_square=r_squared(y_true,y_pred)
    res=residual_summary(y_true,y_pred)
    dict={
        'mae':MAE,
        'rmse':RMSE,
        'r2':r_square,
        'residual_summary':res
    }
    return dict

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

