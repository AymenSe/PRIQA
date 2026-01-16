from scipy import stats
from scipy.optimize import curve_fit
import numpy as np

def format_metrics(metrics, digits=4):
    formatted = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            formatted[k] = round(float(v), digits)
        elif isinstance(v, (np.integer, int)):
            formatted[k] = int(v)
        else:
            formatted[k] = v
    return formatted

def fit_curve(x, y, curve_type='logistic_4params'):
    r'''Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.

    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m

    '''
    assert curve_type in [
        'logistic_4params', 'logistic_5params'], f'curve type should be in [logistic_4params, logistic_5params], but got {curve_type}.'

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(- (x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1. / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == 'logistic_4params':
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == 'logistic_5params':
        logistic = logistic_5params
        betas_init = betas_init_5params
    try :
        betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=500000)
    except:
        print("Warning: Curve fitting failed, using initial parameters.")
        betas = betas_init
    yhat = logistic(x, *betas)
    return yhat

def calculate_rmse(x, y, fit_scale=None, eps=1e-8):
    rmse = np.sqrt(np.mean((x - y) ** 2) + eps)
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
        rmse_2 = np.sqrt(np.mean((x - y) ** 2) + eps)
        return rmse, rmse_2
    return rmse

def calculate_plcc(x, y, fit_scale=None):
    plcc = stats.pearsonr(x, y)[0]
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
        plcc_2 = stats.pearsonr(x, y)[0]
        return plcc, plcc_2
    return plcc

def calculate_srcc(x, y, fit_scale=None):
    srocc = stats.spearmanr(x, y)[0]
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
        srocc_2 = stats.spearmanr(x, y)[0]
        return srocc, srocc_2
    return srocc

def calculate_krcc(x, y, fit_scale=None):
    krcc = stats.kendalltau(x, y)[0]
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
        krcc_2 = stats.kendalltau(x, y)[0]
        return krcc, krcc_2
    return krcc



def compute_metrics(predicted_scores, mos_scores, fit_scale=None):
    """
    Compute RMSE, PLCC, SROCC, and KRCC between predicted scores and MOS scores.
    
    Parameters:
    - predicted_scores (list): List of predicted scores.
    - mos_scores (list): List of ground truth MOS scores.
    - fit_scale (str): Type of curve fitting to apply to predicted scores.

    Returns:
    - metrics (dict): Dictionary containing RMSE, PLCC, SROCC, and KRCC.
    """
    metrics = {}
    if fit_scale is not None:
        metrics["RMSE"], metrics["RMSE_2"] = calculate_rmse(predicted_scores, mos_scores, fit_scale=fit_scale)
        metrics["PLCC"], metrics["PLCC_2"] = calculate_plcc(predicted_scores, mos_scores, fit_scale=fit_scale)
        metrics["SROCC"], metrics["SROCC_2"] = calculate_srcc(predicted_scores, mos_scores, fit_scale=fit_scale)
        metrics["KRCC"], metrics["KRCC_2"] = calculate_krcc(predicted_scores, mos_scores, fit_scale=fit_scale)
    else:            
        metrics["RMSE"] = calculate_rmse(predicted_scores, mos_scores)
        metrics["PLCC"] = calculate_plcc(predicted_scores, mos_scores)
        metrics["SROCC"] = calculate_srcc(predicted_scores, mos_scores)
        metrics["KRCC"] = calculate_krcc(predicted_scores, mos_scores)
    return metrics