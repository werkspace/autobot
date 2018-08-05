import numpy as np


class LineModel(object):
    """
    A linear model 
    """
    def __init__(self, slope, intercept, points, fitted=None):
        self.slope = slope
        self.interceipt = intercept
        self.points = points
        self.fitted = fitted


def pca_proj(x, y):
    """
    Project a 2D dataset onto its first component using PCA.
    This is the equivalent of fitting a total least square regression line.

        :param x: a numpy array of the x cordinate 
        :param y: a numpy array of the y cordinate
    """
    X = np.vstack((x,y)).T

    U, S, V_T = np.linalg.svd(X - X.mean(axis=0))
    X_pca = (X - X.mean(axis=0)).dot(V_T[0].T)
    X_projected = X_pca[:, None].dot(V_T[0, None]) + X.mean(axis=0)

    return X_projected[:,0], X_projected[:,1]


def ransac_line_extraction(data, k_neighbours, degrees_range, consensus, tolerance, n_iter=1000):
    """
    Extract lines landmark in a set of 2D cartesian coordinates using random sampling consensus (RANSAC) to fit 
        :param data: 
        :param k_neighbours: 
        :param degrees_range: 
        :param consensus: 
        :param tolerance: 
        :param n_iter=1000: 

    Returns an array of Line_models objects.
    """ 

    i = 0
    associated_readings = np.array([[],[],[],[]])
    not_associated_readings = data
    associated_readings_proportion = 0
    line_models = []

    while i < n_iter and associated_readings_proportion < .90:
        i += 1
       
        # Pick candidates
        # 1 - Randomly pick one reference data point among candidates :
        # 2 - Get every candidates within D degrees from reference point
        # 3 - Keep K at most random among those within the range 

        reference_candidate_idx = np.random.randint(0, not_associated_readings.shape[1])

        is_within_range =  np.abs(not_associated_readings[3, :] - not_associated_readings[3, reference_candidate_idx]) < degrees_range
    
        candidates_idx = np.random.choice(is_within_range.nonzero()[0], 
                                        size=min(k_neighbours, is_within_range.nonzero()[0].shape[0]), 
                                        replace=False)    

        candidates = not_associated_readings[:, candidates_idx]
            
        # Fit a model.
        # 1 - Total least square (via pca projection)
        # 2 - Compute squared distance to the fitted line and test for tolerance
        # 3 - If enough items fit the model (consensus), refit the model with the best candidates only.
        # 4 - Update dataset by removing associated points.

        fitted_x, fitted_y = pca_proj(candidates[0,:], candidates[1,:])
        slope = (fitted_y[-1] - fitted_y[0]) / (fitted_x[-1] - fitted_x[0])
        intercept =  fitted_y[-1] - fitted_x[-1] * slope
        
        sq_dist = (fitted_x - candidates[0,:])**2 + (fitted_y - candidates[1,:])**2
        is_below_tolerance = (sq_dist <= tolerance)

        if is_below_tolerance.sum() > consensus:
            
            fitted_x, fitted_y = pca_proj(candidates[0, is_below_tolerance], candidates[1, is_below_tolerance])

            slope = (fitted_y[-1] - fitted_y[0]) / (fitted_x[-1] - fitted_x[0])
            intercept =  fitted_y[-1] - fitted_x[-1] * slope
            
            # Store the model and related points
            line_models += [LineModel(slope, intercept, candidates[:, is_below_tolerance], np.vstack([fitted_x, fitted_y]))] 
            associated_readings =  np.hstack([associated_readings, candidates[:, is_below_tolerance]])
            
            # First remove all candidates
            not_associated_readings = np.delete(not_associated_readings, candidates_idx, axis=1)

            # And add back the worst fit candidates
            not_associated_readings = np.hstack([not_associated_readings, candidates[:, np.logical_not(is_below_tolerance)]])

            associated_readings_proportion = float(associated_readings.shape[1]) / float(data.shape[1])
    
    return line_models