# third-party

from sklearn.mixture import GaussianMixture


def optimize_thr_GMM(pred):
    ''' Employes a Gaussian Mixture Model (GMM) to model the distribution of predicted probabilities. Given the tendency of the model to produce low probability scores, we assume a bimodal distribution corresponding to low- and high-confidence predictions. The GMM is fitted with two components, and the threshold was set at the intersection of the Gaussian distributions.  

    Parameters
    ---------- 
    pred : np.array, shape ()
        Predicted probabilities to which the GMM will be fitted.

    Returns
    -------
    optimal_thr : float
        Optimized high-probability threshold.
    '''

    print('Optimizing high-probability threshold...')
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(pred)
    means = gmm.means_.flatten()
    optimal_thr = min(means) + 0.5 * (max(means) - min(means))

    print(f'\t...threshold: {optimal_thr}')
    return optimal_thr
