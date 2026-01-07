import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean, cityblock, chebyshev
from scipy.stats import pearsonr


class FaceFeatureExtractor:
    """
    Converts two face embeddings into an 8D GT-aligned feature vector.
    """

    @staticmethod
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

    def extract(self, emb1, emb2, strict=True):
        """
        Parameters
        ----------
        emb1 : np.ndarray (512,)
        emb2 : np.ndarray (512,)

        Returns
        -------
        features : np.ndarray (8,)
            [cos, ang, euc, euc_n, man, cheb, corr, cos_margin]
        """

        # ------------------------------
        # Cosine & angular distance
        # ------------------------------
        cos = self.cosine_sim(emb1, emb2)
        cos = np.clip(cos, -1.0, 1.0)

        ang = np.arccos(cos)  # ArcFace-native

        # ------------------------------
        # Distances
        # ------------------------------
        euc = euclidean(emb1, emb2)
        man = cityblock(emb1, emb2)
        cheb = chebyshev(emb1, emb2)

        # ------------------------------
        # Correlation
        # ------------------------------
        corr, _ = pearsonr(emb1, emb2)
        corr = np.nan_to_num(corr)

        # ------------------------------
        # Normalized / margin features
        # ------------------------------
        euc_n = euc / (norm(emb1) + norm(emb2) + 1e-8)
        cos_margin = 1.0 - cos

        return np.array(
            [
                cos,        # 0
                ang,        # 1
                euc,        # 2
                euc_n,      # 3
                man,        # 4
                cheb,       # 5
                corr,       # 6
                cos_margin  # 7
            ],
            dtype="float32"
        )
