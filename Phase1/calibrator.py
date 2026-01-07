import numpy as np


class FaceSimilarityGT:
    """
    Threshold-based face verifier (NO ML MODEL)

    Uses control statements with cosine similarity thresholds:
    - >= 0.30 → MATCH (label=1)
    - <= 0.10 → NO_MATCH (label=0)
    - else    → UNCERTAIN (label=0.5)

    INPUT  (7 features):
        [cos, ang, euc, euc_n, man, cheb, corr]

    OUTPUT:
        - probability: descriptive evidence score (0–100)
        - similarity: GT-style weighted score (0–100)
        - decision: MATCH / NO_MATCH / UNCERTAIN / UNUSABLE
        - label: similarity description (NOT identity)
    """

    def __init__(self, match_thr=0.30, likely_thr=0.25, nomatch_thr=0.10, quality_reject_thr=0.40):
        self.match_thr = match_thr
        self.likely_thr = likely_thr
        self.nomatch_thr = nomatch_thr
        self.quality_reject_thr = quality_reject_thr

    # --------------------------------------------------
    # GT SIMILARITY (UNCHANGED)
    # --------------------------------------------------
    @staticmethod
    def compute_gt_similarity(features_7, label):
        cos  = features_7[0]
        euc  = features_7[2]
        man  = features_7[4]
        cheb = features_7[5]
        corr = features_7[6]

        euc_s  = 1 / (1 + euc)
        man_s  = 1 / (1 + man)
        cheb_s = 1 / (1 + cheb)

        score = (
            0.45 * cos +
            0.25 * corr +
            0.15 * euc_s +
            0.10 * man_s +
            0.05 * cheb_s
        )

        score = np.clip(score, 0.0, 1.0)

        if label == 1:
            return 90.0 + 15.0 * score
        elif label == 0.5:
            return 75.0 + 10.0 * score
        elif label == 0.75:
            return 85.0 + 15.0 * score
        else:
            return 10.0 * (1.0 - score)

    # --------------------------------------------------
    # DECISION (UNCHANGED)
    # --------------------------------------------------
    def make_decision(self, cosine_sim):
        if cosine_sim >= self.match_thr:
            return 1, "MATCH"
        elif cosine_sim <= self.nomatch_thr:
            return 0, "NO_MATCH"
        elif self.nomatch_thr < cosine_sim < self.likely_thr:
            return 0.75, "LIKELY"
        else:
            return 0.5, "UNCERTAIN"

    # --------------------------------------------------
    # SAFE SIMILARITY DESCRIPTION
    # --------------------------------------------------
    @staticmethod
    def classify_by_probability(probability):
        if probability >= 90:
            return "Very high facial similarity"
        elif probability >= 80:
            return "High facial similarity"
        elif probability >= 70:
            return "Moderate facial similarity"
        elif probability >= 50:
            return "Low facial similarity"
        else:
            return "Very low facial similarity"

    # --------------------------------------------------
    # HELPER: position inside UNCERTAIN band
    # --------------------------------------------------
    @staticmethod
    def uncertain_position(cos, low=0.10, high=0.30):
        return np.clip((cos - low) / (high - low), 0.0, 1.0)

    # --------------------------------------------------
    # MAIN PREDICT FUNCTION
    # --------------------------------------------------
    def predict(self, features_7, quality_info=None):

        features_7 = np.asarray(features_7, dtype="float32")
        if features_7.shape[0] != 7:
            raise ValueError("Expected exactly 7 input features")

        # Quality rejection (SAFE)
        if quality_info:
            worst_q = quality_info.get("worst_quality", 1.0)
            if worst_q < self.quality_reject_thr:
                return {
                    "decision": "UNUSABLE",
                    "probability": None,
                    "similarity": None,
                    "label": None,
                    "quality_info": quality_info
                }

        cos = float(features_7[0])

        # Decision (UNCHANGED)
        label, decision = self.make_decision(cos)

        # --------------------------------------------------
        # SAFE PROBABILITY CALIBRATION (NEW)
        # --------------------------------------------------
        if label == 1:  # MATCH
            probability = 95.0 + 10.0 * np.clip(cos, 0.0, 1.0)
        
        elif label == 0.75:  # LIKELY
            l_pos = self.uncertain_position(cos, low=self.likely_thr, high=self.match_thr)
            probability = 75.1 + 30.0 * np.clip(cos, 0.0, 1.0)   # 80 → 90

        elif label == 0.5:  # UNCERTAIN
            u_pos = self.uncertain_position(cos)
            probability = 50.0 + 20.0 * u_pos   # 50 → 80

        else:  # NO_MATCH
            probability = 10.0 * (1.0 - np.clip(cos, 0.0, 1.0))

        probability = float(np.clip(probability, 0.0, 100.0))

        # Similarity score (UNCHANGED)
        similarity = self.compute_gt_similarity(features_7, label)

        # Descriptive label (SAFE)
        similarity_label = self.classify_by_probability(probability)

        result = {
            "decision": decision,
            "probability": round(probability, 2),
            "similarity": round(similarity, 2),
            "label": similarity_label
        }

        if quality_info:
            result["quality_info"] = quality_info

        return result
