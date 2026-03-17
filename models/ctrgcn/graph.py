"""
OpenPose-18 graph adjacency matrices for CTR-GCN.
This layout MUST match the Kinetics-400 pretrained checkpoint.
Do NOT change to COCO-17 here — that would invalidate pretrained weights.
"""
import numpy as np

NUM_JOINTS = 18
CENTER     = 1   # neck

BONES = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17)
]

FLIP_PAIRS = [(2,5),(3,6),(4,7),(8,11),(9,12),(10,13),(14,15),(16,17)]

class OpenPoseGraph:
    """Spatial partitioning adjacency (3 subsets) for CTR-GCN."""

    def __init__(self, strategy: str = "spatial", max_hop: int = 1):
        self.num_nodes = NUM_JOINTS
        self.A = self._build(strategy, max_hop)

    def _hop_distance(self) -> np.ndarray:
        D = np.full((NUM_JOINTS, NUM_JOINTS), np.inf)
        np.fill_diagonal(D, 0)
        for i, j in BONES:
            D[i,j] = D[j,i] = 1
        for k in range(NUM_JOINTS):
            for i in range(NUM_JOINTS):
                for j in range(NUM_JOINTS):
                    if D[i,k] + D[k,j] < D[i,j]:
                        D[i,j] = D[i,k] + D[k,j]
        return D

    def _normalise(self, A: np.ndarray) -> np.ndarray:
        Dl = A.sum(1).clip(min=1) ** -0.5
        return np.diag(Dl) @ A @ np.diag(Dl)

    def _build(self, strategy: str, max_hop: int) -> np.ndarray:
        D = self._hop_distance()
        valid = D <= max_hop

        if strategy == "spatial":
            A = np.zeros((3, NUM_JOINTS, NUM_JOINTS))
            for i in range(NUM_JOINTS):
                for j in range(NUM_JOINTS):
                    if not valid[i,j]: continue
                    if D[i,j] == 0:
                        A[0,i,j] = 1
                    elif D[j,CENTER] < D[i,CENTER]:
                        A[1,i,j] = 1
                    else:
                        A[2,i,j] = 1
            for k in range(3):
                A[k] = self._normalise(A[k])
            return A   # (3, 18, 18)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
