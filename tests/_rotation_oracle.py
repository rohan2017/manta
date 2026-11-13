"""Independent matrix Rodrigues oracle for small-angle INS derivative tests."""

import numpy as np


class Rotation:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    @classmethod
    def identity(cls):
        return cls(np.eye(3))

    @classmethod
    def from_rotvec(cls, vector):
        v = np.asarray(vector, dtype=float)
        angle = np.linalg.norm(v)
        x, y, z = v
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return cls(
            np.eye(3)
            + np.sinc(angle / np.pi) * skew
            + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * skew @ skew
        )

    @classmethod
    def from_quat(cls, q, *, scalar_first):
        assert scalar_first
        w, x, y, z = np.asarray(q) / np.linalg.norm(q)
        return cls(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def as_rotvec(self):
        R = self.matrix
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
        angle = np.arctan2(np.linalg.norm(v), (np.trace(R) - 1) / 2)
        assert angle < np.pi / 2, "this test oracle only covers small rotations"
        return v / np.sinc(angle / np.pi)

    def as_quat(self, *, scalar_first):
        assert scalar_first
        v = self.as_rotvec()
        angle = np.linalg.norm(v)
        return np.r_[np.cos(angle / 2), 0.5 * np.sinc(angle / (2 * np.pi)) * v]

    def apply(self, vector):
        return self.matrix @ vector

    def inv(self):
        return Rotation(self.matrix.T)

    def __mul__(self, other):
        return Rotation(self.matrix @ other.matrix)
