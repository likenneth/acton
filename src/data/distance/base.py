import numpy as np

class SnippetDistance():
    def __init__(self, TYPE, TRANSLATION, ROTATION):
        self.type = TYPE
        self.translation = TRANSLATION  # TRANSLATION True means action, to normalize it by translation invariance
        self.rotation = ROTATION
    def __call__(self, a, b):
        ttl = a.shape[0]
        pcd = []
        for x in [a, b]:
            if self.translation:
                # 33: left hip x coordinates, before it there are 11 joints
                # then is x, y, z and right hip
                body_centre_x = (x[33] + x[36]) / 2
                body_centre_y = (x[34] + x[37]) / 2
                body_centre_z = (x[35] + x[38]) / 2
                shift = np.tile(np.array([body_centre_x, body_centre_y, body_centre_z]), ttl // 3)
                x = x - shift
            if self.rotation:
                # using Euler–Rodrigues formula, partially adopted from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
                lh = x[33:36]
                axis = np.array([0, -lh[2], lh[1]])
                theta = -np.arccos(lh[0] / np.sqrt(np.dot(lh, lh)))
                axis = axis / np.sqrt(np.dot(axis, axis))
                a = np.cos(theta / 2.0)
                b, c, d = -axis * np.sin(theta / 2.0)
                aa, bb, cc, dd = a * a, b * b, c * c, d * d
                bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                ttt = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                x = (ttt @ x.reshape(-1, 3).T).T.flatten()
            pcd.append(x)
        return np.linalg.norm(pcd[0] - pcd[1])
