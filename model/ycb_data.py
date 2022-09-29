import numpy as np

COSYPOSE2NAME = {
    "obj_000001": "master chef can",
    "obj_000002": "cracker box",
    "obj_000003": "sugar box",
    "obj_000004": "tomato soup can",
    "obj_000005": "mustard bottle",
    "obj_000006": "tuna fish can",
    "obj_000007": "pudding box",
    "obj_000008": "gelatin box",
    "obj_000009": "potted meat can",
    "obj_000010": "banana",
    "obj_000011": "pitcher base",
    "obj_000012": "bleach cleanser",
    "obj_000013": "bowl",
    "obj_000014": "mug",
    "obj_000015": "power drill",
    "obj_000016": "wood block",
    "obj_000017": "scissors",
    "obj_000018": "large marker",
    "obj_000019": "medium_clamp",
    "obj_000020": "large clamp",
    "obj_000021": "foam brick",
}

COSYPOSE_TRANSFORM = {
    "obj_000001": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000002": (np.array([0.0, 0.0, -10.075]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000003": (np.array([0.0, 0.0, -0.088008]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000004": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000005": (np.array([0.0, 0.0, -0.095704]), np.array(0.97904547248, 0.0, 0.0, 0.20364175114)),
    "obj_000006": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000007": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000008": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000009": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000010": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000011": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000012": (np.array([0.0, 0.0, -0.12532]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000013": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000014": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000015": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000016": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000017": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000018": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000019": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000020": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
    "obj_000021": (np.array([0.0, 0.0, 0.0]), np.array(0.0, 0.0, 0.0, 0.0)),
}

COSYPOSE_BBOX = {
    "obj_000005": {
        "x": 0.09598095715045929,
        "y": 0.058198802173137665,
        "z": 0.1913120150566101
    },
    "obj_000012": {
        "x": 0.10242000222206116,
        "y": 0.06771500408649445,
        "z": 0.2505960166454315
    }
}