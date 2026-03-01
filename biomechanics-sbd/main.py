# def main():
#     print("Hello from biomechanics-sbd!")


# if __name__ == "__main__":
#     main()



import math
import numpy as np

def calculate_angle(a, b, c) -> float:
    """
    Angle at vertex *b* (degrees), formed by vectors b→a and b→c.
    Returns 0.0 when points are coincident.
    """
    a, b, c = (np.array(p, dtype=float) for p in (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cos_a))), 1)


print(calculate_angle((0,0),(1,1),(2,0)))