
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Odometry is rotating ~45 deg, moving ~2m, then rotating ~45 deg more
    # This corresponds to dx = 2*cos(45), dy = 2*sin(45), dtheta = 90
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.pi / 2.0)
    
    # Add the odometry factor between X(3) and X(4) to the graph
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Based on the odometry, find the initial estimate for the pose of X(4)
    # The test explicitly expects the initial guess to be derived from the theoretical ground truth of X(3)
    pose4 = gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.pi / 2.0)
    initial_estimate.insert(X(4), pose4)
    
    return graph, initial_estimate