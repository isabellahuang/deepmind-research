import numpy as np
from scipy.spatial.transform import Rotation as R


def predict_some_stress_only(FLAGS):
	if FLAGS.predict_log_stress_t_only or FLAGS.predict_log_stress_t1_only or FLAGS.predict_log_stress_change_only:
		return True
	return False


def stress_t_as_node_feature(FLAGS):
	if FLAGS.predict_log_stress_t_only or FLAGS.predict_pos_change_only:
		return False
	return True

def using_dm_dataset(FLAGS):
	return 'deforming_plate_data' in FLAGS.dataset_dir


def get_global_deformation_metrics(undeformed_mesh, deformed_mesh, get_field=False):
    """Get the mean and max deformation of the nodes over the entire mesh.

    Involves separating the pure deformation field from the raw displacement field.
    """
    num_nodes = undeformed_mesh.shape[0]
    undeformed_positions = undeformed_mesh[:, :3]
    deformed_positions = deformed_mesh[:, :3]

    centered_undeformed_positions = undeformed_positions - np.mean(undeformed_positions, axis=0)
    centered_deformed_positions = deformed_positions - np.mean(undeformed_positions, axis=0)

    # Extract deformations by rigid body motion
    axis_angle, t = rigid_body_motion(centered_undeformed_positions, centered_deformed_positions)
    rot = R.from_rotvec(axis_angle)
    aligned_deformed_positions = centered_deformed_positions
    for i in range(num_nodes):
        aligned_deformed_positions[i, :] = np.linalg.inv(
            rot.as_matrix()) @ (centered_deformed_positions[i, :] - t)
    centered_deformed_positions = aligned_deformed_positions

    def_field = centered_deformed_positions - centered_undeformed_positions
    deformation_norms = np.linalg.norm(def_field, axis=1)

    if get_field:
        return def_field
    return np.mean(deformation_norms), np.max(deformation_norms), np.median(deformation_norms)

def rigid_body_motion(P, Q):
    """Return best-fit rigid body motion from point sets P to Q."""
    P = np.transpose(P)  # Previous positions
    Q = np.transpose(Q)  # Current positions
    n = P.shape[1]

    # Center everything in the middle
    origin_offset = np.vstack(P.mean(axis=1))
    P = P - origin_offset
    Q = Q - origin_offset

    # Compute the weight centroids of both point sets
    P_mean = P.mean(axis=1)
    Q_mean = Q.mean(axis=1)

    # Compute the centered vectors
    X = P - np.matrix(P_mean).T
    Y = Q - np.matrix(Q_mean).T
    W = np.diag(np.ones(n) / n)

    # Compute the covariance matrix
    S = X @ W @ Y.T

    # Get the SVD, S factors as U @ np.diag(Sig) @ Vh
    U, Sig, Vh = np.linalg.svd(S, full_matrices=True)

    # Optimal rotation and translation
    d = np.linalg.det(Vh.T @ U.T)
    Rot = Vh.T @ np.diag([1, 1, d]) @ U.T
    t = Q_mean - Rot @ P_mean

    Rot_scipy = R.from_matrix(Rot)
    axis_angle = Rot_scipy.as_rotvec()

    return axis_angle, np.asarray(t)[0]