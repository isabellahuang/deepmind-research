import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import trimesh


def get_output_size(FLAGS):
	if predict_some_stress_only(FLAGS):
		return 1 
	else:
		return 4 # velocity and stress

def check_consistencies(FLAGS):
	if FLAGS.predict_stress_change_only:
		assert(FLAGS.gripper_force_action_input)


def predict_some_stress_only(FLAGS):
	if FLAGS.predict_log_stress_t_only or FLAGS.predict_log_stress_t1_only or FLAGS.predict_log_stress_change_only or FLAGS.predict_stress_change_only or FLAGS.predict_stress_t_only:
		assert(not FLAGS.predict_pos_change_only)
		return True
	return False


def stress_t_as_node_feature(FLAGS):
	if FLAGS.predict_log_stress_t_only or FLAGS.predict_pos_change_only or FLAGS.predict_stress_t_only:
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


def open_gripper_at_pose(tfn):
  ''' Using numpy instead of tf operations.'''

  # Load original finger positions
  finger1_path = os.path.join('meshgraphnets', 'assets', 'finger1_face_uniform' + '.stl')
  f1_trimesh = trimesh.load_mesh(finger1_path)
  f1_verts_original = f1_trimesh.vertices

  finger2_path = os.path.join('meshgraphnets', 'assets', 'finger2_face_uniform' + '.stl')
  f2_trimesh = trimesh.load_mesh(finger2_path)
  f2_verts_original = f2_trimesh.vertices
  f_pc_original = np.concatenate((f1_verts_original, f2_verts_original), axis=0)



  # Load transformation params (euler and translation)
  euler, trans = np.split(np.squeeze(tfn), 2, axis=0)


  tf_from_euler = R.from_euler('xyz', euler)
  f_pc = tf_from_euler.apply(f_pc_original) + trans
  original_normal = np.array([1, 0, 0])
  gripper_normal = tf_from_euler.apply(original_normal)
  f1_verts, f2_verts = np.split(f_pc, 2, axis=0)


  # tf_from_euler = tfg_transformation.rotation_matrix_3d.from_euler(euler)
  # f_pc = tfg_transformation.rotation_matrix_3d.rotate(f_pc_original, tf_from_euler) + trans
  # original_normal = tf.constant([1., 0., 0.], dtype=tf.float32)
  # gripper_normal = tfg_transformation.rotation_matrix_3d.rotate(original_normal, tf_from_euler)
  # f1_verts, f2_verts = tf.split(f_pc, 2, axis=0)

  return f1_verts, f2_verts, gripper_normal

def f_verts_at_pos(tfn, gripper_pos):
  f1_verts, f2_verts, gripper_normal = open_gripper_at_pose(tfn)
  f1_verts_closed = f1_verts -  gripper_normal * (0.04 - gripper_pos)
  f2_verts_closed = f2_verts + gripper_normal * (0.04 - gripper_pos)
  f_verts = np.concatenate((f1_verts_closed, f2_verts_closed), axis=0)
  return f_verts
  


def classification_accuracy_threshold(predicted, actual, percentile):

  threshold = np.percentile(actual, percentile)
  num_g = len(predicted)
  assert(num_g == len(actual))

  actual_top = [k for k in range(num_g) if actual[k] >= threshold]
  actual_bottom = [k for k in range(num_g) if actual[k] < threshold]

  predicted_top = [k for k in range(num_g) if predicted[k] >= threshold]
  predicted_bottom = [k for k in range(num_g) if predicted[k] < threshold]

  predicted_bottom_correct = [k for k in predicted_bottom if k in actual_bottom]
  predicted_top_correct = [k for k in predicted_top if k in actual_top]

  # print("Actual", np.min(actual), threshold, np.max(actual))
  # print("Predicted", np.min(predicted), threshold, np.max(predicted))

  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)

def classification_accuracy_ranking(predicted, actual, percentile):
  actual_threshold = np.percentile(actual, percentile)
  pred_threshold = np.percentile(predicted, percentile)
  num_g = len(predicted)
  assert(num_g == len(actual))

  actual_top = [k for k in range(num_g) if actual[k] >= actual_threshold]
  actual_bottom = [k for k in range(num_g) if actual[k] < actual_threshold]

  predicted_top = [k for k in range(num_g) if predicted[k] >= pred_threshold]
  predicted_bottom = [k for k in range(num_g) if predicted[k] < pred_threshold]

  predicted_bottom_correct = [k for k in predicted_bottom if k in actual_bottom]
  predicted_top_correct = [k for k in predicted_top if k in actual_top]

  # print("Actual", np.min(actual), threshold, np.max(actual))
  # print("Predicted", np.min(predicted), threshold, np.max(predicted))
  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)