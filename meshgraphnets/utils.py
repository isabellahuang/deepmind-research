import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import trimesh
import tensorflow as tf 


import sys
sys.path.append('/home/isabella/deformable_object_grasping/graspsampling-py-internship1')
import graspsampling
from graspsampling import sampling, utilities, hands, io
sys.path.append('/home/isabella/deformable_object_grasping/isabella')
import data_creator_utils as dcu

def get_output_size(FLAGS):
    if predict_some_stress_only(FLAGS):
        return 1 
    elif predict_some_def_only(FLAGS):
        return 3
    else:
        return 4 # velocity and stress

def check_consistencies(FLAGS):
    if FLAGS.predict_stress_change_only:
        assert(FLAGS.gripper_force_action_input)

def predict_stress_and_def(FLAGS):
    if FLAGS.predict_stress_t_only and FLAGS.predict_pos_change_from_initial_only:
        return True 
    return False

def predict_some_stress_only(FLAGS):
    if (FLAGS.predict_log_stress_t_only or FLAGS.predict_log_stress_t1_only or FLAGS.predict_log_stress_change_only or FLAGS.predict_stress_change_only or FLAGS.predict_stress_t_only) and not predict_stress_and_def(FLAGS):
        assert(not FLAGS.predict_pos_change_only)
        return True
    return False

def predict_some_def_only(FLAGS):
    if FLAGS.predict_pos_change_from_initial_only and not predict_stress_and_def(FLAGS):
        return True 
    return False


def stress_t_as_node_feature(FLAGS):
    if FLAGS.predict_log_stress_t_only or FLAGS.predict_pos_change_only or FLAGS.predict_stress_t_only or FLAGS.predict_pos_change_from_initial_only:
        return False
    return True

def using_dm_dataset(FLAGS):
    return 'deforming_plate_data' in FLAGS.dataset_dir


def get_global_deformation_metrics(undeformed_mesh, deformed_mesh, get_field=False):
    """Get the mean and max deformation of the nodes over the entire mesh.

    Involves separating the pure deformation field from the raw displacement field.
    """

    # # Convert to numpy
    # undeformed_mesh = undeformed_mesh.numpy()
    # deformed_mesh = deformed_mesh.numpy()

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

  # return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)

  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted_top) + len(predicted_bottom)



def classification_accuracy_ranking(predicted, actual, percentile):
  ''' How many of the top and bottom 30 percent of predicted are in the top and bottom 50 of actual'''

  percentile = min(percentile, 100-percentile)

  actual_threshold = np.percentile(actual, percentile)
  pred_threshold = np.percentile(predicted, percentile)

  pred_threshold_top = np.percentile(predicted, 100-percentile)
  pred_threshold_bottom = np.percentile(predicted, percentile)
  num_g = len(predicted)
  assert(num_g == len(actual))

  actual_top = [k for k in range(num_g) if actual[k] >= actual_threshold]
  actual_bottom = [k for k in range(num_g) if actual[k] < actual_threshold]

  predicted_top = [k for k in range(num_g) if predicted[k] >= pred_threshold_top]
  predicted_bottom = [k for k in range(num_g) if predicted[k] < pred_threshold_bottom]

  predicted_bottom_correct = [k for k in predicted_bottom if k in actual_bottom]
  predicted_top_correct = [k for k in predicted_top if k in actual_top]

  # print("Actual", np.min(actual), threshold, np.max(actual))
  # print("Predicted", np.min(predicted), threshold, np.max(predicted))

  # quit()

  # return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted)
  return len(predicted_bottom_correct) + len(predicted_top_correct), len(predicted_top) + len(predicted_bottom)


def sample_grasps(obj_stl_file, number_of_grasps):

    grasp_sampling_object = utilities.instantiate_mesh(file=obj_stl_file, scale=1.0).bounding_box_oriented

    gripper = hands.create_gripper('panda_visual')

    cls_sampler = graspsampling.sampling.AntipodalSampler
    sampler = cls_sampler(gripper, grasp_sampling_object, 0.0, 4) # Antipodal

    results, grasp_success = graspsampling.sampling.collision_free_grasps(gripper, grasp_sampling_object, sampler, number_of_grasps)
    if len(results) == 0:
      print("Grasping sampling failed")
    results = np.asarray(results)
    print("+++++", np.mean(results))
    transformation_matrices = dcu.poses_wxyz_to_mats(results)
    return transformation_matrices

def get_mtx_from_h5(h5file):
    import h5py
    f = h5py.File(h5file, 'r')
    poses = f['poses'][:]
    f.close()
    transformation_matrices = dcu.poses_wxyz_to_mats(poses)
    return transformation_matrices

def tmtx_to_tensor(transform):

    gripper_r = R.from_matrix(transform[:3,:3])
    gripper_t = transform[:3, 3]
    gym_r = R.from_matrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    # Get total transformation
    total_mtx = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), transform[:3,:3])
    total_r = R.from_matrix(total_mtx)
    total_t = gym_r.apply(gripper_t) + [0, 1, 0]
    euler = total_r.as_euler('xyz')
    candidate_tfn = np.expand_dims(np.expand_dims(np.concatenate((euler, total_t)), 0), 0)
    candidate_tfn_tensor = tf.convert_to_tensor(candidate_tfn, dtype=tf.float32)
    candidate_tfn_tensor = tf.Variable(tf.tile(candidate_tfn_tensor, [48, 1, 1]))
    return candidate_tfn_tensor