'''
adapted module for predicting 3d poses from 2d joints.
see the meaning of configurations in README.md
'''

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import cameras
import data_utils
import linear_model
import procrustes
import viz
from config import config, load_config, NUM_CAMERAS
from data_utils_mod import delete_depth_coordinate, store_results, set_test_folders, set_model_folder
from collections import OrderedDict
import copy
import argparse
import data_utils_mod

FLAGS = None
current_folder = os.path.dirname(__file__)

def initialize_flags():    
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
    tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")
    tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
    tf.app.flags.DEFINE_integer("epochs", 100, "How many epochs we should train for")
    tf.app.flags.DEFINE_boolean("camera_frame", config.CAMERA_FRAME, "Convert 3d poses to camera coordinates")
    tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
    tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")
    
    # Data loading
    tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
    tf.app.flags.DEFINE_string("action", config.ACTION, "The action to train on. 'All' means all the actions")
    
    # Architecture
    tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
    tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")
    
    # Evaluation
    tf.app.flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")
    tf.app.flags.DEFINE_boolean("evaluateActionWise", True, "The dataset to use either h36m or heva")
    
    # Directories
    tf.app.flags.DEFINE_string("cameras_path", os.path.join(current_folder, "..", "data", "h36m", "metadata.xml"), "File with h36m metadata, including cameras")
    tf.app.flags.DEFINE_string("data_dir", os.path.join(current_folder, "..", "data", "h36m"), "Data directory")
    tf.app.flags.DEFINE_string("train_dir", os.path.join(model_folder, "experiments"), "Training directory.")
    
    # Train or load
    tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
    tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
    tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.") # 2437100 71600
    
    # Misc
    tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
    
    global FLAGS
    FLAGS = tf.app.flags.FLAGS


def create_model( session, actions, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """
  
  train_dir = os.path.join( FLAGS.train_dir, config.MODEL_NAME,
    'dropout_{0}'.format(FLAGS.dropout),
    'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
    'lr_{0}'.format(FLAGS.learning_rate),
    'residual' if FLAGS.residual else 'not_residual',
    'depth_{0}'.format(FLAGS.num_layers),
    'linear_size{0}'.format(FLAGS.linear_size),
    'batch_size_{0}'.format(FLAGS.batch_size),
    'procrustes' if FLAGS.procrustes else 'no_procrustes',
    'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
    'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
    'predict_14' if FLAGS.predict_14 else 'predict_17')
    
  summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

  # To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
  os.system('mkdir -p {}'.format(summaries_dir))

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  print("train_dir", train_dir )

  if FLAGS.load == 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.compat.v1.global_variables_initializer() )
    return model, train_dir
  
  elif FLAGS.load == -1:
    print("Loading model from the latest checkpoint.")
    model_checkpoint_path = tf.train.latest_checkpoint(train_dir, latest_filename="checkpoint")
    model.saver.restore( session, model_checkpoint_path )
                         
    return model, train_dir
  
  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model, train_dir

  print("Could not find checkpoint. Aborting.")
  raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )


def train():
  # always train from scratch  
  FLAGS.load = 0
    
  """Train a linear model for 3d pose estimation"""

  actions = data_utils.define_actions( FLAGS.action )

  number_of_actions = len( actions )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  this_file = os.path.dirname(os.path.realpath(__file__))
  
  if (FLAGS.camera_frame):
    rcams = cameras.load_cameras(os.path.join(this_file, "..", FLAGS.cameras_path), SUBJECT_IDS)
  else :
    rcams = {}

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  # Read groundtruth 2D projections
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )
  print( "done reading and normalizing data." )

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(
    config=tf.ConfigProto(
      device_count=device_count,
      allow_soft_placement=True
    )
  ) as sess:

    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model, train_dir = create_model( sess, actions, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100

    for _ in range( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches( train_set_2d, train_set_3d, FLAGS.camera_frame, training=True )
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batches
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False

      if FLAGS.evaluateActionWise:

        print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs

        cum_err = 0
        for action in actions:

          print("{0:<12} ".format(action), end="")
          # Get 2d and 3d testing data for this action
          action_test_set_2d = get_action_subset( test_set_2d, action )
          action_test_set_3d = get_action_subset( test_set_3d, action )
          encoder_inputs, decoder_outputs = model.get_all_batches( action_test_set_2d, action_test_set_3d, FLAGS.camera_frame, training=False)

          act_err, _, step_time, loss = evaluate_batches( sess, model,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
            current_step, encoder_inputs, decoder_outputs )
          cum_err = cum_err + act_err

          print("{0:>6.2f}".format(act_err))

        summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
        model.test_writer.add_summary( summaries, current_step )
        print("{0:<12} {1:>6.2f}".format("Average", cum_err/float(len(actions) )))
        print("{0:=^19}".format(''))

      else:

        n_joints = 17 if not(FLAGS.predict_14) else 14
        encoder_inputs, decoder_outputs = model.get_all_batches( test_set_2d, test_set_3d, FLAGS.camera_frame, training=False)

        total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
          data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
          data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
          current_step, encoder_inputs, decoder_outputs, current_epoch )

        print("=============================\n"
              "Step-time (ms):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % ( 1000*step_time, loss, total_err ))

        for i in range(n_joints):
          # 6 spaces, right-aligned, 5 decimal places
          print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
        print("=============================")

        # Log the error to tensorboard
        summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
        model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}


def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess: tensorflow session
    model: tensorflow model to run evaluation with
    data_mean_3d: the mean of the training data in 3d
    data_std_3d: the standard deviation of the training data in 3d
    dim_to_use_3d: out of all the 96 dimensions that represent a 3d body in h36m, compute results for this subset
    dim_to_ignore_3d: complelment of the above
    data_mean_2d: mean of the training data in 2d
    data_std_2d: standard deviation of the training data in 2d
    dim_to_use_2d: out of the 64 dimensions that represent a body in 2d in h35m, use this subset
    dim_to_ignore_2d: complement of the above
    current_step: training iteration step
    encoder_inputs: input for the network
    decoder_outputs: expected output for the network
    current_epoch: current training epoch
  Returns
    total_err: average mm error over all joints
    joint_err: average mm error per joint
    step_time: time it took to evaluate one batch
    loss: validation loss of the network
  """

  n_joints = (config.NUM_JOINTS + 1) if not(FLAGS.predict_14) else 14
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    root_index = config.ROOT_INDEX
    hip_coords = [root_index*3, root_index*3+1, root_index*3+2]
    
    dtu3d = np.hstack( (hip_coords, dim_to_use_3d) ) if not(FLAGS.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]

    assert dec_out.shape[0] == FLAGS.batch_size
    assert poses3d.shape[0] == FLAGS.batch_size

    if FLAGS.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(FLAGS.batch_size):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c

        poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(FLAGS.predict_14) else np.reshape(out,[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss


def sample(visualization=True):
  # always load the latest model  
  FLAGS.load = -1
    
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  this_file = os.path.dirname(os.path.realpath(__file__))
  
  if (FLAGS.camera_frame):
    rcams = cameras.load_cameras(os.path.join(this_file, "..", FLAGS.cameras_path), SUBJECT_IDS)
  else :
    rcams = {}

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams)
  print( "done reading and normalizing data." )

  results_3d = OrderedDict()
  
  if (config.MODEL_NAME.find("_inf_") == -1):
    normalization_directory = os.path.join(model_folder, "normalizations", config.MODEL_NAME)  
      
    if not os.path.exists(normalization_directory):
      os.mkdir(normalization_directory)  
      
      np.save(os.path.join(normalization_directory, "data_mean_3d.npy"), data_mean_3d)
      np.save(os.path.join(normalization_directory, "data_mean_2d.npy"), data_mean_2d)
      np.save(os.path.join(normalization_directory, "data_std_3d.npy"), data_std_3d)
      np.save(os.path.join(normalization_directory, "data_std_2d.npy"), data_std_2d)
   

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = 128
    model, _ = create_model(sess, actions, batch_size)
    print("Model loaded")

    for key2d in test_set_2d.keys():

      (subj, b, fname) = key2d
      print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.cdf'.format(fname.split('.')[0]))
      # key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and FLAGS.camera_frame else key3d

      enc_in  = test_set_2d[ key2d ]
      n2d, _ = enc_in.shape
      dec_out = test_set_3d[ key3d ]
      n3d, _ = dec_out.shape
      assert n2d == n3d

      # Split into about-same-size batches
      enc_in   = np.array_split( enc_in,  n2d // batch_size ) if n2d > batch_size else [enc_in]
      dec_out  = np.array_split( dec_out, n3d // batch_size ) if n3d > batch_size else [dec_out]
      all_poses_3d = []

      for bidx in range( len(enc_in) ):

        # Dropout probability 0 (keep probability 1) for sampling
        dp = 1.0
        _, _, poses3d = model.step(sess, enc_in[bidx], dec_out[bidx], dp, isTraining=False)

        # denormalize
        enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
        dec_out[bidx] = data_utils.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d )
        poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
        all_poses_3d.append( poses3d )

      # Put all the poses together
      enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, all_poses_3d] )

      # if key3d == (9, 'Directions', 'Directions.54138969.h5'):
      #   print("unnormalized")
      #   print(dec_out[0][24:27])

      # Convert back to world coordinates
      if FLAGS.camera_frame:
        N_CAMERAS = NUM_CAMERAS
        N_JOINTS_H36M = 32

        # Add global position back
        dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1,N_JOINTS_H36M] )
        
        # if key3d == (9, 'Directions', 'Directions.54138969.h5'):
        #   print("unshifted")
        #   print(dec_out[0][24:27])
        
        poses3d = poses3d + np.tile( test_root_positions[ key3d ], [1, N_JOINTS_H36M] )

        # Load the appropriate camera
        subj, _, sname = key3d

        cname = sname.split('.')[1] # <-- camera name
        scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
        scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
        the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
        R, T, f, c, k, p, name = the_cam
        assert name == cname

        def cam2world_centered(data_3d_camframe):
          data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
          data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
          # subtract root translation
          return data_3d_worldframe # - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

        # Apply inverse rotation and translation
        dec_out = cam2world_centered(dec_out)
        poses3d = cam2world_centered(poses3d)
        
        # if key3d == (9, 'Directions', 'Directions.54138969.h5'):
        #     print("in world coordinates")
        #     print (dec_out[0][24:27])
        #     print (poses3d[0][24:27])
        
      else :
        N_JOINTS_H36M = 32
        
        poses3d = poses3d + np.tile( test_root_positions[ key3d ], [1, N_JOINTS_H36M] )
        dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1, N_JOINTS_H36M] )
        
      results_3d[key3d] = {
        "true": dec_out,
        "pred": poses3d    
      }

  # Grab a random batch to visualize
  enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
  enc_in = delete_depth_coordinate(dec_out)
  
  # idx = np.random.permutation( enc_in.shape[0] )
  idx = np.arange(enc_in.shape[0])
  enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]

  if (visualization):
      # Visualize random samples
      import matplotlib.gridspec as gridspec
    
      # 1080p    = 1,920 x 1,080
      fig = plt.figure( figsize=(19.2, 10.8) )
    
      gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
      gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
      plt.axis('off')
    
      subplot_idx, exidx = 1, 0
      nsamples = min(15, n2d)
      for i in np.arange( nsamples ):
    
        # Plot 2d pose
        ax1 = plt.subplot(gs1[subplot_idx-1])
        p2d = enc_in[exidx,:]
        viz.show2Dpose( p2d, ax1, add_labels=False)
        # ax1.invert_yaxis()
        
        # Plot 3d gt
        ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
        p3d = dec_out[exidx,:]
        viz.show3Dpose( p3d, ax2, add_labels=False)
    
        # Plot 3d predictions
        ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
        p3d = poses3d[exidx,:]
        viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=False)
    
        exidx = exidx + 1
        subplot_idx = subplot_idx + 3

  plt.show()
  
  return results_3d, dim_to_use_3d


def calc_error(results_3d, dims_to_use, use_original_metrics=True):
  # original code includes root (hip) joint where the difference is zero...  
  root_index = config.ROOT_INDEX
  hip_coords = [root_index*3, root_index*3+1, root_index*3+2]
  dims_to_use = np.insert(dims_to_use, root_index*3, hip_coords) 
    
  errors_by_action = {}
  results_by_action = {}
  
  # merge results of different subjects (and cameras)
  for key3d, data in results_3d.items():
    (_, cleaned_action, _) = key3d  
    
    results = results_by_action.setdefault(cleaned_action, {})
    
    if (results == {}): # avoid side-effects to results_3d by copying
        results = copy.deepcopy(data)
        
    else :
        results["true"] = np.concatenate([results["true"], data["true"]], axis=0)
        results["pred"] = np.concatenate([results["pred"], data["pred"]], axis=0)
        
    results_by_action[cleaned_action] = results
    
  for action, data in results_by_action.items():    
    diff = data["true"][:, dims_to_use] - data["pred"][:, dims_to_use]
    squared_diff = np.square(diff)
        
    if use_original_metrics:
      # last, incomplete batch will be ignored
      batch_size = FLAGS.batch_size
      num_batches = squared_diff.shape[0] // batch_size
      squared_diff = squared_diff[:num_batches*batch_size, :]
    
    squared_diff_by_coords = squared_diff.reshape(-1, 3)
    squared_diff_by_coords_sum = np.sum(squared_diff_by_coords, axis=1)
    sqrt_err = np.sqrt(squared_diff_by_coords_sum)
    mean_sqrt_err = np.average(sqrt_err)
    
    squared_diff_by_joint = squared_diff.reshape((-1,config.NUM_JOINTS_WITH_ROOT, 3))
    squared_diff_by_joint_sum_sqrt = np.sqrt(np.sum(squared_diff_by_joint, axis=2))
    pck_150mm = np.sum((squared_diff_by_joint_sum_sqrt < 150).astype(int), axis=1) 
    pck_150mm_percent = np.average(pck_150mm / config.NUM_JOINTS_WITH_ROOT * 100)
    
    errors = errors_by_action.setdefault(action, {"rmse": [], "pck": []})
    errors["rmse"].append(mean_sqrt_err)
    errors["pck"].append(pck_150mm_percent)
  
  for action, errors in errors_by_action.items():
    print(action, round(np.average(errors["rmse"]), 2), round(np.average(errors["pck"]), 2))  
  
  rmse_errors = list(map(lambda errors: errors["rmse"], errors_by_action.values()))
  pck_errors = list(map(lambda errors: errors["pck"], errors_by_action.values()))
  print("Average", round(np.average(rmse_errors), 2), round(np.average(pck_errors), 2))



def eval():
  results_3d, dim_to_use_3d = sample(visualization=False)
  calc_error(results_3d, dim_to_use_3d)
  store_results(results_3d, dim_to_use_3d)
  
"""  
Configurations: 

* "h36m_16j", "h36m_16j_debug", "h36m_16j_cam", 
* "h36m_16j_inf_narrat3d_val", "h36m_16j_cam_inf_narrat3d_val"
* "narrat3d_16j", "narrat3d_16j_cam",
* "narrat3d_21j", "narrat3d_21j_cam", "narrat3d_21j_inf_narrat3d_test"
"""
def main(_):    
  load_config("h36m_16j")  
  initialize_flags()
  data_utils.initialize_methods()
  
  if (not config.USE_ORIGINAL_VALIDATION_DATA and not config.USE_OWN_TEST_DATA):
    data_utils_mod.initialize_trainval_data()
  
  # train()
  eval()


if __name__ == "__main__":
  global model_folder
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_folder", default=current_folder)
  parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
  parser.add_argument("--sub_folders", default="")
  args = parser.parse_args()

  model_folder = args.model_folder
  out_folder = args.out_folder
  sub_folders = args.sub_folders
  set_model_folder(model_folder)
  set_test_folders(out_folder, sub_folders)
    
  tf.compat.v1.app.run()
