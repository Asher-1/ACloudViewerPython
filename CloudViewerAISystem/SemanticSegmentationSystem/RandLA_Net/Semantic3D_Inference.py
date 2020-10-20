import time
import numpy as np
import tensorflow as tf


class ModelTester:
    def __init__(self, logger, restore_snap=None, on_cpu=False):
        self.log_out = logger
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars)

        # Create a session for running Ops on the Graph.
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            self.log_out.info("Model restored from " + restore_snap)
        else:
            assert False, "must specify model path..."

    def inference(self, model, dataset, use_votes=False, num_votes=100):
        t1 = time.time()

        test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                      for l in dataset.input_trees[dataset.cloud_split]]

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (tf.nn.softmax(model.logits),
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.test_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                self.log_out.info('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility[dataset.cloud_split])))

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility[dataset.cloud_split])
                self.log_out.info('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min))

                if last_min + 4 < new_min or not use_votes:

                    preds_list = []

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    self.log_out.info('Reproject Vote #{:d}'.format(int(np.floor(new_min))))

                    i_test = 0
                    for pc_num in dataset.raw_size_list:
                        # Reproject probs
                        probs = np.zeros(shape=[pc_num, 19], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)
                        preds_list.append(preds)
                        i_test += 1

                    t2 = time.time()
                    self.log_out.info('[ModelTester.inference] Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return preds_list

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return []
