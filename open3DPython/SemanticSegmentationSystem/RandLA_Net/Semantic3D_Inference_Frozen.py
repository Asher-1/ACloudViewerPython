import time
from os import makedirs
from os.path import exists, join

import numpy as np
import tensorflow as tf

import helper_tf_util as tools
from helper_ply import read_ply


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, config, restore_snap, on_cpu=False):
        self._load_model(restore_snap, on_cpu)
        self.config = config

    def _load_model(self, model_file, on_cpu=False):
        """
        load detector model for local detection
        :return: NONE
        """
        # Load a (frozen) Tensorflow model into memory.
        print("we are testing ====>>>>", model_file)

        # Create a session for running Ops on the Graph.
        self.graph = tools.load_graph(model_file)

        # get pb tensors according to model graph
        self._read_pb_tensors()

        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            c_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
            c_proto.gpu_options.allow_growth = True

        # create detector session for tensorflow
        self.sess = tf.Session(config=c_proto, graph=self.graph)

    def _read_pb_tensors(self):
        # self.input_img = self.graph.get_tensor_by_name("input_img:0")
        self.prob_logits = self.graph.get_tensor_by_name("results/Softmax:0")

    def test(self, model, dataset, num_votes=100):

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,)

                file_path = ""
                points = self.load_evaluation_points(file_path)
                points = points.astype(np.float16)

                stacked_probs = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.num_points, model.config.num_classes])

                # Insert false columns for ignored labels
                probs2 = stacked_probs
                for l_ind, label_value in enumerate(dataset.label_values):
                    if label_value in dataset.ignored_labels:
                        probs2 = np.insert(probs2, l_ind, 0, axis=1)

                # Get the predicted labels
                preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                # Save ascii preds
                ascii_name = join(test_path, 'predictions', "result.labels")
                np.savetxt(ascii_name, preds, fmt='%d')

            except tf.errors.OutOfRangeError as e:
                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T