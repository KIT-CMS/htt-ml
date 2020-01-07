import tensorflow as tf
from keras.models import load_model
from keras import backend as K

from tensorflow_derivative.outputs import Outputs

import logging
logger = logging.getLogger("keras_to_tensorflow")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create function to convert saved keras model to tensorflow graph
def convert_to_pb(weight_file,input_fld='',output_fld=''):

    import os.path as osp
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io

    # change filename to a .pb tensorflow file
    output_graph_name = weight_file[:-2]+'pb'
    weight_file_path = osp.join(input_fld, weight_file)

    logger.info("Load keras model {}.".format(weight_file_path))
    model = load_model(weight_file_path, compile=False)

    pred_node_names = [node.op.name for node in model.outputs]

    input_node_name = [node.op.name for node in model.inputs][0]

    # if osp.exists(osp.join(output_fld, output_graph_name)):
    #     return model, osp.join(output_fld, output_graph_name), input_node_name
    # else:
    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               pred_node_names)
    graph_io.write_graph(constant_graph,
                         output_fld,
                         output_graph_name,
                         as_text=False)
    logger.info('Saved the constant graph (ready for inference) at: {} '
                .format(osp.join(output_fld, output_graph_name)))

    return model, osp.join(output_fld, output_graph_name), input_node_name

def load_graph(frozen_graph_filename, input_node_name):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )

    dropout_name = None
    for node in graph.as_graph_def().node:
        #print(node.name)
        if 'keras_learning_phase' in node.name:
            dropout_name = node.name + ':0'

    #print(dropout_name)

    input_name = input_node_name + ':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name, dropout_name

def get_tensorflow_model(args, config_train, config_test):

    keras_model, model_path, input_node_name = convert_to_pb(weight_file = config_test["model"][args.fold],
                                                input_fld=config_train["output_path"],
                                                output_fld=config_train["output_path"])

    tensorflow_model, tf_input, tf_output, dropout_name = load_graph(model_path, input_node_name)

    outputs = Outputs(tensorflow_model.get_tensor_by_name(tf_output), config_train["classes"])


    return keras_model, tensorflow_model, outputs, tf_input, tf_output, dropout_name