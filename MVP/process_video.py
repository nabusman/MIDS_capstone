import argparse
import os

import h5py
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from PIL import Image


def main(model_path, video_path, sample_rate, sample_unit, output_path, 
    classes_path, anchors_path, score_threshold, iou_threshold):
    # Get session
    sess = K.get_session()
    if os.path.isdir(video_path):
        for video_file in os.listdir(video_path):
            predict_video(model_path, os.path.join(video_path, video_file),
                sample_rate, sample_unit, output_path, classes_path,
                anchors_path, score_threshold, iou_threshold, sess)
    else:
        predict_video(model_path, video_path, sample_rate, sample_unit,
            output_path, classes_path, anchors_path, score_threshold,
            iou_threshold, sess)
    sess.close()



def predict_video(model_path, video_path, sample_rate, sample_unit, output_path, 
    classes_path, anchors_path, score_threshold, iou_threshold, sess):
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

    # Make output dir if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the model
    yolo_model = load_model(model_path)

    # Convert sample rate into milliseconds
    if sample_unit == 's':
        sample_rate = sample_rate * 1000
    elif sample_unit == 'm':
        sample_rate = sample_rate * 60000
    elif sample_unit == 'h':
        sample_rate = sample_rate * 3.6e6
    else:
        raise Exception("sample_unit not recognized")

    # Load anchors
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # Load classes
    with open(classes_path) as f:
           class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))


    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold = score_threshold,
        iou_threshold = iou_threshold)
    # Load video
    vidcap = cv2.VideoCapture(video_path)
    current_msec = 0
    all_preds = []

    print('Predicting video file: ' + os.path.basename(video_path))
    while True:
        # Move to the correct point in the video
        vidcap.set(cv2.CAP_PROP_POS_MSEC, current_msec)
        # Extract frame
        success, img = vidcap.read()
        if success:
            # Predict frame
            prediction = predict(img, yolo_model, class_names, anchors, 
                model_image_size, boxes, scores, classes, sess,
                input_image_shape)
            print('Found ' + str(len(prediction)) + ' people at ' + 
                str(current_msec))
            prediction = map(lambda x: x + [current_msec], prediction)
            all_preds.extend(prediction)
            # Move to the next time step
            current_msec += sample_rate
        else:
            # Since the frame is not captured, assume that video has finished
            break
    # Write output to output_path
    all_preds = np.array(all_preds)
    output_filename = os.path.join(output_path, os.path.basename(video_path) +
        '_prediction.csv')
    np.savetxt(output_filename, all_preds, delimiter = ',', header = 
        'score,left,top,right,bottom,milliseconds', comments = '',
        fmt = '%10.2f')
    
            

def predict(img, yolo_model, class_names, anchors, model_image_size, 
    boxes, scores, classes, sess, input_image_shape):
    """
    Returns a list of lists in the format:
    [[score, left, top, right, bottom],...]
    """
    output = []
    image = Image.fromarray(img)
    resized_image = image.resize(
        tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        if not predicted_class == 'person':
            continue
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        output.append([score, left, top, right, bottom])
    return output


def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.
    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Process video and extract \
        the number of people per time unit')
    # model file path
    parser.add_argument('-m', '--model_path', required = True, help = 'Path \
        to the Keras h5 model path')
    # video file path
    parser.add_argument('-v', '--video_path', required = True, help = 'Path \
        to the video file to process or path to directory of videos')
    # COCO class file path
    parser.add_argument('-c', '--classes_path', required = False, help = 'Path \
        to classes file, defaults to coco_classes.txt', 
        default='model_data/coco_classes.txt')
    # YOLO Anchors file path
    parser.add_argument('-a', '--anchors_path', required = False, help = 'Path \
        to anchors file, defaults to yolo_anchors.txt',
        default='model_data/yolo_anchors.txt')
    # sampling rate
    parser.add_argument('-s', '--sample_rate', required = False, default = 1,
        type = int, help = 'The rate to sample time unit, must be an integer, \
        for instance if you want to sample 1 frame per second, put a 1 here')
    # time unit of sampling rate (s, m, h)
    parser.add_argument('-u', '--sample_unit', required = False, default = 's',
        choices = ['s', 'm', 'h'], help = 'The time unit to sample, default is \
        "m" for minutes, options are: s (seconds), m (minutes), h (hours); \
        for instance if you want to sample 1 frame per second, put a "s" here')
    # output directory path
    parser.add_argument('-d', '--output_path', required = False, help = 'Path \
        to the directory for output', default = 'predictions/')
    # score threshold
    parser.add_argument('--score_threshold', required = False, type = float,
        help = 'Threshold for bounding box scores, default .3', default = '0.3')
    # IOU threshold
    parser.add_argument('--iou_threshold', required = False, type = float,
        help = 'Threshold for non max suppression IOU, default .5', 
        default = '0.5')
    args = parser.parse_args()
    main(**vars(args))