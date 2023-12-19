from datetime import datetime
from enum import Enum
import os
import random
import time
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import backend as K
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LeakyReLU, UpSampling2D, BatchNormalization, Conv2D, Concatenate, Lambda, ReLU, Multiply, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback

from functools import wraps, reduce
import numpy as np
from PIL import Image

from libs.config_loader import get_anchors, get_classes, get_colors, get_dataset
from libs.drawer import draw_boxes
from libs.postprocess import yolo3_postprocess_np
from libs.preprocess import letterbox_resize, normalize_image, preprocess_image
from libs.loss import yolo3_loss

class DistYOLO:
    def __init__(self) -> None:
        # configs
        self.anchors = get_anchors('yolo3_anchors.txt')
        self.class_names = get_classes('kitty_all_except_nodata.txt')
        self.colors = get_colors(self.class_names)
        self.num_anchors = len(self.anchors) // 3
        self.num_feature_layers = 3
        self.num_classes = len(self.class_names)

        self.model_image_size = (608, 608)

    def get_base_model(self):
        out_filters = self.num_anchors * (self.num_classes + 5+1)
        input_tensor = Input(shape=(608, 608, 3), name='image_input')

        # Backbone
        backbone = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)
        backbone_len = 132

        f1 = backbone.get_layer('block14_sepconv2_act').output
        f2 = backbone.get_layer('block13_sepconv2_bn').output
        f3 = backbone.get_layer('block4_sepconv2_bn').output

        f1_channel_num = 1024
        f2_channel_num = 512
        f3_channel_num = 256

        x, y1 = make_last_layers(f1, f1_channel_num//2, out_filters, predict_id='1')        
        #upsample fpn merge for feature map 1 & 2
        x = compose(
                DarknetConv2D_BN_Leaky(f2_channel_num//2, (1,1)),
                UpSampling2D(2))(x)
        x = Concatenate()([x,f2])

        #feature map 2 head & output (26x26 for 416 input)
        x, y2 = make_last_layers(x, f2_channel_num//2, out_filters, predict_id='2')

        #upsample fpn merge for feature map 2 & 3
        x = compose(
                DarknetConv2D_BN_Leaky(f3_channel_num//2, (1,1)),
                UpSampling2D(2))(x)
        x = Concatenate()([x, f3])

        #feature map 3 head & output (52x52 for 416 input)
        x, y3 = make_last_layers(x, f3_channel_num//2, out_filters, predict_id='3')

        return Model(inputs = input_tensor, outputs=[y1,y2,y3]), backbone_len


    def fit(self, total_epoch, batch_size, dataset_working_directory, annotation_file, val_annotation_file = None):
        annotation_file = os.path.join(dataset_working_directory, annotation_file)
        log_dir = os.path.join('./logs', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        checkpoint_period = 1
        
        class_names = self.class_names
        num_classes = len(class_names)
        anchors = self.anchors
        num_anchors = len(anchors)
        label_smoothing = 0
        elim_grid_sense = False
        optimizer = Adam(lr=1e-3, decay=0)

        self.model, backbone_len = self.get_base_model()
        for i in range(backbone_len):
             self.model.layers[i].trainable = False       # Freeze backbone
        
        y_true = [Input(shape=(None, None, 3, self.num_classes+5+1), name=f'y_true_{l}') for l in range(self.num_feature_layers)]
        
        model_loss, location_loss, confidence_loss, class_loss, dist_loss = Lambda(yolo3_loss, name='yolo_loss',
            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing, 'elim_grid_sense': elim_grid_sense, 'use_diou_loss' : False})(
        [*self.model.output, *y_true])

        self.model = Model([self.model.input, *y_true], model_loss)

        loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss, 'dist_loss':dist_loss}
        add_metrics(self.model, loss_dict)
        self.model.compile(optimizer=optimizer, loss={ 'yolo_loss': lambda y_true, y_pred: y_pred })

        # callbacks untuk proses training
        logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
        checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_weights_only=False,
            save_best_only=False,
            period=checkpoint_period)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
        terminate_on_nan = TerminateOnNaN()

        callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

        # get train&val dataset
        dataset = get_dataset(annotation_file, dataset_working_directory, shuffle=False)
        if val_annotation_file:
            val_dataset = get_dataset(val_annotation_file, dataset_working_directory)
            num_train = len(dataset)
            num_val = len(val_dataset)
            dataset.extend(val_dataset)
        else:
            val_split = 0.2
            num_val = int(len(dataset)*val_split)
            num_train = len(dataset) - num_val

        # assign multiscale interval
        rescale_interval = -1  # Doesn't rescale

        # input shape
        input_shape = self.model_image_size

        # get different model type & train&val data generator
        model = self.model
        data_generator = yolo3_data_generator_wrapper

        optimizer = 'adam'
        learning_rate = 1e-3
        optimizer = get_optimizer(optimizer, learning_rate, decay_type=None)
        optimizer.clipnorm = True
        enhance_augment = None
        multi_anchor_assign = False
        
        print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, batch_size, input_shape))
        model.fit_generator(data_generator(dataset[:num_train], batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign=multi_anchor_assign),
            steps_per_epoch=max(1, num_train//batch_size),
            #validation_data=val_data_generator,
            validation_data=data_generator(dataset[num_train:], batch_size, input_shape, anchors, num_classes, multi_anchor_assign=multi_anchor_assign),
            validation_steps=max(1, num_val//batch_size),
            epochs=total_epoch,
            initial_epoch=0,
            #verbose=1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)
        
        model.save(os.path.join(log_dir, 'trained_final.h5'))
        model.save('proto_trained_final.h5')
        self.save_model_from_training('trained_final.h5', 'proto_trained_final.h5')

    def save_model_from_training(self, model_path, weights_path):
        self.model, _ = self.get_base_model()
        self.model.load_weights(weights_path)
        self.model.save(model_path)

    def load_model(self, model_path):
        custom_object_dict = get_custom_objects()
        self.model = load_model(model_path, compile=False, custom_objects=custom_object_dict)

    def detect(self, image):
        # preprocess image
        image_data = preprocess_image(image, self.model_image_size)
        
        # bentuk image, dalam format (height, width)
        image_shape = tuple(reversed(image.size))   # reversed karena PIL.Image.size berformat (width, height)
        
        # melakukan prediksi dan postprocess
        prediction = self.model.predict([image_data])
        pred_boxes, pred_classes, pred_scores, pred_distances = yolo3_postprocess_np(prediction, image_shape, self.anchors,
                                                                                     self.num_classes, self.model_image_size)
        return pred_boxes, pred_classes, pred_scores, pred_distances

    def detect_image(self, image):
        # deteksi
        pred_boxes, pred_classes, pred_scores, pred_distances = self.detect(image)

        # gambar bounding box dan label
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, pred_boxes, pred_classes, pred_scores, pred_distances, self.class_names, self.colors)
        return Image.fromarray(image_array)

def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }

    return custom_objects_dict

def yolo3_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None,
                                 rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment,
                                rescale_interval, multi_anchor_assign)

def yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment,
                         rescale_interval, multi_anchor_assign):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # prepare multiscale config
    # rescale_step = 0
    # input_shape_list = get_multiscale_list()
    while True:
        if rescale_interval > 0:
            # Do multi-scale training on different input shape
            rescale_step = (rescale_step + 1) % rescale_interval
            if rescale_step == 0:
                input_shape = input_shape_list[random.randint(0, len(input_shape_list) - 1)]

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment_mode=AugmentMode.DO_NOT_AUGMENT)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        if enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, multi_anchor_assign)
        yield [image_data, *y_true], np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                         anchors for a single ground truth

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    # Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    # and image relative coordinate.
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + 1),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Sort anchors according to IoU score
        # to find out best assignment
        best_anchors = np.argsort(iou, axis=-1)[..., ::-1]

        if not multi_anchor_assign:
            best_anchors = best_anchors[..., 0]
            # keep index dim for the loop in following
            best_anchors = np.expand_dims(best_anchors, -1)

        for t, row in enumerate(best_anchors):
            for l in range(num_layers):
                for n in row:
                    # use different matching policy for single & multi anchor assign
                    if multi_anchor_assign:
                        matching_rule = (iou[t, n] > iou_thresh and n in anchor_mask[l])
                    else:
                        matching_rule = (n in anchor_mask[l])

                    if matching_rule:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1
                        y_true[l][b, j, i, k, 5 + num_classes] = true_boxes[
                            b, t, 5]  # here cannot be 5+num_classes+1, the version without +1 is correct

    return y_true

class AugmentMode(Enum):
    DO_NOT_AUGMENT = 0,
    NO_PADDING_SCALING = 1,
    ALL = 2
def get_ground_truth_data(annotation_line, input_shape, augment_mode, max_boxes=100):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    try:
        image.load()
    except Exception as e:
        raise Exception(f"Failed to load image file name '{line[0]}' from annotation line '{annotation_line}'", e)
    image_size = image.size
    model_input_size = tuple(reversed(input_shape))
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if augment_mode == AugmentMode.DO_NOT_AUGMENT:
        new_image, padding_size, offset = letterbox_resize(image, target_size=model_input_size,
                                                           return_padding_info=True)
        image_data = np.array(new_image)
        image_data = normalize_image(image_data)

        # reshape boxes
        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size,
                              offset=offset)
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]

        # fill in box data
        box_data = np.zeros((max_boxes, 6))
        if len(boxes) > 0:
            box_data[:len(boxes)] = boxes

        return image_data, box_data

def reshape_boxes(boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False, vertical_flip=False):
    """
    Reshape bounding boxes from src_shape image to target_shape image,
    usually for training data preprocess

    # Arguments
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).
        src_shape: origin image shape,
            tuple of format (width, height).
        target_shape: target image shape,
            tuple of format (width, height).
        padding_shape: padding image shape,
            tuple of format (width, height).
        offset: top-left offset when padding target image.
            tuple of format (dx, dy).
        horizontal_flip: whether to do horizontal flip.
            boolean flag.
        vertical_flip: whether to do vertical flip.
            boolean flag.

    # Returns
        boxes: reshaped bounding box numpy array
    """
    if len(boxes)>0:
        src_w, src_h = src_shape
        target_w, target_h = target_shape
        padding_w, padding_h = padding_shape
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*padding_w/src_w + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*padding_h/src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0,2]] = target_w - boxes[:, [2,0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1,3]] = target_h - boxes[:, [3,1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

    return boxes

def add_metrics(model, metric_dict):
    for (name, metric) in metric_dict.items():
        model.add_metric(metric, name=name, aggregation='mean')

def get_optimizer(optim_type, learning_rate, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler, amsgrad=False)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler, rho=0.9, momentum=0.0, centered=False)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler, momentum=0.0, nesterov=False)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer

def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = learning_rate
    # elif decay_type == 'cosine':
    #     lr_scheduler = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
    # elif decay_type == 'exponential':
    #     lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    # elif decay_type == 'polynomial':
    #     lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, end_learning_rate=learning_rate/100)
    # elif decay_type == 'piecewise_constant':
    #     #apply a piecewise constant lr scheduler, including warmup stage
    #     boundaries = [500, int(decay_steps*0.9), decay_steps]
    #     values = [0.001, learning_rate, learning_rate/10., learning_rate/100.]
    #     lr_scheduler = PiecewiseConstantDecay(boundaries=boundaries, values=values)
    # else:
    #     raise ValueError('Unsupported lr decay type')

    return lr_scheduler

def make_last_layers(x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(
            DarknetConv2D_BN_Leaky(predict_filters, (3,3)),
            DarknetConv2D(out_filters, (1,1), name='predict_conv_' + predict_id))(x)
    return x, y

def compose(*funcs):
     return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))

@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    L2_FACTOR = 1e-5
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return Conv2D(*args, **yolo_conv_kwargs)

@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)

def CustomBatchNormalization(*args, **kwargs):
    return BatchNormalization(*args, **kwargs)

def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * K.sigmoid(x)

def hard_sigmoid(x):
    return ReLU(6.)(x + 3.) * (1. / 6.)

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

def mish(x):
    return x * K.tanh(K.softplus(x))
