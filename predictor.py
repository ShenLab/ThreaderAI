import tensorflow as tf
import numpy as np
import time
import glob
import logging
import argparse

template_feature_dim = 41
target_feature_dim = 28

pair_feature_dim = 9
feature_dim = template_feature_dim + target_feature_dim + pair_feature_dim

from sklearn.metrics import roc_curve, auc, precision_recall_curve
logger = logging.getLogger()


class Predictor(object):
    def __init__(self):
        pass

    def predict(self, model_path, input_file, output_dir):
        #model = ThreaderModel(self.model_config)
        model = tf.keras.models.load_model(model_path)
        model.summary()
        self.model = model
        self._evaluate_one(model, input_file, output_dir)

    def _calc_auc(self, label, score):
        #print(label)
        #print(score)
        label = label.flatten().astype(np.int32)
        score = score.flatten()
        score = score[label > -1]
        new_label = label[label > -1]
        precision, recall, _ = precision_recall_curve(new_label, score)
        pr_auc = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(new_label, score)
        roc_auc = auc(fpr, tpr)
        return roc_auc, pr_auc

    def _save_res(self, path, pred):
        np.save(path, pred)

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, None], dtype=tf.float32, name='labels'),
        tf.TensorSpec(
            shape=[None, feature_dim, None, None],
            dtype=tf.float32,
            name='logits'),
        tf.TensorSpec(
            shape=[None, None, None], dtype=tf.float32, name='pos_weight')
    ])
    def _run_batch(self, label, feature, pos_weight):
        def compute_loss(labels, logits, pos_weight):
            per_example_loss = tf.nn.weighted_cross_entropy_with_logits(
                labels, logits, pos_weight)

            mask = tf.greater(labels, -1)
            per_example_loss = tf.boolean_mask(per_example_loss, mask)
            batch_loss = tf.reduce_mean(per_example_loss)
            return batch_loss

        logit = self.model(feature)
        pred = tf.sigmoid(logit)
        loss = compute_loss(label, logit, pos_weight)
        return pred, loss

    def _evaluate_one(self, model, input_file, output_dir):

        dataset = self._build_dataset(input_file, 2)
        sum_loss = 0.0
        batch_cnt = 0
        sample_cnt = 0
        auc_list = []
        all_score = []
        all_label = []
        for label, feature, t1_name, t2_name, t1_len, t2_len, pos_weight in dataset:
            batch_cnt += 1
            pred, per_example_loss = self._run_batch(label[:, :, :, 2],
                                                     feature, pos_weight)
            sum_loss += per_example_loss

            pred_n = np.array(pred)
            label_n = np.array(label[:, :, :, 0])
            t1_name_n = np.array(t1_name)
            t2_name_n = np.array(t2_name)

            for pred_, label_, t1_len_, t2_len_, t1_name_, t2_name_ in\
                    zip(pred_n, label_n, t1_len, t2_len, t1_name_n, t2_name_n):

                sample_cnt += 1
                mask_pred_ = pred_[:t1_len_, :t2_len_]
                mask_label_ = label_[:t1_len_, :t2_len_]
                roc_auc_, pr_auc_ = self._calc_auc(mask_label_, mask_pred_)
                auc_list.append([roc_auc_, pr_auc_])

                t1_name_ = t1_name_.decode('utf-8')
                t2_name_ = t2_name_.decode('utf-8')

                output_path = '{}/{}-{}.pred'.format(output_dir, t1_name_,
                                                     t2_name_)
                self._save_res(output_path, mask_pred_)
                all_score.extend(list(mask_pred_.flatten()))
                all_label.extend(list(mask_label_.flatten()))

        logger.info('End')
        mean_auc = np.mean(np.array(auc_list), axis=0)
        all_roc_auc, all_pr_auc = self._calc_auc(
            np.array(all_label), np.array(all_score))
        log_loss = sum_loss / batch_cnt
        logger.info(
            f'sample_num= {sample_cnt} loss= {log_loss:.3f} roc_auc= {mean_auc[0]:.3f} pr_auc= {mean_auc[1]:.3f} all_roc_auc= {all_roc_auc:.3f} all_pr_auc= {all_pr_auc:.3f}'
        )

    def _build_dataset(self, input_tfrecord_files, batch_size):
        feature_description = {
            't1_fea_1d': tf.io.FixedLenFeature([], tf.string),
            't2_fea_1d': tf.io.FixedLenFeature([], tf.string),
            'fea_2d': tf.io.FixedLenFeature([], tf.string),
            'label2': tf.io.FixedLenFeature([], tf.string),
            't1_len': tf.io.FixedLenFeature([], tf.int64),
            't2_len': tf.io.FixedLenFeature([], tf.int64),
            't1_name': tf.io.FixedLenFeature([], tf.string),
            't2_name': tf.io.FixedLenFeature([], tf.string),
            #'fea_1d_dim': tf.io.FixedLenFeature([], tf.io.int64)
        }

        def _parser(example_proto):
            parsed = tf.io.parse_single_example(example_proto,
                                                feature_description)
            t1_fea_1d = tf.io.decode_raw(parsed['t1_fea_1d'], tf.float32)
            t2_fea_1d = tf.io.decode_raw(parsed['t2_fea_1d'], tf.float32)
            fea_2d = tf.io.decode_raw(parsed['fea_2d'], tf.float32)
            label = tf.io.decode_raw(parsed['label2'], tf.float32)

            t1_len = parsed['t1_len']
            t2_len = parsed['t2_len']
            t1_name = parsed['t1_name']
            t2_name = parsed['t2_name']
            t1_fea_1d = tf.reshape(t1_fea_1d,
                                   tf.stack([t1_len, template_feature_dim]))
            t2_fea_1d = tf.reshape(t2_fea_1d,
                                   tf.stack([t2_len, target_feature_dim]))
            fea_2d = tf.reshape(fea_2d, [t1_len, t2_len, pair_feature_dim])

            label = tf.reshape(label, [t1_len, t2_len, 3])

            #bugs here;filter unsolved region and padding regions
            #label0 = label[:, :, 0]
            #label2 = label[:, :, 2]
            #label = tf.where(label0 < 0.0, -1.0, label2)

            v1 = tf.expand_dims(t1_fea_1d, axis=1)
            v2 = tf.expand_dims(t2_fea_1d, axis=0)
            v1 = tf.tile(v1, [1, t2_len, 1])
            v2 = tf.tile(v2, [t1_len, 1, 1])
            feature = tf.concat([v1, v2, fea_2d], axis=-1)
            #channel first
            feature = tf.transpose(feature, perm=[2, 0, 1])
            pos_weight = tf.fill(
                [t1_len, t2_len],
                0.5 * tf.dtypes.cast(t1_len + t2_len, tf.float32))
            print('feature', feature.shape, t1_len, t2_len, pos_weight.shape)
            return label, feature, t1_name, t2_name, t1_len, t2_len, pos_weight

        dataset = tf.data.TFRecordDataset(input_tfrecord_files)
        dataset = dataset.map(_parser, num_parallel_calls=8)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, None, 3], [feature_dim, None, None], [], [],
                           [], [], [None, None]),
            padding_values=(-1.0, 0.0, "NULL", "NULL",
                            tf.dtypes.cast(0, tf.int64),
                            tf.dtypes.cast(0, tf.int64), 1.0),
            drop_remainder=True)
        dataset = dataset.prefetch(32)
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    predictor_ = Predictor()
    predictor_.predict(args.model_path, args.input, args.output)
