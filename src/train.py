"""
It trains model

Usage: python3 train.py [-h]
"""
import argparse
from os import path, environ
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import PROCESSED_DATA_PATH, MODELS_PATH
from utils import load_test_train_data, try_makedirs
from models import get_model, IntervalEvaluation


def init_argparse():
    """Initializes argparse"""
    parser = argparse.ArgumentParser(
        description='Trains toxic comment classifier')
    parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='model architecture (lstm_cnn, gru)',
        default='lstm_cnn',
        type=str)
    parser.add_argument(
        '-t',
        '--train',
        nargs='?',
        help='path to train.csv file',
        default=path.join(PROCESSED_DATA_PATH, 'train.csv'),
        type=str)
    parser.add_argument(
        '-T',
        '--test',
        nargs='?',
        help='path to test.csv file',
        default=path.join(PROCESSED_DATA_PATH, 'test.csv'),
        type=str)
    parser.add_argument(
        '--gpu',
        nargs='?',
        help='GPU device number (1 or 1,2,3)',
        default=1,
        type=str)
    return parser


def plot(history, aucs, model_path=None):
    """It saves into files accuracy and loss plots"""
    import matplotlib
    # generates images without having a window appear
    matplotlib.use('Agg')
    import matplotlib.pylab as plt

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path.join(model_path, 'accuracy.png'))
    # summarize history for loss
    # TODO: some strange curves at the top of graphic
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(aucs)
    plt.title('model loss, ROC AUC')
    plt.ylabel('loss, ROC AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'ROC AUC'], loc='upper left')
    plt.savefig(path.join(model_path, 'loss.png'))


def main():
    """Main function"""
    args = init_argparse().parse_args()

    environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not path.isfile(args.train):
        print('Cannot open {} file'.format(args.train))
        return
    print('Loading train and test data')
    top_words = 10000
    max_comment_length = 1000
    (data, labels), test_data = load_test_train_data(
        args.train, args.test, top_words, max_comment_length)
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.20, random_state=42)
    # loading the model
    model = get_model(
        args.model,
        gpu=args.gpu,
        top_words=top_words,
        max_comment_length=max_comment_length,
        embedding_vector_length=32)
    print('Training model')
    print(model.summary())
    ival = IntervalEvaluation(validation_data=(val_data, val_labels))
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=2,
        batch_size=256,
        callbacks=[ival])
    # history of training
    print(history.history.keys())
    # Saving architecture + weights + optimizer state
    model_path = path.join(MODELS_PATH, '{}_{:.4f}_{:.4f}_{:.4f}'.format(
        args.model, ival.aucs[-1], history.history['val_loss'][-1]
        if 'val_loss' in history.history else history.history['loss'][-1],
        history.history['val_acc'][-1]
        if 'val_acc' in history.history else history.history['acc'][-1]))
    print('Saving model')
    try_makedirs(model_path)
    model.save(path.join(model_path, 'model.h5'))
    plot(history, ival.aucs, model_path)
    # Calculate metrics of the model
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Loss: %.2f%%" % (scores[0] * 100))
    # print("Accuracy: %.2f%%" % (scores[1] * 100))

    print('Generating predictions')
    predictions = model.predict(test_data, batch_size=64)
    pd.DataFrame({
        'id': pd.read_csv(args.test)['id'],
        'toxic': predictions[:, 0],
        'severe_toxic': predictions[:, 1],
        'obscene': predictions[:, 2],
        'threat': predictions[:, 3],
        'insult': predictions[:, 4],
        'identity_hate': predictions[:, 5]
    }).to_csv(
        path.join(model_path, 'predictions.csv'), index=False)
    # Don't round predictions.
    # Rounding makes predictions much worse!


if __name__ == '__main__':
    main()
