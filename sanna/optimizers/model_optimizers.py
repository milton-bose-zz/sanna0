from __future__ import print_function, division

import timeit
import logging
import numpy as np

logger = logging.getLogger(__name__)


def optimize_params(model, improvement_threshold=0.995,
        min_iter=2000, min_iter_increase=1.5, n_epochs=200,
        early_stopping=True):

    start_time = timeit.default_timer()

    n_train_batches = model.train_X.get_value(
            borrow=True
            ).shape[0] // model.batch_size
    if early_stopping:
        n_valid_batches = model.valid_X.get_value(
            borrow=True
            ).shape[0] // model.batch_size

    best_loss = np.inf
    training_losses, validation_losses = [], []

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        training_loss = []
        for index in range(n_train_batches):
            training_output = model.train(index)
            iter_ = (epoch - 1) * n_train_batches + index

            if min_iter <= iter_:
                done_looping = True

            training_loss.append(training_output[1])

        training_loss = np.mean(training_loss)
        if early_stopping:
            validation_loss = np.mean([model.validate(i)
                                   for i in range(n_valid_batches)])
            if validation_loss < best_loss * improvement_threshold:
                min_iter = int(max(min_iter, iter_ * min_iter_increase))
                best_loss = validation_loss
                best_epoch = epoch
                best_params = model.get_params()
                logger.info(
                        ("(epoch: %i, iter: %i): " % (epoch, iter_ + 1) +
                        "minimum iteration %d >> validation_loss: %.5f" % (
                            min_iter, best_loss
                            )
                        )
                        )
                done_looping = False
            else:
                logger.debug(
                        ('[epoch: %i, iter: %i] training_loss: %.5f ' % (
                            epoch, iter_ + 1, training_loss
                            ) +
                            'validation loss: %.5f (best: %.5f)' % (validation_loss, best_loss)
                            )
                        )

            validation_losses.append(validation_loss)
        else:
            if training_loss < best_loss * improvement_threshold:
                min_iter = int(max(min_iter, iter_ * min_iter_increase))
                best_loss = training_loss
                best_epoch = epoch
                best_params = model.get_params()
                logger.info(
                        ("(epoch: %i, iter: %i): " % (epoch, iter_ + 1) +
                        "minimum iteration %d >> training_loss: %.5f" % (
                            min_iter, best_loss
                            )
                        )
                        )
                done_looping = False
            else:
                logger.debug(
                        ('[epoch: %i, iter: %i] training_loss: %.5f ' % (
                            epoch, iter_ + 1, training_loss
                            )
                        )
                        )

            training_losses.append(training_loss)

    if early_stopping:
        last_loss = validation_losses[-1]
    else:
        last_loss = training_losses[-1]

    if last_loss < best_loss:
        best_loss = last_loss
        best_epoch = epoch
    else:
        model.set_params(best_params)

    end_time = timeit.default_timer()

    logger.info(
            (
                'finished in %.4f min. best validation score: %.5f' % (
                    (end_time - start_time) / 60., best_loss
                    ) +
                'on epoch %i. (# epoch: %i, # iter: %i)' % (
                    best_epoch, epoch, iter_ + 1
                    )
                )
            )

    return training_losses, validation_losses, best_loss
