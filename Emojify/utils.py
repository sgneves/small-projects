
import collections
import inspect
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def get_last_elem(x):
    '''Gets the last element of the input variable.'''

    if isinstance(x, collections.abc.Iterable):
        return x[-1]
    else:
        return x


def set_spines(ax=None):
    '''Sets the box outline around the axis ax.

    Args:
      ax: Matplotlib axes object. Defaults to the current axis.
    '''

    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def compare_model_param(create_fun, create_param, fit_fun, fit_param, tuning_param,
                        xscale='linear', xticks_labels=None, print_info=False):
    ''' Compares the models created and fitted with the specified hyperparameters.'''

    # Create and fit the models
    train_accuracies = []
    val_accuracies = []
    times = []

    for val in tuning_param[1]:

        model = create_fun(**create_param, **{tuning_param[0]:val})

        outputs = fit_fun(model, **fit_param, print_info=print_info)

        train_accuracies.append(get_last_elem(outputs[0]))
        val_accuracies.append(get_last_elem(outputs[1]))
        times.append(outputs[2])

    # Plot the accuracies and training times
    if xticks_labels is None:
        x = tuning_param[1]
    else:
        x = range(len(tuning_param[1]))

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, train_accuracies)
    plt.plot(x, val_accuracies)
    plt.xscale(xscale)
    plt.xlabel(tuning_param[0])
    plt.xticks(x, xticks_labels)
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], frameon=False)
    set_spines()

    plt.subplot(1, 2, 2)
    plt.plot(x, times)
    plt.xscale(xscale)
    plt.xlabel(tuning_param[0])
    plt.xticks(x, xticks_labels)
    plt.ylabel('Training time (s)')
    set_spines()

def fit_keras_model(model, X_train, Y_train, datagen=None, batch_size=32, epochs=1, verbose=0,
                    callbacks=None, validation_split=0, validation_data=None, print_info=True):
    '''
    Fits a Keras model to the training set and calculates its accuracy
    and training time.
    '''

    # Fit the model and calculate the training time
    if datagen is None:
        data = (X_train,Y_train)
    else:
        data = (datagen.flow(X_train, Y_train, batch_size=batch_size),)

    start = timer()
    history = model.fit(*data, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        callbacks=callbacks, validation_split=validation_split,
                        validation_data=validation_data).history
    time = timer() - start

    train_accuracy = history['accuracy']

    if 'val_accuracy' in history:
        val_accuracy = history['val_accuracy']
    else:
        val_accuracy = None

    # Display information of the training process
    if print_info:
        print('Train accuracy: {:.2%}'.format(train_accuracy[-1]), end='')
        if val_accuracy is not None:
            print('; Validation accuracy: {:.2%}'.format(val_accuracy[-1]), end='')
        print('; Training time: {:.1f}s'.format(time))

    return train_accuracy, val_accuracy, time

def compare_keras_param(create_fun, create_param, fit_fun, fit_param, tuning_param, legend=None,
                        print_info=False):
    '''Compares the Keras models created and fitted with the specified hyperparameters.'''

    # Set default legend
    if legend == None:
        legend = [str(val) for val in tuning_param[1]]

    # Create and fit the model
    for val in tuning_param[1]:

        if tuning_param[0] in inspect.getfullargspec(create_fun).args:

            model = create_fun(**create_param, **{tuning_param[0]:val})

            train_accuracy = fit_fun(model, **fit_param, print_info=print_info)[0]
        else:
            model = create_fun(**create_param)

            train_accuracy = fit_fun(model, **fit_param, **{tuning_param[0]:val},
                                     print_info=print_info)[0]

        plt.plot(range(1, len(train_accuracy) + 1), train_accuracy)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(legend, frameon=False)
    plt.title(tuning_param[0])
    set_spines()
