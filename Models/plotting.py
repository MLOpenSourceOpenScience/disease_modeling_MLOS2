import datetime

import torch
import matplotlib.dates as md
import matplotlib.pyplot as plt


def plot_prediction(y_pred, y_truth, batch_size, num_nodes, index):
    s = y_truth.shape
    y_true = y_truth.reshape(s[0], batch_size, num_nodes, s[-1])
    y_true = y_true[:, :, :, 0]

    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], batch_size, num_nodes, s[-1])
    y_pred = y_pred[:, :, :, 0]

    start_date = datetime.date(2013, 5, 13)
    full_len = int(y_true.shape[0] * 10 / 2 * batch_size)
    t = [
        start_date + datetime.timedelta(weeks=t)
        for t in range(full_len - y_true.shape[0] * batch_size, full_len)
    ]

    figure, axis = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.6)

    to_plt = [5, 7, 9, 21]
    for i in range(2):
        for j in range(2):
            idx = to_plt[i * 2 + j]
            axis[i, j].xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
            axis[i, j].xaxis.set_major_locator(md.DayLocator(interval=200))
            axis[i, j].plot(
                t, torch.flatten(y_true[:, :, idx]), label="Actual", color="blue"
            )
            axis[i, j].plot(
                t, torch.flatten(y_pred[:, :, idx]), label="Predicted", color="orange"
            )
            axis[i, j].set_title(index[idx])
            axis[i, j].set_xlabel("Week")
            axis[i, j].set_ylabel("Disease Cases")
            axis[i, j].legend()

    plt.tight_layout()
    plt.show()


def plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index):
    s = y_truth.shape
    y_true = y_truth.reshape(s[0], batch_size, num_nodes, s[-1])
    y_true = y_true[:, :, :, 0]

    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], batch_size, num_nodes, s[-1])
    y_pred = y_pred[:, :, :, 0]

    start_date = datetime.date(2013, 5, 13)
    t = [
        start_date + datetime.timedelta(weeks=t)
        for t in range(y_true.shape[0] * batch_size)
    ]

    figure, axis = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.6)

    to_plt = [5, 7, 9, 21]
    for i in range(2):
        for j in range(2):
            idx = to_plt[i * 2 + j]
            axis[i, j].xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
            axis[i, j].xaxis.set_major_locator(md.DayLocator(interval=700))
            axis[i, j].plot(
                t, torch.flatten(y_true[:, :, idx]), label="Actual", color="blue"
            )
            axis[i, j].plot(
                t[: int(0.7 * len(t))],
                torch.flatten(y_pred[:, :, idx])[: int(0.7 * len(t))],
                label="Train",
                color="orange",
            )
            axis[i, j].plot(
                t[int(0.7 * len(t)) :],
                torch.flatten(y_pred[:, :, idx])[int(0.7 * len(t)) :],
                label="Predicted",
                color="red",
            )
            axis[i, j].set_title(index[idx])
            axis[i, j].set_xlabel("Week")
            axis[i, j].set_ylabel("Disease Cases")
            axis[i, j].legend()

    plt.tight_layout()
    plt.show()


def plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, idx):
    s = y_truth.shape
    y_true = y_truth.reshape(s[0], batch_size, num_nodes, s[-1])
    y_true = y_true[:, :, :, 0]

    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], batch_size, num_nodes, s[-1])
    y_pred = y_pred[:, :, :, 0]

    start_date = datetime.date(2013, 5, 7)
    t = [
        start_date + datetime.timedelta(weeks=t)
        for t in range(y_true.shape[0] * batch_size)
    ]
    # t = [t for t in range(y_true.shape[0] * batch_size)]

    plt.figure(figsize=(10, 6))
    plt.gca().xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(md.DayLocator(interval=400))

    # axis.plot(
    #     t, torch.flatten(y_true[:, :, idx]), label="Actual", color="blue"
    # )
    # axis.plot(
    #     t[:int(0.7 * len(t))], torch.flatten(y_pred[:, :, idx])[:int(0.7 * len(t))], label="Train", color="orange"
    # )
    plt.plot(
        t[: int(0.7 * len(t))],
        torch.flatten(y_true[:, :, idx])[: int(0.7 * len(t))],
        label="Train Data",
        color="blue",
    )
    plt.plot(
        t[int(0.7 * len(t)) :],
        torch.flatten(y_true[:, :, idx])[int(0.7 * len(t)) :],
        label="Actual Test Data",
        color="orange",
    )
    plt.plot(
        t[int(0.7 * len(t)) :],
        torch.flatten(y_pred[:, :, idx])[int(0.7 * len(t)) :],
        label="Predicted Test Data",
        color="red",
    )
    plt.title(index[idx])
    plt.xlabel("Week")
    plt.ylabel("Disease Cases")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


def plot_predictions_layered(y_preds, y_truths):
    raise NotImplementedError()
