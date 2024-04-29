import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from keras import Sequential
from keras.src.layers import Dense, LSTM
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
    state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)
    yc = state_data["cases"]
    Xc = state_data.drop(
        [
            "Week",
            "region",
            "cases",
            "minTime",
            "minNdvi",
            "maxNdvi",
            "meanNdvi",
            "minPrecipitationcal",
            "maxPrecipitationcal",
            "meanPrecipitationcal",
            "meanCanopint_Inst",
            "meanPsurf_F_Inst",
        ],
        axis=1,
    )

    ma = []
    rms = []

    for s in [0.6, 0.7, 0.8, 0.9, 1.0]:
        y = yc[:int(len(yc) * s)]
        X = Xc[:int(len(yc) * s)]

        # Normalize features
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        # Convert to sequences
        def create_dataset(X, y, time_steps=3):
            Xs, ys = [], []
            for i in range(time_steps, len(X) - 3):
                v = X[i - time_steps : i]
                Xs.append(v)
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        time_steps = 3
        X_seq, y_seq = create_dataset(X_scaled, y_scaled, time_steps)

        # Split into train and test sets
        split_point = int(len(X_seq) * 0.7)
        test_point = int(len(X_seq) * 0.7)
        train_X, test_X = X_seq[:split_point], X_seq[test_point:]
        train_y, test_y = y_seq[:split_point], y_seq[test_point:]

        # Build the LSTM model
        model = Sequential()
        model.add(
            LSTM(50, activation="relu", input_shape=(train_X.shape[1], train_X.shape[2]))
        )
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Fit the model
        model.fit(train_X, train_y, epochs=40, batch_size=4, verbose=0, shuffle=False)

        # Predict
        predicted = model.predict(test_X)

        # Invert scaling
        predicted = scaler_y.inverse_transform(predicted)
        actual = scaler_y.inverse_transform(test_y)
        train_actual = scaler_y.inverse_transform(train_y.reshape(-1, 1))

        # Calculate MAE, RMSE
        mae = mean_absolute_error(actual, predicted)
        rmse = root_mean_squared_error(actual, predicted)
        ma += [mae]
        rms += [rmse]

    print(f"Average MAE: {np.mean(ma)}, Standard Deviation: {np.std(ma)}")
    print(f"Average RMSE: {np.mean(rms)}, Standard Deviation: {np.std(rms)}:")

    # Print
    start_date = datetime.date(2013, 5, 13)
    t = [start_date + datetime.timedelta(weeks=t) for t in range(len(y_seq))]
    plt.figure(figsize=(10, 6))
    plt.gca().xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(md.DayLocator(interval=400))
    plt.plot(t[:split_point], train_actual, label="Training Data", color="blue")
    plt.plot(t[split_point:], actual, label="Actual Test Data", color="orange")
    plt.plot(
        t[split_point:], predicted, label="Forecasted Data", color="red", linestyle="--"
    )
    plt.legend()
    plt.title("Dengue Cases Forecast vs Actual in Kalutara with LSTM")
    plt.xlabel("Weeks")
    plt.ylabel("Dengue Cases")
    plt.show()
