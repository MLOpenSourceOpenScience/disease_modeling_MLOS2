import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
    state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)

    # Separate the target variable and the features
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

    maes = []
    rms = []

    for s in [0.6, 0.7, 0.8, 0.9, 1.0]:
        y = yc[:int(len(yc) * s)]
        X = Xc[:int(len(yc) * s)]

        # 70-30 train-test split
        split_point = int(len(y) * 0.7)
        test_point = int(len(y) * 0.7)

        train_y, test_y = y[:split_point], y[test_point:]
        train_X, test_X = X[:split_point], X[test_point:]

        # Instantiate and train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(train_X, train_y)

        # Make predictions
        predictions = rf_model.predict(test_X)

        # Evaluate the model
        mae = mean_absolute_error(test_y, predictions)
        rmse = root_mean_squared_error(test_y, predictions)
        maes += [mae]
        rms += [rmse]

    print(f"Average MAE: {np.mean(maes)}, Standard Deviation: {np.std(maes)}")
    print(f"Average RMSE: {np.mean(rms)}, Standard Deviation: {np.std(rms)}:")

    # Plotting the actual vs forecasted values
    start_date = datetime.date(2013, 5, 13)
    t = [start_date + datetime.timedelta(weeks=t) for t in range(len(yc))]
    plt.figure(figsize=(10, 6))
    plt.gca().xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(md.DayLocator(interval=400))
    plt.plot(t[: int(0.7 * len(t))], train_y, label="Training Data", color="blue")
    plt.plot(t[int(0.7 * len(t)) :], test_y, label="Actual Test Data", color="orange")
    plt.plot(
        t[int(0.7 * len(t)) :],
        predictions,
        label="Forecasted Data",
        color="red",
        linestyle="--",
    )
    plt.legend()
    plt.title("Dengue Cases Forecast vs Actual in Kalutara with Random Forest")
    plt.xlabel("Week")
    plt.ylabel("Dengue Cases")
    plt.show()
