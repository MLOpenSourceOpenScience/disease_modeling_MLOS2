import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    data = pd.read_csv("../Data/Datasets/MLSO2_Final.csv")
    state_data = data[data["region"] == "Kalutara"].reset_index(drop=True)

    # Separate target variable and exogenous factors
    y = state_data["cases"]
    X = state_data.drop(
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

    # 80-20 train-test split
    split_point = int(len(y) * 0.7)
    train_y, test_y = y[:split_point], y[split_point:]
    train_X, test_X = X[:split_point], X[split_point:]

    # Fit the SARIMAX model
    model = SARIMAX(train_y, exog=train_X, order=(1, 2, 3), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=len(test_y), exog=test_X)
    forecast = forecast.to_numpy()
    test_y = test_y.to_numpy()
    print(
        mean_absolute_error(test_y, forecast), root_mean_squared_error(test_y, forecast)
    )

    # Plot the actual vs forecasted values
    start_date = datetime.date(2013, 5, 13)
    t = [start_date + datetime.timedelta(weeks=t) for t in range(len(y))]
    plt.figure(figsize=(10, 6))
    plt.gca().xaxis.set_major_formatter(md.DateFormatter("%m/%d/%Y"))
    plt.gca().xaxis.set_major_locator(md.DayLocator(interval=400))
    plt.plot(t[: int(0.7 * len(t))], train_y, label="Training Data", color="blue")
    plt.plot(t[int(0.7 * len(t)) :], test_y, label="Actual Test Data", color="orange")
    plt.plot(
        t[int(0.7 * len(t)) :],
        forecast,
        label="Forecasted Data",
        color="red",
        linestyle="--",
    )
    plt.legend()
    plt.title("Dengue Cases Forecast vs Actual in Kalutara with ARIMAX")
    plt.xlabel("Week")
    plt.ylabel("Dengue Cases")
    plt.gcf().autofmt_xdate()
    plt.show()
