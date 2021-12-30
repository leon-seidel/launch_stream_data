import pandas as pd
from matplotlib import pyplot as plt

file_name = "mission_data/Transporter2Mission.csv"
df = pd.read_csv(file_name)

plot_type = "plot"
data_type = "velo"

if plot_type == "plot":
    if data_type == "velo":
        plt.plot(df.t, df.v1)
        plt.plot(df.t, df.v2)
    elif data_type == "alti":
        plt.plot(df.t, df.h1)
        plt.plot(df.t, df.h2)
    elif data_type == "acc":
        plt.plot(df.t, df.a1)
        plt.plot(df.t, df.a2)

elif plot_type == "scatter":
    if data_type == "velo":
        plt.scatter(df.t, df.v1)
        plt.scatter(df.t, df.v2)
    elif data_type == "alti":
        plt.scatter(df.t, df.h1)
        plt.scatter(df.t, df.h2)
    elif data_type == "acc":
        plt.scatter(df.t, df.a1)
        plt.scatter(df.t, df.a2)

if data_type == "velo":
    plt.title("Time vs. velocity")
    plt.xlabel("Time in s")
    plt.ylabel("Velocity in kph")
elif data_type == "alti":
    plt.title("Time vs. altitude")
    plt.xlabel("Time in s")
    plt.ylabel("Altitude in km")
elif data_type == "acc":
    plt.title("Time vs. acceleration")
    plt.xlabel("Time in s")
    plt.ylabel("Acceleration in gs")

plt.legend(["Stage 1", "Stage 2"])
plt.grid()
plt.show()


