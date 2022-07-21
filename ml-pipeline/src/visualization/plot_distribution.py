import pandas as pd
from matplotlib import pyplot as plt


def plot_distribution_pie_chart(distribution: dict[str, int], display_n: int = 22):
    activity_dict = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    # Convert to minutes & round to 2 decimal places
    steps_per_minute = 60 * 60
    activity_dict = {k: round(v / steps_per_minute, 2) for k, v in activity_dict.items()}

    data = list(activity_dict.values())
    labels = list(activity_dict.keys())

    # Take only the first n important labels
    others = sum(data[display_n:])
    data = data[:display_n]
    labels = labels[:display_n]
    if others > 0:
        data.append(others)
        labels.append('others')

    plt.figure(figsize=(12, 10))
    plt.pie(data, labels=labels, startangle=180, labeldistance=1.05, pctdistance=0.92, autopct='%1.1f%%')
    plt.suptitle("People Distribution", y=0.95)
    plt.title(f"{len(activity_dict)} people – {round(sum(data))} minutes", y=1.04)
    plt.axis('equal')
    plt.show()
    plt.savefig('plots/pie_chart.png')

def plot_distribution_bar_chart(distribution: dict[str, int]):
    activity_dict = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    # Convert to minutes & round to 2 decimal places
    steps_per_minute = 60 * 60
    activity_dict = {k: round(v / steps_per_minute, 2) for k, v in activity_dict.items()}

    plt.figure(figsize=(12, 10))
    pd.Series(activity_dict).plot(kind="bar")
    plt.title("Activity Distribution")
    plt.ylabel("Minutes")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    plt.savefig('plots/bar_chart.png')

