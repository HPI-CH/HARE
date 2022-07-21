import pandas as pd

from sonar_types import Recording


def count_activity_length(recordings: list[Recording]):
    values = recordings[0].activities.value_counts()
    for rec in recordings[1:]:
        values = values.add(rec.activities.value_counts(), fill_value=0)

    return values


def count_person_length(recordings: list[Recording]):
    values = pd.Series({recordings[0].subject: recordings[0].activities.count()})
    for rec in recordings[1:]:
        values = values.add(pd.Series({rec.subject: rec.activities.count()}), fill_value=0)

    return values
