import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_nan_col_slice(df: pd.DataFrame, first_nan_col, nan_index: int):
    return df.loc[:,first_nan_col].iloc[nan_index-50:nan_index+50]

df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, np.nan],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
first_nan_col = df.loc[:, df.isna().any()].columns[0]
nan_index = df.loc[:,first_nan_col].loc[df.loc[:,first_nan_col].isna()].index[0]
x = np.linspace(0, 3, 4)
y = np.array(get_nan_col_slice(df, first_nan_col, nan_index))
df = df.fillna(
                method="ffill").fillna(method="bfill")
x_new = np.linspace(0, 3, 4)
y_new = np.array(get_nan_col_slice(df, first_nan_col, nan_index))
plt.plot(x, y, 'g.-')
plt.savefig('before.png')
plt.show()
plt.plot(x_new, y_new, 'r.-')
plt.savefig('after.png')
plt.show()