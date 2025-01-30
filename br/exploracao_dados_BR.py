import pandas as pd # type: ignore
import numpy as np # type: ignore

from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "br/cpi_m2_gdp_BR.csv", 
    sep=",",
    na_values='.',
    dtype={'DATE': object, 'CPIAUCSL': np.float64, 'M2SL': np.float64, 'GDP': np.float64},
    decimal="."
)
df.columns = ['date', 'cpi', 'm2', 'gdp']
df["date"] = pd.to_datetime(df["date"]).dt.date
print(df.dtypes)
df.head()

df.dropna(inplace=True)
df['m2_gdp'] = df['m2'] / df['gdp']

df['m2_rolling_4'] = df['m2'].shift(4)
df['m2_rolling_8'] = df['m2'].shift(8)
df['m2_rolling_20'] = df['m2'].shift(20)

df['gdp_rolling_4'] = df['gdp'].shift(4)
df['gdp_rolling_8'] = df['gdp'].shift(8)
df['gdp_rolling_20'] = df['gdp'].shift(20)

df['m2_gdp_rolling_4'] = df['m2_gdp'].shift(4)
df['m2_gdp_rolling_8'] = df['m2_gdp'].shift(8)
df['m2_gdp_rolling_20'] = df['m2_gdp'].shift(20)

df_m2_shift_4 = df[['cpi', 'm2_rolling_4']].dropna()
pearson_coef, p_value = pearsonr(df_m2_shift_4['cpi'], df_m2_shift_4['m2_rolling_4'])
spearman_coef, p_value_sp = spearmanr(df_m2_shift_4['cpi'], df_m2_shift_4['m2_rolling_4'])
print(f"CPI x M2_roll_4\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_m2_shift_8 = df[['cpi', 'm2_rolling_8']].dropna()
pearson_coef, p_value = pearsonr(df_m2_shift_8['cpi'], df_m2_shift_8['m2_rolling_8'])
spearman_coef, p_value_sp = spearmanr(df_m2_shift_8['cpi'], df_m2_shift_8['m2_rolling_8'])
print(f"CPI x M2_roll_8\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_m2_shift_20 = df[['cpi', 'm2_rolling_20']].dropna()
pearson_coef, p_value = pearsonr(df_m2_shift_20['cpi'], df_m2_shift_20['m2_rolling_20'])
spearman_coef, p_value_sp = spearmanr(df_m2_shift_20['cpi'], df_m2_shift_20['m2_rolling_20'])
print(f"CPI x M2_roll_20\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")

df_gdp_shift_4 = df[['cpi', 'gdp_rolling_4']].dropna()
pearson_coef, p_value = pearsonr(df_gdp_shift_4['cpi'], df_gdp_shift_4['gdp_rolling_4'])
spearman_coef, p_value_sp = spearmanr(df_gdp_shift_4['cpi'], df_gdp_shift_4['gdp_rolling_4'])
print(f"CPI x GDP_roll_4\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_gdp_shift_8 = df[['cpi', 'gdp_rolling_8']].dropna()
pearson_coef, p_value = pearsonr(df_gdp_shift_8['cpi'], df_gdp_shift_8['gdp_rolling_8'])
spearman_coef, p_value_sp = spearmanr(df_gdp_shift_8['cpi'], df_gdp_shift_8['gdp_rolling_8'])
print(f"CPI x GDP_roll_8\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_gdp_shift_20 = df[['cpi', 'gdp_rolling_20']].dropna()
pearson_coef, p_value = pearsonr(df_gdp_shift_20['cpi'], df_gdp_shift_20['gdp_rolling_20'])
spearman_coef, p_value_sp = spearmanr(df_gdp_shift_20['cpi'], df_gdp_shift_20['gdp_rolling_20'])
print(f"CPI x GDP_roll_20\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")

df_m2_gdp_shift_4 = df[['cpi', 'm2_gdp_rolling_4']].dropna()
pearson_coef, p_value = pearsonr(df_m2_gdp_shift_4['cpi'], df_m2_gdp_shift_4['m2_gdp_rolling_4'])
spearman_coef, p_value_sp = spearmanr(df_m2_gdp_shift_4['cpi'], df_m2_gdp_shift_4['m2_gdp_rolling_4'])
print(f"CPI x M2_GDP_roll_4\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_m2_gdp_shift_8 = df[['cpi', 'm2_gdp_rolling_8']].dropna()
pearson_coef, p_value = pearsonr(df_m2_gdp_shift_8['cpi'], df_m2_gdp_shift_8['m2_gdp_rolling_8'])
spearman_coef, p_value_sp = spearmanr(df_m2_gdp_shift_8['cpi'], df_m2_gdp_shift_8['m2_gdp_rolling_8'])
print(f"CPI x M2_GDP_roll_8\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
df_m2_gdp_shift_20 = df[['cpi', 'm2_gdp_rolling_20']].dropna()
pearson_coef, p_value = pearsonr(df_m2_gdp_shift_20['cpi'], df_m2_gdp_shift_20['m2_gdp_rolling_20'])
spearman_coef, p_value_sp = spearmanr(df_m2_gdp_shift_20['cpi'], df_m2_gdp_shift_20['m2_gdp_rolling_20'])
print(f"CPI x M2_GDP_roll_20\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")

df['cor_cpi_m2_roll_4'] = df['cpi'].rolling(4).corr(df['m2'])
df['cor_cpi_m2_roll_8'] = df['cpi'].rolling(8).corr(df['m2'])
df['cor_cpi_m2_roll_20'] = df['cpi'].rolling(20).corr(df['m2'])
df['cor_cpi_gdp_roll_4'] = df['cpi'].rolling(4).corr(df['gdp'])
df['cor_cpi_gdp_roll_8'] = df['cpi'].rolling(8).corr(df['gdp'])
df['cor_cpi_gdp_roll_20'] = df['cpi'].rolling(20).corr(df['gdp'])
df['cor_cpi_m2gdp_roll_4'] = df['cpi'].rolling(4).corr(df['m2_gdp'])
df['cor_cpi_m2gdp_roll_8'] = df['cpi'].rolling(8).corr(df['m2_gdp'])
df['cor_cpi_m2gdp_roll_20'] = df['cpi'].rolling(20).corr(df['m2_gdp'])

df['cor_cpi_m2_roll_4'] = df['m2'].rolling(4).corr(df['cpi'])
df['cor_cpi_m2_roll_8'] = df['m2'].rolling(8).corr(df['cpi'])
df['cor_cpi_m2_roll_20'] = df['m2'].rolling(20).corr(df['cpi'])
df['cor_cpi_gdp_roll_4'] = df['gdp'].rolling(4).corr(df['cpi'])
df['cor_cpi_gdp_roll_8'] = df['gdp'].rolling(8).corr(df['cpi'])
df['cor_cpi_gdp_roll_20'] = df['gdp'].rolling(20).corr(df['cpi'])
df['cor_cpi_m2gdp_roll_4'] = df['m2_gdp'].rolling(4).corr(df['cpi'])
df['cor_cpi_m2gdp_roll_8'] = df['m2_gdp'].rolling(8).corr(df['cpi'])
df['cor_cpi_m2gdp_roll_20'] = df['m2_gdp'].rolling(20).corr(df['cpi'])

pearson_coef, p_value = pearsonr(df['cpi'], df['m2'])
spearman_coef, p_value_sp = spearmanr(df['cpi'], df['m2'])
print(f"CPI x M2\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
pearson_coef, p_value = pearsonr(df['cpi'], df['gdp'])
spearman_coef, p_value_sp = spearmanr(df['cpi'], df['gdp'])
print(f"CPI x GDP\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}\n")
pearson_coef, p_value = pearsonr(df['cpi'], df['m2_gdp'])
spearman_coef, p_value_sp = spearmanr(df['cpi'], df['m2_gdp'])
print(f"CPI x M2_GDP\nPearson  Corr.: {pearson_coef:.4f}. T-Test p-value: {p_value:.6f}")
print(f"Spearman Corr.: {spearman_coef:.4f}. T-Test p-value: {p_value_sp:.6f}")

list_cols = [x for x in df.columns if x.startswith("cor_")]

df[list_cols].median(axis=0)


fig, axis = plt.subplots(3, 3,figsize=(12, 8)) 
fig.suptitle("Histograma das correlações móveis para M2, GDP, M2/GDP (Brasil)", y=.995, fontsize=16)
fig.tight_layout() 
axis[0,0].hist(df[list_cols[0]], edgecolor = 'grey')
axis[0,0].set_title(list_cols[0])
axis[0,1].hist(df[list_cols[1]], edgecolor = 'grey')
axis[0,1].set_title(list_cols[1])
axis[0,2].hist(df[list_cols[2]], edgecolor = 'grey')
axis[0,2].set_title(list_cols[2])
axis[1,0].hist(df[list_cols[3]], edgecolor = 'grey')
axis[1,0].set_title(list_cols[3])
axis[1,1].hist(df[list_cols[4]], edgecolor = 'grey')
axis[1,1].set_title(list_cols[4])
axis[1,2].hist(df[list_cols[5]], edgecolor = 'grey')
axis[1,2].set_title(list_cols[5])
axis[2,0].hist(df[list_cols[6]], edgecolor = 'grey')
axis[2,0].set_title(list_cols[6])
axis[2,1].hist(df[list_cols[7]], edgecolor = 'grey')
axis[2,1].set_title(list_cols[7])
axis[2,2].hist(df[list_cols[8]], edgecolor = 'grey')
axis[2,2].set_title(list_cols[8])
plt.show();


fig, ax1 = plt.subplots(figsize=(10, 5))
plt.xticks(rotation=90)
plt.grid(visible=True, alpha=0.2)

# ax1.set_xlim([df['date'].values[0], df['date'].values[-1]])
ax1.xaxis.set_major_locator(mdates.YearLocator(2))

ax2 = ax1.twinx()

ax1.plot(df['date'], df['cpi'], c='blue', lw=2)
ax1.set_xlabel("")
ax1.set_ylabel("Inflação acum 12m (%)", c='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.plot(df['date'], df['m2_gdp'], c='red', lw=2)
ax2.set_ylabel("M2 / GDP", c="red")
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle("Inflação Brasil x Emissão Monetária sobre PIB - Dados Trimestrais", fontsize=14)
# fig.autofmt_xdate(which='minor')
fig.tight_layout()
plt.show();