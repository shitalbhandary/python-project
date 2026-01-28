#Statistical Tests with Phython
# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols

# %%
df = pd.read_excel('~/Downloads/Sales.xlsx')
summary = df.describe()
print(f'Sales summary: {summary}')

advert_freq = df['Advert'].value_counts()
print(f'Frequency of advert categories: {advert_freq}')

price_freq = df['Price'].value_counts()
print(f'Frequency of price categories: {price_freq}')

advert_summary = df.groupby('Advert')['Sales'].mean()
print(f'Sales summary by advert categories: {advert_summary}')

price_summary = df.groupby('Price')['Sales'].mean()
print(f'Sales summary by price categories: {price_summary}')

# %%
#Shapiro-Wilk test
df1 = df['Sales']
stat, p = stats.shapiro(df1)
print(f'Shapiro-Wilk: stat={stat:.3f}, p={p:.3f}')
if p > 0.05:
    print("Sales data is likely normal")
else:
    print("Sales data is likely not normal")

# Q-Q Plot using statsmodels
sm.qqplot(df1, line='s') # 's' for standardized line
plt.title('Q-Q Plot for Normal Data:Sales')
plt.show()

# %%
#Data filter
df2 = df[df['Advert'] == 'Low'] ['Sales']
df3 = df[df['Advert'] == 'High'] ['Sales']

#Variacne test (two groups)

statistic, p_value = levene(df2,df3)
print(f'LeveneTest: stat={statistic:.3f}, p={p_value:.3f}')
if p_value > 0.05:
    print("Group variance are equal")
else:
    print("Group variance are not equal")

# %%
#Student's two sample t-test
t_statistic, tp_value = stats.ttest_ind(df2,df3, equal_var=True)
print(f"Welch's t-statistic: {t_statistic}")
print(f"P-value: {tp_value}")
if tp_value > 0.05:
    print("Group means are equal")
else:
    print("Group means are not equal")

# %%
#Variacne test (Three groups)
#Data filter
df4 = df[df['Price'] == 'Low'] ['Sales']
df5 = df[df['Price'] == 'Medium'] ['Sales']
df6 = df[df['Price'] == 'High'] ['Sales']

l_statistic, lp_value = levene(df4,df5,df6)
print(f'LeveneTest: stat={l_statistic:.3f}, p={lp_value:.3f}')
if lp_value > 0.05:
    print("Group variance are equal")
else:
    print("Group variance are not equal")

# %%
#One-way ANOVA F-test
f_statistic, fp_value = stats.f_oneway(df4,df5,df6, equal_var=True)
print(f"One-way ANOVA: {f_statistic}")
print(f"ANOVA P-value: {fp_value}")
if fp_value > 0.05:
    print("Group means are equal")
else:
    print("Group means are not equal: Post-hoc test needed")

#Post-hoc test
# Perform Tukey's HSD post hoc test
tukey_results = pairwise_tukeyhsd(endog=df['Sales'], groups=df['Price'], alpha=0.05)

# Display the results
print("\nTukey's HSD Post Hoc Test Results:")
print(tukey_results)

# %%
#Two-way ANOVAx: Advert and Price together
model = ols('Sales ~ C(Advert) + C(Price) + C(Advert):C(Price)', data=df).fit()
anova_table = sm.stats.anova_lm(model, type=2)
print(f'Two-way ANOVA: {anova_table}')
print('Interaction term is NOT statistically significant')

# %%
#Linear regression
lm = smf.ols("Sales ~ C(Advert) + C(Price)", data=df).fit()
print(lm.summary())