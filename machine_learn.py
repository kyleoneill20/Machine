import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, silhouette_score

# 1. Load SAPS data
df_saps = pd.read_csv('SAPS2016_ED3409.csv', encoding='latin-1', low_memory=False)
df_saps['EDdesc'] = df_saps['GEOGDESC']  # so merge key exists

# 2 Helper to load two column CSVs (income, pension, welfare)
def load_two_col(path, col_name, skiprows):
    df = pd.read_csv(
        path,
        encoding='latin-1',
        skiprows=skiprows,
        header=None,
        names=['RawED', col_name],
        low_memory=False
    )
    # Strip Co. Carlow
    df['EDdesc'] = (
        df['RawED']
          .str.replace(r'^\s*\d+\s+', '', regex=True)
          .str.replace(r',\s*Co\. Carlow$', '', regex=True)
    )
    return df.dropna(subset=['EDdesc'])[['EDdesc', col_name]]

# 3 Load the other CSVs
df_gross = load_two_col('EDGrossIncome2016.csv', 'Median_Gross_HH_Income', skiprows=2)
df_sd    = load_two_col('EDMedianGrossIncomeSDCommuter.csv', 'Median_Gross_HH_Income_SD', skiprows=4)
df_ld    = load_two_col('EDMedianGrossIncomeLDCommuter.csv', 'Median_Gross_HH_Income_LD', skiprows=4)
df_pen   = load_two_col('EDPension.csv', 'Pension_%', skiprows=4)
df_sw    = load_two_col('EDSocialWelfare.csv', 'SocialWelfare_%', skiprows=4)

# 4 Load sector proportions
sec = pd.read_csv('EDProportionSector.csv', encoding='latin-1', skiprows=3, header=0, low_memory=False)
# Drop any row where the first column contains "2016"
sec = sec[~sec.iloc[:,0].astype(str).str.contains(r'2016')].copy()
# Rename and clean columns
raw_col = sec.columns[1]
sec = sec.rename(columns={raw_col:'RawED'}).drop(columns=[sec.columns[0]])
sec['EDdesc'] = (
    sec['RawED']
       .str.replace(r'^\s*\d+\s+', '', regex=True)
       .str.replace(r',\s*Co\. Carlow$', '', regex=True)
)
sec = sec.drop(columns=['RawED'])
# Reorder so EDdesc is first
sec = sec[['EDdesc'] + [c for c in sec.columns if c!='EDdesc']]

# 5 Merge everything on EDdesc
df = df_saps[['EDdesc'] + [c for c in df_saps.columns if c not in ('GUID','GEOGID','EDdesc')]]
for other in (df_gross, df_sd, df_ld, df_pen, df_sw, sec):
    df = df.merge(other, on='EDdesc', how='inner')
print(f"Merged dataset: {df.shape[0]} EDs × {df.shape[1]} features")

# 6 Preparing for modeling
#    Only keep numeric columns , and separate the target
numeric = df.select_dtypes(include=['number']).copy()
target = 'Median_Gross_HH_Income'
# If the target isn't detected as numeric. bring it in 
if target not in numeric.columns:
    numeric[target] = df[target]
X = numeric.drop(columns=[target])
y = numeric[target]

# 6.1 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 6.2 Regression
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
print("Linear   R²:", r2_score(y_test, lr.predict(X_test)))
print("RandomForest R²:", r2_score(y_test, rf.predict(X_test)))
print("Linear   MSE:", mean_squared_error(y_test, lr.predict(X_test)))
print("RandomForest MSE:", mean_squared_error(y_test, rf.predict(X_test)))

# 6.3 Classification 
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
y_bin = kbd.fit_transform(y.values.reshape(-1,1)).ravel()
Xtr, Xte, ytr, yte = train_test_split(X, y_bin, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
clf_lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("RF Accuracy:", accuracy_score(yte, clf_rf.predict(Xte)))
print("LR Accuracy:", accuracy_score(yte, clf_lr.predict(Xte)))

# 6.4 Clustering k=4 KMeans
X_std = StandardScaler().fit_transform(X)
clusters = KMeans(n_clusters=4, random_state=42).fit(X_std)
print("Silhouette Score:", silhouette_score(X_std, clusters.labels_))

# 6.5 Top 10 feature importances from the RF regressor
importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
print("\nTop 10 Predictive Features:\n", importances)
 