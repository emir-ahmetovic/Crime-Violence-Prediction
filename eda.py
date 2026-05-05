import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("crime_clean.csv")

# ===================
# 1️⃣ Target distribucija
# ===================
plt.figure(figsize=(5,4))
sns.countplot(x='Part 1-2', data=df)
plt.title("Part 1 vs Part 2 Crimes")
plt.show()

# ===================
# 2️⃣ Crime po satu (TIME OCC)
# ===================
df['hour'] = df['TIME OCC'] // 100

plt.figure(figsize=(8,4))
sns.countplot(x='hour', data=df)
plt.title("Crimes by Hour")
plt.xticks(rotation=90)
plt.show()

# ===================
# 3️⃣ Top 10 AREA NAME
# ===================
plt.figure(figsize=(8,4))
df['AREA NAME'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Areas by Crime Count")
plt.show()

# ===================
# 4️⃣ Vict Age distribucija
# ===================
plt.figure(figsize=(6,4))
sns.histplot(df['Vict Age'], bins=30)
plt.title("Victim Age Distribution")
plt.show()

# ===================
# 5️⃣ Weapon used (ako postoji)
# ===================
if 'Weapon Used Cd' in df.columns:
    weapon_used = df['Weapon Used Cd'].notna().astype(int)
    sns.countplot(x=weapon_used)
    plt.title("Weapon Used (0 = No, 1 = Yes)")
    plt.show()
