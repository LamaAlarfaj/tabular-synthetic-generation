# Synthetic Data Generation for Tabular Datasets
# Using CTGAN for mixed numerical and categorical data
# Covers: Insurance dataset (pure tabular) + Real Estate dataset (tabular + text)

# ============================================================
# SETUP & IMPORTS
# ============================================================

import os
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from ctgan import CTGAN

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    print("SDV not installed. Skipping GaussianCopula baseline.")
    SDV_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def dataset_summary(df, name="Dataset"):
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"{'='*60}\n")


def column_dictionary(df):
    data_dict = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        unique_count = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:3].tolist() if df[col].dtype in ['object', 'category'] \
            else df[col].dropna().iloc[:3].tolist()
        data_dict.append({
            'Column': col,
            'Type': dtype,
            'Unique': unique_count,
            'Missing': f"{missing} ({missing_pct:.1f}%)",
            'Sample Values': str(sample_vals)[:50]
        })
    return pd.DataFrame(data_dict)


def plot_numeric_distributions(df, columns, bins=30, figsize=(15, 4)):
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    for idx, col in enumerate(columns):
        axes[idx].hist(df[col].dropna(), bins=bins, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, columns, top_k=10, figsize=(15, 4)):
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    for idx, col in enumerate(columns):
        value_counts = df[col].value_counts().head(top_k)
        axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def compare_distributions(real_df, synth_df, column, column_type='numeric', bins=30):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if column_type == 'numeric':
        axes[0].hist(real_df[column].dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title(f'Real Data: {column}', fontweight='bold')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        axes[1].hist(synth_df[column].dropna(), bins=bins, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title(f'Synthetic Data: {column}', fontweight='bold')
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('Frequency')
    else:
        real_counts = real_df[column].value_counts().head(10)
        synth_counts = synth_df[column].value_counts().head(10)
        axes[0].bar(range(len(real_counts)), real_counts.values, alpha=0.7, color='blue')
        axes[0].set_title(f'Real Data: {column}', fontweight='bold')
        axes[0].set_xticks(range(len(real_counts)))
        axes[0].set_xticklabels(real_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[1].bar(range(len(synth_counts)), synth_counts.values, alpha=0.7, color='orange')
        axes[1].set_title(f'Synthetic Data: {column}', fontweight='bold')
        axes[1].set_xticks(range(len(synth_counts)))
        axes[1].set_xticklabels(synth_counts.index, rotation=45, ha='right')
        axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, title="Correlation Heatmap", figsize=(10, 8)):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


def sanity_checks(df, name="Dataset"):
    print(f"\nSanity Checks for {name}")
    print(f"{'-'*50}")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count} ({(dup_count/len(df)*100):.2f}%)")
    missing = df.isna().sum().sum()
    print(f"Missing values: {missing} ({(missing/df.size*100):.2f}%)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns range check:")
        for col in numeric_cols:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    print(f"{'-'*50}\n")


# ============================================================
# CASE 1: PURE TABULAR — INSURANCE DATASET
# ============================================================

DATA_PATH_CASE1 = "/home/insurance.csv"
df_insurance = pd.read_csv(DATA_PATH_CASE1)

dataset_summary(df_insurance, "Insurance Dataset")
print(column_dictionary(df_insurance))

# --- EDA ---
print(df_insurance.describe())
for col in ['sex', 'smoker', 'region']:
    print(f"\n{col.upper()}:")
    print(df_insurance[col].value_counts())
    print(df_insurance[col].value_counts(normalize=True).round(3))

plot_numeric_distributions(df_insurance, ['age', 'bmi', 'children', 'charges'])
plot_categorical_distributions(df_insurance, ['sex', 'smoker', 'region'])
plot_correlation_heatmap(df_insurance, title="Insurance Dataset Correlations")

# --- Baseline Model (trained on real data) ---
X = df_insurance.drop('charges', axis=1)
y = df_insurance['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

categorical_cols = ['sex', 'smoker', 'region']
numeric_cols = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

baseline_metrics = {
    'r2': r2_score(y_test, y_pred),
    'mae': mean_absolute_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
}

print(f"\nBaseline (Real Data): R2={baseline_metrics['r2']:.4f} | MAE=${baseline_metrics['mae']:,.2f} | RMSE=${baseline_metrics['rmse']:,.2f}")

# --- CTGAN v1 (initial run — to demonstrate common pitfalls) ---
discrete_columns = ['sex', 'smoker', 'region']

ctgan_model = CTGAN(epochs=500, batch_size=500, verbose=True, cuda=True)
ctgan_model.fit(df_insurance, discrete_columns=discrete_columns)

df_synthetic = ctgan_model.sample(len(df_insurance))
df_synthetic = df_synthetic[~(df_synthetic['charges'] < 0)]

sanity_checks(df_synthetic, "Synthetic Insurance (v1)")

for col in discrete_columns:
    invalid = set(df_synthetic[col].unique()) - set(df_insurance[col].unique())
    if invalid:
        print(f"Invalid categories in {col}: {invalid}")

# Fidelity checks
for col in ['age', 'bmi', 'charges']:
    compare_distributions(df_insurance, df_synthetic, col, 'numeric')
for col in ['smoker', 'region']:
    compare_distributions(df_insurance, df_synthetic, col, 'categorical')

plot_correlation_heatmap(df_insurance, title="Real Data Correlations")
plot_correlation_heatmap(df_synthetic, title="Synthetic Data Correlations")

# Utility: train on synthetic, test on real
X_synth = df_synthetic.drop('charges', axis=1)
y_synth = df_synthetic['charges']

synth_model = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
])
synth_model.fit(X_synth, y_synth)
y_pred_synth = synth_model.predict(X_test)

r2_synth = r2_score(y_test, y_pred_synth)
mae_synth = mean_absolute_error(y_test, y_pred_synth)
rmse_synth = np.sqrt(mean_squared_error(y_test, y_pred_synth))

print(f"\nSynthetic v1: R2={r2_synth:.4f} | MAE=${mae_synth:,.2f} | RMSE=${rmse_synth:,.2f}")

# --- Diagnostics ---
from scipy.stats import ks_2samp, wasserstein_distance

for col in ['charges', 'bmi', 'age']:
    ks_stat, ks_pval = ks_2samp(df_insurance[col], df_synthetic[col])
    wasserstein = wasserstein_distance(df_insurance[col], df_synthetic[col])
    print(f"{col}: KS={ks_stat:.4f} (p={ks_pval:.4e}) | Wasserstein={wasserstein:.2f}")

for col in ['age', 'children']:
    n_fractional = (~(df_synthetic[col] == df_synthetic[col].astype(int))).sum()
    print(f"{col} fractional values: {n_fractional}")

for dataset_name, df in [("REAL", df_insurance), ("SYNTHETIC v1", df_synthetic)]:
    smoker_yes = df[df['smoker'] == 'yes']['charges']
    smoker_no = df[df['smoker'] == 'no']['charges']
    ratio = smoker_yes.mean() / smoker_no.mean() if smoker_no.mean() > 0 else 0
    print(f"{dataset_name} | Smoker YES mean: ${smoker_yes.mean():,.2f} | NO mean: ${smoker_no.mean():,.2f} | Ratio: {ratio:.2f}x")

# --- CTGAN v2 (improved: log transform + smaller batch + better postprocessing) ---

df_insurance_train_v2 = df_insurance.copy()
df_insurance_train_v2['charges_log'] = np.log1p(df_insurance_train_v2['charges'])
df_insurance_train_v2 = df_insurance_train_v2.drop('charges', axis=1)

try:
    ctgan_model_v2 = CTGAN(
        epochs=1000, batch_size=128, pac=10,
        generator_dim=(256, 256), discriminator_dim=(256, 256),
        verbose=True, cuda=True
    )
except TypeError:
    ctgan_model_v2 = CTGAN(epochs=1000, batch_size=128, verbose=True, cuda=True)

ctgan_model_v2.fit(df_insurance_train_v2, discrete_columns=discrete_columns)

df_synthetic_v2_raw = ctgan_model_v2.sample(len(df_insurance))
df_synthetic_v2 = df_synthetic_v2_raw.copy()

# Postprocessing
df_synthetic_v2['charges'] = np.expm1(df_synthetic_v2['charges_log']).clip(lower=0)
df_synthetic_v2 = df_synthetic_v2.drop('charges_log', axis=1)

for col in ['age', 'children']:
    df_synthetic_v2[col] = df_synthetic_v2[col].round().astype(int).clip(
        lower=df_insurance[col].min(), upper=df_insurance[col].max()
    )

for col in discrete_columns:
    valid_cats = set(df_insurance[col].unique())
    invalid_cats = set(df_synthetic_v2[col].unique()) - valid_cats
    if invalid_cats:
        mode_val = df_insurance[col].mode()[0]
        for cat in invalid_cats:
            df_synthetic_v2.loc[df_synthetic_v2[col] == cat, col] = mode_val

sanity_checks(df_synthetic_v2, "Synthetic Insurance (v2)")

# Preprocessing pipeline for utility evaluation
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

X_synth_v2 = df_synthetic_v2.drop('charges', axis=1)
y_synth_v2 = df_synthetic_v2['charges']
X_synth_v2_processed = preprocessing_pipeline.transform(X_synth_v2)

model_synth_v2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_synth_v2.fit(X_synth_v2_processed, y_synth_v2)
y_pred_synth_v2 = model_synth_v2.predict(X_test_processed)

r2_synth_v2 = r2_score(y_test, y_pred_synth_v2)
mae_synth_v2 = mean_absolute_error(y_test, y_pred_synth_v2)
rmse_synth_v2 = np.sqrt(mean_squared_error(y_test, y_pred_synth_v2))

print(f"\n{'='*80}")
print(f"{'Metric':<15} {'Baseline (Real)':<20} {'Synthetic v1':<20} {'Synthetic v2'}")
print(f"{'='*80}")
print(f"{'R2':<15} {baseline_metrics['r2']:<20.4f} {r2_synth:<20.4f} {r2_synth_v2:.4f}")
print(f"{'MAE':<15} {baseline_metrics['mae']:<20,.2f} {mae_synth:<20,.2f} {mae_synth_v2:,.2f}")
print(f"{'RMSE':<15} {baseline_metrics['rmse']:<20,.2f} {rmse_synth:<20,.2f} {rmse_synth_v2:,.2f}")
print(f"{'='*80}")

# --- Optional: SDV GaussianCopula Baseline ---
if SDV_AVAILABLE:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_insurance)
    gc_model = GaussianCopulaSynthesizer(metadata)
    gc_model.fit(df_insurance)
    df_gc_synthetic = gc_model.sample(len(df_insurance))

    X_gc_synth = df_gc_synthetic.drop('charges', axis=1)
    y_gc_synth = df_gc_synthetic['charges']

    gc_rf = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ])
    gc_rf.fit(X_gc_synth, y_gc_synth)
    y_pred_gc = gc_rf.predict(X_test)

    r2_gc = r2_score(y_test, y_pred_gc)
    mae_gc = mean_absolute_error(y_test, y_pred_gc)
    print(f"\nGaussianCopula: R2={r2_gc:.4f} | MAE=${mae_gc:,.2f}")


# ============================================================
# CASE 2: TABULAR + TEXT — REAL ESTATE DATASET
# ============================================================

DATA_PATH_CASE2 = "/home/SA_Aqar.csv"
df_real_estate = pd.read_csv(DATA_PATH_CASE2)

dataset_summary(df_real_estate, "Real Estate Dataset")
print(column_dictionary(df_real_estate))

# Clean and sample
df_real_estate_clean = df_real_estate.dropna(subset=['price', 'details', 'city']).copy()
df_real_estate_clean = df_real_estate_clean.sample(
    n=min(5000, len(df_real_estate_clean)), random_state=RANDOM_SEED
)

# EDA
numeric_re_cols = ['size', 'property_age', 'bedrooms', 'bathrooms', 'price']
print(df_real_estate_clean[numeric_re_cols].describe())
print(df_real_estate_clean['city'].value_counts().head(10))
plot_numeric_distributions(df_real_estate_clean, ['price', 'size', 'bedrooms'])

# Text exploration
df_real_estate_clean['text_length'] = df_real_estate_clean['details'].fillna('').str.len()
df_real_estate_clean['text_word_count'] = df_real_estate_clean['details'].fillna('').str.split().str.len()
print(f"Avg characters: {df_real_estate_clean['text_length'].mean():.0f}")
print(f"Avg words: {df_real_estate_clean['text_word_count'].mean():.0f}")

# Structured columns only
structured_cols = ['city', 'front', 'size', 'property_age', 'bedrooms', 'bathrooms',
                   'furnished', 'ac', 'pool', 'garage', 'price']

df_structured = df_real_estate_clean[structured_cols].copy().fillna({
    'size': df_real_estate_clean['size'].median(),
    'property_age': 0,
    'bedrooms': df_real_estate_clean['bedrooms'].median(),
    'bathrooms': df_real_estate_clean['bathrooms'].median(),
    'furnished': 0, 'ac': 0, 'pool': 0, 'garage': 0
})

discrete_cols_re = ['city', 'front', 'furnished', 'ac', 'pool', 'garage']


# --- Approach A: CTGAN on structured columns only ---

ctgan_re = CTGAN(epochs=100, batch_size=500, verbose=True, cuda=False)
ctgan_re.fit(df_structured, discrete_columns=discrete_cols_re)
df_synth_structured = ctgan_re.sample(len(df_structured))

sanity_checks(df_synth_structured, "Synthetic Real Estate (Structured)")
compare_distributions(df_structured, df_synth_structured, 'price', 'numeric')
compare_distributions(df_structured, df_synth_structured, 'city', 'categorical')


# --- Approach B: Template-Based Text Generation ---

def generate_template_text(row):
    templates = [
        f"Property for rent in {row['city']}. {int(row['bedrooms'])} bedrooms, {int(row['bathrooms'])} bathrooms, "
        f"{int(row['size'])} sqm. {'Furnished' if row['furnished'] == 1 else 'Unfurnished'}. "
        f"{'With AC' if row['ac'] == 1 else 'No AC'}. {'Pool available' if row['pool'] == 1 else 'No pool'}. "
        f"Price: {int(row['price'])} SAR.",

        f"Available in {row['city']}: {int(row['bedrooms'])}-bedroom property, {int(row['size'])} sqm, "
        f"{int(row['bathrooms'])} bathrooms. {'Includes furniture' if row['furnished'] == 1 else 'Not furnished'}. "
        f"{'Air conditioning installed.' if row['ac'] == 1 else ''} "
        f"{'Swimming pool.' if row['pool'] == 1 else ''} Monthly rent: {int(row['price'])} SAR.",

        f"{int(row['bedrooms'])} BR in {row['city']}, {int(row['size'])} sqm, {int(row['bathrooms'])} bathrooms. "
        f"{'Fully furnished.' if row['furnished'] == 1 else 'Unfurnished.'} "
        f"{'Central AC.' if row['ac'] == 1 else ''} Contact for {int(row['price'])} SAR/month."
    ]
    return np.random.choice(templates)

df_synth_with_text = df_synth_structured.copy()
df_synth_with_text['details'] = df_synth_with_text.apply(generate_template_text, axis=1)
print("\nSample synthetic entry (template-based):")
print(df_synth_with_text[['city', 'price', 'details']].iloc[0])


# --- Approach C: Clustering-Based Text Representation ---

vectorizer = TfidfVectorizer(max_features=100, min_df=5, max_df=0.8)
text_vectors = vectorizer.fit_transform(df_real_estate_clean['details'].fillna(''))

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
clusters = kmeans.fit_predict(text_vectors)

df_with_clusters = df_structured.copy()
df_with_clusters['text_cluster'] = clusters

ctgan_clustered = CTGAN(epochs=100, batch_size=500, verbose=True, cuda=False)
ctgan_clustered.fit(df_with_clusters, discrete_columns=discrete_cols_re + ['text_cluster'])

df_synth_clustered = ctgan_clustered.sample(len(df_structured))
df_synth_clustered['text_cluster'] = df_synth_clustered['text_cluster'].round().clip(0, n_clusters - 1).astype(int)

cluster_to_texts = {
    i: df_real_estate_clean[clusters == i]['details'].values
    for i in range(n_clusters)
}

df_synth_clustered['details'] = df_synth_clustered['text_cluster'].apply(
    lambda cid: np.random.choice(cluster_to_texts[cid]) if len(cluster_to_texts.get(cid, [])) > 0
    else "Property description not available."
)
print("\nSample synthetic entry (cluster-based):")
print(df_synth_clustered[['city', 'price', 'text_cluster', 'details']].iloc[0])


# --- Approach D: LLM-Based Text Generation (placeholder) ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    def generate_llm_text(row):
        # In production: call OpenAI or similar API here.
        # Example prompt:
        # f"Generate a generic property listing in Arabic for: {row['bedrooms']} BR,
        #   {row['bathrooms']} BA, {row['size']} sqm in {row['city']}, {row['price']} SAR."
        return f"[LLM-generated text for {int(row['bedrooms'])} BR in {row['city']}]"

    print("LLM text generation placeholder active. Set OPENAI_API_KEY to enable.")
else:
    print("OPENAI_API_KEY not set. Skipping LLM text generation.")


# --- Utility Evaluation (Real Estate) ---

X_re = df_structured.drop('price', axis=1)
y_re = df_structured['price']
X_train_re, X_test_re, y_train_re, y_test_re = train_test_split(
    X_re, y_re, test_size=0.2, random_state=RANDOM_SEED
)

numeric_re_model_cols = ['size', 'property_age', 'bedrooms', 'bathrooms']
categorical_re_model_cols = ['city', 'front', 'furnished', 'ac', 'pool', 'garage']

preprocessor_re = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_re_model_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     categorical_re_model_cols)
])

baseline_re_model = Pipeline([
    ('preprocessor', preprocessor_re),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
])
baseline_re_model.fit(X_train_re, y_train_re)
y_pred_re = baseline_re_model.predict(X_test_re)

r2_re_baseline = r2_score(y_test_re, y_pred_re)
mae_re_baseline = mean_absolute_error(y_test_re, y_pred_re)
rmse_re_baseline = np.sqrt(mean_squared_error(y_test_re, y_pred_re))

X_synth_re = df_synth_structured.drop('price', axis=1)
y_synth_re = df_synth_structured['price']

synth_re_model = Pipeline([
    ('preprocessor', preprocessor_re),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
])
synth_re_model.fit(X_synth_re, y_synth_re)
y_pred_synth_re = synth_re_model.predict(X_test_re)

r2_synth_re = r2_score(y_test_re, y_pred_synth_re)
mae_synth_re = mean_absolute_error(y_test_re, y_pred_synth_re)
rmse_synth_re = np.sqrt(mean_squared_error(y_test_re, y_pred_synth_re))

print(f"\nReal Estate Utility:")
print(f"{'='*60}")
print(f"{'Metric':<15} {'Baseline (Real)':<20} {'Synthetic'}")
print(f"{'='*60}")
print(f"{'R2':<15} {r2_re_baseline:<20.4f} {r2_synth_re:.4f}")
print(f"{'MAE (SAR)':<15} {mae_re_baseline:<20,.2f} {mae_synth_re:,.2f}")
print(f"{'RMSE (SAR)':<15} {rmse_re_baseline:<20,.2f} {rmse_synth_re:,.2f}")
print(f"{'='*60}")


# ============================================================
# ADVANCED: LLM EXTRACTION + ENRICHED CTGAN (SDV)
# ============================================================
# This section uses vLLM to extract structured fields from Arabic property descriptions,
# then trains a second CTGAN on the enriched dataset to compare utility.

# To run this block:
#   1. Install vLLM: pip install vllm
#   2. Set MODEL_ID env variable or use the default below
#   3. Have a GPU with sufficient memory (Qwen3-4B needs ~8GB VRAM)

try:
    import time
    from tqdm.auto import tqdm
    from vllm import LLM, SamplingParams

    MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    TP = 1

    llm = LLM(model=MODEL_ID, tensor_parallel_size=TP, dtype="auto", trust_remote_code=True)
    gen = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=220)

    SYSTEM_PROMPT = (
        "You extract structured fields from Arabic real-estate listing text. "
        "Return ONLY valid JSON. If a field is not mentioned, use null. Do not guess."
    )

    SCHEMA = {
        "listing_type": ["rent", "sale", "unknown"],
        "property_type": ["villa", "duplex", "apartment", "floor", "townhouse", "land", "building", "unknown"],
        "city_from_text": None,
        "street_width_m": None,
        "corner_lot": None,
        "payment_mode": ["yearly", "one_payment", "two_payments", "three_payments", "monthly", "unknown"],
        "negotiable": None,
        "furnished": ["furnished", "unfurnished", "unknown"],
        "ac_type": ["central", "split", "unknown"],
        "kitchen_installed": None,
        "has_driver_room": None,
        "has_maid_room": None,
        "has_laundry_room": None,
        "has_annex": None,
        "warranty_years": None,
        "near_landmark": None,
        "summary_1_sentence_ar": None,
    }

    def build_prompt(details_text: str) -> str:
        details_text = str(details_text).strip()
        json_template = {k: None for k in SCHEMA}
        allowed_values_hint = (
            f"Allowed values:\n"
            f"- listing_type: {SCHEMA['listing_type']}\n"
            f"- property_type: {SCHEMA['property_type']}\n"
            f"- payment_mode: {SCHEMA['payment_mode']}\n"
            f"- furnished: {SCHEMA['furnished']}\n"
            f"- ac_type: {SCHEMA['ac_type']}\n"
            "Booleans must be true/false. Numbers must be numeric.\n"
        )
        return (
            "<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
            "<|im_start|>user\nText:\n" + details_text[:2500] + "\n\n"
            + allowed_values_hint
            + "Fill this JSON template (return JSON only):\n"
            + json.dumps(json_template, ensure_ascii=False)
            + "\n<|im_end|>\n<|im_start|>assistant\n"
        )

    def parse_json_safely(text: str):
        text = (text or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    # Extract structured fields from 1000 listings
    N_ROWS = 1000
    BATCH_SIZE = 50
    base_df = df_real_estate.dropna(subset=["details"]).head(N_ROWS).copy().reset_index(drop=True)

    all_objs = []
    failed = 0
    for start in range(0, len(base_df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(base_df))
        texts = base_df.loc[start:end - 1, "details"].astype(str).tolist()
        prompts = [build_prompt(t) for t in texts]
        outputs = llm.generate(prompts, gen)
        for out in outputs:
            obj = parse_json_safely(out.outputs[0].text)
            if obj is None:
                failed += 1
                obj = {"_parse_failed": True}
            else:
                obj["_parse_failed"] = False
            all_objs.append(obj)

    extracted_1000 = pd.DataFrame(all_objs)
    df_enriched_1000 = pd.concat([base_df, extracted_1000], axis=1)
    print(f"Extraction done. Failed: {failed}")

    # Prepare and train CTGAN on enriched data
    def prepare_df_for_model(df, target_col="price"):
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(subset=[target_col])
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
        numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_ = [c for c in df.columns if c not in numeric_cols_]
        for c in numeric_cols_:
            if c == target_col:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
        for c in cat_cols_:
            df[c] = df[c].astype(str).fillna("MISSING")
            df.loc[df[c].isin(["nan", "None", "NaN"]), c] = "MISSING"
        return df

    df_original_structured = df_real_estate.drop(columns=["details"], errors="ignore")
    df_original_structured = prepare_df_for_model(df_original_structured)

    df_enriched_structured = df_enriched_1000.drop(columns=["details", "_raw"], errors="ignore")
    df_enriched_structured = df_enriched_structured[df_enriched_structured.get("_parse_failed", False) == False]
    df_enriched_structured = df_enriched_structured.drop(columns=["_parse_failed"], errors="ignore")
    df_enriched_structured = prepare_df_for_model(df_enriched_structured)

    train_real_orig, test_real_orig = train_test_split(df_original_structured, test_size=0.2, random_state=RANDOM_SEED)
    train_real_enr, test_real_enr = train_test_split(df_enriched_structured, test_size=0.2, random_state=RANDOM_SEED)

    def train_ctgan_and_sample(df_train, n_samples=8000, epochs=400):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_train)
        ctgan = CTGANSynthesizer(
            metadata=metadata, epochs=epochs,
            enforce_min_max_values=True, enforce_rounding=True,
            verbose=True, cuda=True
        )
        ctgan.fit(df_train)
        return ctgan.sample(n_samples)

    syn_orig = prepare_df_for_model(train_ctgan_and_sample(train_real_orig))
    syn_enr = prepare_df_for_model(train_ctgan_and_sample(train_real_enr))

    def train_and_eval_regression(df_train, df_test, target_col="price"):
        X_train_ = df_train.drop(columns=[target_col])
        y_train_ = df_train[target_col].values
        X_test_ = df_test.drop(columns=[target_col])
        y_test_ = df_test[target_col].values
        num_cols_ = X_train_.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_ = [c for c in X_train_.columns if c not in num_cols_]
        pre = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols_),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_),
        ], remainder="drop")
        model = Pipeline([
            ("preprocessor", pre),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1))
        ])
        model.fit(X_train_, y_train_)
        preds = model.predict(X_test_)
        return {
            "R2": r2_score(y_test_, preds),
            "MAE": mean_absolute_error(y_test_, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test_, preds))
        }

    jobs = [
        ("original", "TRTR", train_real_orig, test_real_orig),
        ("original", "TSTR", syn_orig, test_real_orig),
        ("enriched", "TRTR", train_real_enr, test_real_enr),
        ("enriched", "TSTR", syn_enr, test_real_enr),
    ]

    results = []
    for dataset, setting, df_tr, df_te in tqdm(jobs):
        metrics = train_and_eval_regression(df_tr, df_te)
        results.append({"dataset": dataset, "setting": setting, **metrics})

    results_df = pd.DataFrame(results)
    print("\nUtility Comparison (Enriched vs Original):")
    print(results_df.to_string(index=False))

except ImportError:
    print("vLLM not installed. Skipping LLM extraction block.")
