import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import scipy.stats as stats

save_path = "C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW5/"

# Load the CSV file
data = pd.read_csv("C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW5/winequality-red.csv")


# Separate features and target
features = data.drop(columns=['quality'])
target = data['quality']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_result = tsne.fit_transform(features_scaled)

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_reducer.fit_transform(features_scaled)

# Function to create scatter plots
def plot_results(reduction_result, title, targ, ax):
    scatter = ax.scatter(reduction_result[:, 0], reduction_result[:, 1], c=targ, cmap='viridis', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    return scatter

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

scatter_pca = plot_results(pca_result, "PCA", target, axes[0])
scatter_tsne = plot_results(tsne_result, "t-SNE", target, axes[1])
scatter_umap = plot_results(umap_result, "UMAP", target, axes[2])

# Add a color bar
cbar = fig.colorbar(scatter_pca, ax=axes, orientation='horizontal', fraction=0.03, pad=0.1)
cbar.set_label('Quality')

plt.show()


# Extract alcohol and quality
alcohol = data['alcohol']
quality = data['quality']

# Perform Spearman correlation
corr_spearman, p_value_spearman = stats.spearmanr(alcohol, quality)

# Perform Pearson correlation
corr_pearson, p_value_pearson = stats.pearsonr(alcohol, quality)

# Print results
print("Spearman Correlation: ", corr_spearman, ", p-value: ", p_value_spearman)
print("Pearson Correlation: ", corr_pearson, ", p-value: ", p_value_pearson)

# Plot the scatter plot with trendline
plt.figure(figsize=(10, 6))
sns.regplot(x=alcohol, y=quality, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, ci=None)
plt.title("Alcohol vs. Quality")
plt.xlabel("Alcohol Content")
plt.ylabel("Wine Quality")
plt.grid(alpha=0.3)
plt.show()
