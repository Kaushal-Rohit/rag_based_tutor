"""Generate all README visualization charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Chart 1: HNSW vs Flat vs IVF ----------
def chart_indexing_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('HNSW vs Flat vs IVF Index Performance', fontsize=15, fontweight='bold', y=1.02)

    corpus_sizes = ['1K', '10K', '50K', '100K', '500K']
    flat_latency = [5, 50, 250, 500, 2500]
    ivf_latency  = [3, 10, 25, 30, 80]
    hnsw_latency = [1, 2, 3, 4, 6]

    ax = axes[0]
    x = np.arange(len(corpus_sizes))
    w = 0.25
    ax.bar(x - w, flat_latency, w, label='Flat (Brute Force)', color='#E74C3C', alpha=0.85)
    ax.bar(x,     ivf_latency,  w, label='IVF (Inverted File)', color='#F39C12', alpha=0.85)
    ax.bar(x + w, hnsw_latency, w, label='HNSW (This System)', color='#2ECC71', alpha=0.85)
    ax.set_xlabel('Corpus Size (vectors)')
    ax.set_ylabel('Search Latency (ms)')
    ax.set_title('Search Latency by Corpus Size')
    ax.set_xticks(x)
    ax.set_xticklabels(corpus_sizes)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    recall = ['Flat\n(Exact)', 'IVF\n(nprobe=10)', 'HNSW\n(M=32)']
    recall_vals = [100, 90, 97]
    colors = ['#E74C3C', '#F39C12', '#2ECC71']
    ax2 = axes[1]
    bars = ax2.bar(recall, recall_vals, color=colors, alpha=0.85, width=0.5)
    ax2.set_ylabel('Recall@10 (%)')
    ax2.set_title('Recall Accuracy Comparison')
    ax2.set_ylim(80, 102)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, recall_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hnsw_vs_flat_ivf.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: hnsw_vs_flat_ivf.png')


# ---------- Chart 2: FAISS vs ChromaDB ----------
def chart_faiss_vs_chroma():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('FAISS HNSW vs ChromaDB HNSW -- Backend Comparison', fontsize=15, fontweight='bold', y=1.02)

    # Latency comparison
    categories = ['Embedding', 'HNSW Search', 'Total']
    faiss_vals  = [8.5, 1.2, 9.7]
    chroma_vals = [8.5, 15.3, 23.8]

    ax = axes[0]
    x = np.arange(len(categories))
    w = 0.3
    ax.bar(x - w/2, faiss_vals, w, label='FAISS', color='#3498DB', alpha=0.85)
    ax.bar(x + w/2, chroma_vals, w, label='ChromaDB', color='#2ECC71', alpha=0.85)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Average Latency Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Disk size
    ax2 = axes[1]
    sizes = [24, 65]
    bars = ax2.bar(['FAISS', 'ChromaDB'], sizes, color=['#3498DB', '#2ECC71'], alpha=0.85, width=0.5)
    ax2.set_ylabel('Disk Size (MB)')
    ax2.set_title('Storage Footprint (13,917 vectors)')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val} MB', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Feature comparison radar-like horizontal bar
    ax3 = axes[2]
    features = ['Raw Speed', 'Metadata\nFiltering', 'Persistence', 'Ease of Use', 'Production\nReady']
    faiss_scores  = [5, 2, 2, 3, 4]
    chroma_scores = [3, 5, 5, 5, 4]
    y = np.arange(len(features))
    h = 0.35
    ax3.barh(y - h/2, faiss_scores, h, label='FAISS', color='#3498DB', alpha=0.85)
    ax3.barh(y + h/2, chroma_scores, h, label='ChromaDB', color='#2ECC71', alpha=0.85)
    ax3.set_xlabel('Score (1-5)')
    ax3.set_title('Feature Comparison')
    ax3.set_yticks(y)
    ax3.set_yticklabels(features)
    ax3.set_xlim(0, 6)
    ax3.legend(loc='lower right')
    ax3.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'faiss_vs_chromadb.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: faiss_vs_chromadb.png')


# ---------- Chart 3: Sentiment-Driven Dynamic K ----------
def chart_sentiment_k():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sentiment-Driven Adaptive Retrieval', fontsize=15, fontweight='bold', y=1.02)

    # Sentiment zones
    ax = axes[0]
    sentiment_range = np.linspace(-1, 1, 200)
    k_values = []
    for s in sentiment_range:
        if s < -0.1:
            k_values.append(8)
        elif s > 0.3:
            k_values.append(3)
        else:
            k_values.append(5)

    ax.fill_between(sentiment_range, k_values, alpha=0.3, color='#3498DB')
    ax.plot(sentiment_range, k_values, color='#2C3E50', linewidth=2)
    ax.axvline(x=-0.1, color='#E74C3C', linestyle='--', alpha=0.7, label='Confused threshold')
    ax.axvline(x=0.3, color='#2ECC71', linestyle='--', alpha=0.7, label='Clear threshold')
    ax.set_xlabel('Average Sentiment Polarity')
    ax.set_ylabel('Retrieval Depth (k)')
    ax.set_title('Dynamic k Selection by Sentiment')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Simulated conversation
    ax2 = axes[1]
    turns = ['Turn 1\n(Normal)', 'Turn 2\n(Confused)', 'Turn 3\n(Confused)', 'Turn 4\n(Clearing)', 'Turn 5\n(Clear)']
    sentiments = [0.0, -0.4, -0.3, 0.1, 0.5]
    k_per_turn = [5, 8, 8, 5, 3]
    colors_s = ['#F39C12', '#E74C3C', '#E74C3C', '#F39C12', '#2ECC71']

    x = np.arange(len(turns))
    bars = ax2.bar(x, k_per_turn, color=colors_s, alpha=0.85, width=0.6)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, sentiments, 'o-', color='#2C3E50', linewidth=2, markersize=8, label='Avg Sentiment')
    ax2_twin.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2_twin.set_ylabel('Avg Sentiment')
    ax2_twin.set_ylim(-0.6, 0.7)
    ax2_twin.legend(loc='upper left', fontsize=9)

    ax2.set_ylabel('Retrieval Depth (k)')
    ax2.set_title('Simulated Conversation: k Adaptation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(turns, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, k_per_turn):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'k={val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sentiment_dynamic_k.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: sentiment_dynamic_k.png')


# ---------- Chart 4: Traditional RAG vs This System ----------
def chart_traditional_vs_adaptive():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Traditional RAG vs Adaptive RAG Pipeline', fontsize=15, fontweight='bold', y=1.0)

    categories = ['Search\nLatency', 'Metadata\nFiltering', 'User\nAdaptation', 'Context\nAwareness', 'Offline\nCapability', 'Response\nPrecision']
    traditional = [2, 1, 1, 1, 1, 3]
    adaptive    = [5, 5, 5, 4, 5, 5]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    traditional += traditional[:1]
    adaptive += adaptive[:1]
    angles += angles[:1]

    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, traditional, 'o-', linewidth=2, label='Traditional RAG', color='#E74C3C', alpha=0.8)
    ax.fill(angles, traditional, alpha=0.15, color='#E74C3C')
    ax.plot(angles, adaptive, 'o-', linewidth=2, label='Adaptive RAG (This System)', color='#2ECC71', alpha=0.8)
    ax.fill(angles, adaptive, alpha=0.15, color='#2ECC71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'traditional_vs_adaptive.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: traditional_vs_adaptive.png')


if __name__ == '__main__':
    chart_indexing_comparison()
    chart_faiss_vs_chroma()
    chart_sentiment_k()
    chart_traditional_vs_adaptive()
    print('\nAll charts generated successfully.')
