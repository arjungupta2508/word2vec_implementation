import numpy as np

corpus = """
the king is a strong man
the queen is a strong woman
the king rules the kingdom
the queen rules the kingdom
the king wears a crown
the queen wears a crown
the king lives in a castle
the queen lives in a castle
the king has royal power
the queen has royal power
the king is respected by the people
the queen is respected by the people
the prince is a young king
the princess is a young queen
the prince will inherit the throne
the princess will inherit the throne
the prince wears fine clothes
the princess wears fine clothes
the man is strong
the woman is strong
the man works in the city
the woman works in the city
the man walks to work
the woman walks to work
the man reads the news
the woman reads the news
the boy is a young man
the girl is a young woman
the boy plays with the ball
the girl plays with the ball
the boy runs in the yard
the girl runs in the yard
the boy laughs with his friends
the girl laughs with her friends

the dog is a small animal
the cat is a small animal
the dog is a pet
the cat is a pet
the dog runs in the park
the cat runs in the park
the dog eats food
the cat eats food
the dog sleeps at night
the cat sleeps at night
the dog likes to play
the cat likes to play
the dog has soft fur
the cat has soft fur
the dog barks loudly
the cat meows loudly
the lion is a big animal
the tiger is a big animal
the lion is a wild animal
the tiger is a wild animal
the lion hunts in the jungle
the tiger hunts in the jungle
the lion eats meat
the tiger eats meat
the lion has sharp claws
the tiger has sharp claws
the lion is a dangerous animal
the tiger is a dangerous animal
the lion roars loudly
the tiger roars loudly

the car is fast
the bike is fast
the car has wheels
the bike has wheels
the car moves on the road
the bike moves on the road
the car needs fuel
the bike needs pedals
the car is a vehicle
the bike is a vehicle
the car is parked outside
the bike is parked outside
the car carries passengers
the bike carries one rider

the apple is a fruit
the banana is a fruit
the apple is sweet
the banana is sweet
the apple grows on a tree
the banana grows on a tree
the apple is healthy food
the banana is healthy food
the apple has a peel
the banana has a peel
the apple is sold at the market
the banana is sold at the market
the apple is red or green
the banana is yellow

the doctor works in a hospital
the nurse works in a hospital
the doctor helps sick people
the nurse helps sick people
the doctor wears a white coat
the nurse wears a white coat
the doctor treats patients
the nurse treats patients
the doctor gives medicine
the nurse gives medicine
the doctor is a trained professional
the nurse is a trained professional
the teacher works in a school
the student studies in a school
the teacher teaches the student
the student learns from the teacher
the teacher explains the lesson
the student listens to the lesson
the teacher gives homework
the student does homework
the teacher is a trained professional
the student is a young learner

the sun is bright
the moon is bright
the sun rises in the morning
the moon rises in the evening
the sun is in the sky
the moon is in the sky
the sun gives light
the moon gives light
the ocean is deep and blue
the sky is bright and blue
the ocean is full of water
the sky is full of clouds
the fish swims in the ocean
the bird flies in the sky
the fish lives in water
the bird lives in trees
the fish has fins
the bird has wings
the fish is a sea animal
the bird is a sky animal
"""

import string

# Strip punctuation so "park." and "park" aren't treated as different tokens
translator = str.maketrans("", "", string.punctuation)
words = corpus.lower().translate(translator).split()
print(f"total words in corpus = {len(words)}")

vocab = sorted(set(words))
vocab_size = len(vocab)
print(f"total words in vocab = {len(vocab)}")

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

WINDOW_SIZE = 3


def create_training_pairs(words, window_size):
    pairs = []
    for centre_pos in range(len(words)):
        centre_idx = word_to_idx[words[centre_pos]]
        window_start = max(0, centre_pos - window_size)
        window_end = min(len(words), centre_pos + window_size + 1)
        for neighbor_pos in range(window_start, window_end):
            if neighbor_pos != centre_pos:
                neighbor_idx = word_to_idx[words[neighbor_pos]]
                pairs.append((centre_idx, neighbor_idx))
    return pairs


training_pairs = create_training_pairs(words, WINDOW_SIZE)
print(f"total training pairs = {len(training_pairs)}")

EMBEDDING_DIM = 8
np.random.seed(42)
vectors = np.random.randn(vocab_size, EMBEDDING_DIM) * 0.1


def cosine_similarity_idx(idx1, idx2):
    v1, v2 = vectors[idx1], vectors[idx2]
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def cosine_similarity(word1, word2):
    return cosine_similarity_idx(word_to_idx[word1], word_to_idx[word2])


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))


LEARNING_RATE = 0.01
NUM_EPOCHS = 100
NUM_NEGATIVES = 5

print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    for centre_idx, neighbor_idx in training_pairs:
        centre_vec = vectors[centre_idx]
        neighbor_vec = vectors[neighbor_idx]
        similarity = np.dot(centre_vec, neighbor_vec)
        confidence = sigmoid(similarity)
        error = confidence - 1
        total_loss += -np.log(confidence + 1e-10)

        vectors[centre_idx] = centre_vec - LEARNING_RATE * error * neighbor_vec
        vectors[neighbor_idx] = neighbor_vec - LEARNING_RATE * error * centre_vec

        
        for _ in range(NUM_NEGATIVES):
            neg_idx = np.random.randint(0, vocab_size)
            neg_vec = vectors[neg_idx]
            neg_similarity = np.dot(centre_vec, neg_vec)
            neg_confidence = sigmoid(neg_similarity)
            neg_error = neg_confidence  

            total_loss += -np.log(1 - neg_confidence + 1e-10)

            vectors[centre_idx] -= LEARNING_RATE * neg_error * neg_vec
            vectors[neg_idx] -= LEARNING_RATE * neg_error * centre_vec

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(training_pairs)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")

print("\n" + "=" * 60)
print("WORD SIMILARITIES")
print("=" * 60)

sim_cat_dog = cosine_similarity("cat", "dog")
sim_lion_tiger = cosine_similarity("lion", "tiger")
sim_king_queen = cosine_similarity("king", "queen")
sim_man_woman = cosine_similarity("man", "woman")
sim_car_bike = cosine_similarity("car", "bike")
sim_apple_banana = cosine_similarity("apple", "banana")
sim_doctor_nurse = cosine_similarity("doctor", "nurse")
sim_king_bike = cosine_similarity("king", "bike")      # unrelated pair, sanity check
sim_cat_king = cosine_similarity("cat", "king")        # unrelated pair, sanity check

print(f"cat + dog          : {sim_cat_dog:+.3f}")
print(f"lion + tiger       : {sim_lion_tiger:+.3f}")
print(f"king + queen       : {sim_king_queen:+.3f}")
print(f"man + woman        : {sim_man_woman:+.3f}")
print(f"car + bike         : {sim_car_bike:+.3f}")
print(f"apple + banana     : {sim_apple_banana:+.3f}")
print(f"doctor + nurse     : {sim_doctor_nurse:+.3f}")
print(f"king + bike (unrelated) : {sim_king_bike:+.3f}")
print(f"cat + king (unrelated)  : {sim_cat_king:+.3f}")




import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Pick words you care about, grouped by category for coloring
groups = {
    "royalty": ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"],
    "animals": ["cat", "dog", "lion", "tiger"],
    "vehicles": ["car", "bike"],
    "food": ["apple", "banana"],
    "professions": ["doctor", "nurse", "teacher", "student"],
    "nature": ["sun", "moon", "sky", "ocean", "fish", "bird"],
}

colors = {"royalty": "#8B5CF6", "animals": "#F59E0B", "vehicles": "#EF4444",
          "food": "#10B981", "professions": "#3B82F6", "nature": "#06B6D4"}

words_to_plot = [w for ws in groups.values() for w in ws if w in word_to_idx]
vecs_to_plot = np.array([vectors[word_to_idx[w]] for w in words_to_plot])

pca = PCA(n_components=2)
coords = pca.fit_transform(vecs_to_plot)

plt.figure(figsize=(10, 8))
for group_name, group_words in groups.items():
    idxs = [i for i, w in enumerate(words_to_plot) if w in group_words]
    plt.scatter(coords[idxs, 0], coords[idxs, 1],
                c=colors[group_name], label=group_name, s=120, alpha=0.8, edgecolors='white', linewidths=1.5)

for i, word in enumerate(words_to_plot):
    plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=10, xytext=(5, 5), textcoords='offset points')

plt.title("Word Embeddings (PCA Projection)", fontsize=14, fontweight='bold')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='best')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("embeddings_pca.png")

import seaborn as sns

heatmap_words = ["cat", "dog", "lion", "tiger", "king", "queen", "man", "woman",
                  "car", "bike", "apple", "banana", "doctor", "nurse"]

sim_matrix = np.zeros((len(heatmap_words), len(heatmap_words)))
for i, w1 in enumerate(heatmap_words):
    for j, w2 in enumerate(heatmap_words):
        sim_matrix[i, j] = cosine_similarity(w1, w2)

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, xticklabels=heatmap_words, yticklabels=heatmap_words,
            cmap="RdYlBu_r", center=0, annot=True, fmt=".2f", square=True,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title("Pairwise Word Similarity Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("similarity_heatmap.png", dpi=150)
plt.show()

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Pick words you care about, grouped by category for coloring
groups = {
    "royalty": ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"],
    "animals": ["cat", "dog", "lion", "tiger"],
    "vehicles": ["car", "bike"],
    "food": ["apple", "banana"],
    "professions": ["doctor", "nurse", "teacher", "student"],
    "nature": ["sun", "moon", "sky", "ocean", "fish", "bird"],
}

colors = {"royalty": "#8B5CF6", "animals": "#F59E0B", "vehicles": "#EF4444",
          "food": "#10B981", "professions": "#3B82F6", "nature": "#06B6D4"}

words_to_plot = [w for ws in groups.values() for w in ws if w in word_to_idx]
vecs_to_plot = np.array([vectors[word_to_idx[w]] for w in words_to_plot])

# Normalize vectors before projecting (removes magnitude bias, keeps direction)
vecs_normalized = vecs_to_plot / np.linalg.norm(vecs_to_plot, axis=1, keepdims=True)

# t-SNE instead of PCA — better for showing tight local clusters
tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
coords = tsne.fit_transform(vecs_normalized)

plt.figure(figsize=(10, 8))
for group_name, group_words in groups.items():
    idxs = [i for i, w in enumerate(words_to_plot) if w in group_words]
    plt.scatter(coords[idxs, 0], coords[idxs, 1],
                c=colors[group_name], label=group_name, s=140, alpha=0.85,
                edgecolors='white', linewidths=1.5)

for i, word in enumerate(words_to_plot):
    plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=10,
                 xytext=(6, 6), textcoords='offset points')

plt.title("Word Embeddings (t-SNE Projection)", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(loc='best')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("embeddings_tsne.png", dpi=150)
plt.show()