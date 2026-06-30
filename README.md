# Beyond Political Stances

## Topic Selection

Topic selection starts from `reddit_narratives.csv`, which already contains comments that passed our narrative heuristic filter. We then sequentially scan this file, keep comments with `word_count > 80`, and take the first 50,000 comments as the working sample for topic discovery.

We run BERTopic on this 50,000-comment sample to obtain semantic clusters. Since the resulting clusters are not purely political, we do not use BERTopic labels directly as final topic labels. Instead, we compare each cluster against Comparative Agendas Project (CAP) major-topic prototypes. Each CAP prototype is written as a short semantic description built from the main topic and its representative subtopics.

For each BERTopic cluster, we construct a cluster text by combining its topic name, top terms, and representative documents. We encode both the cluster texts and the CAP prototypes with the same sentence-transformer model, compute cosine similarity, and keep the top 3 CAP candidates for each cluster. We then manually review these candidates and group clusters into `direct_political`, `borderline`, and `non_political`.

Our current exported political-topic subset keeps all `direct_political` clusters plus selected borderline clusters. This yields `15,233` comments in `clustering/reddit_narratives_selected_political_comments.csv`, which serves as the current political-topic subset for downstream annotation and modeling.
