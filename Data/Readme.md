# EVO-Clust-VAE-WGAN-GP – Research Code
This repository contains the source code for **EVO-Clust-VAE-WGAN-GP**, our Evolutionary multi-objective generative framework.  
It enables **utility-, fairness-, and privacy-aware synthetic data generation**, to identify balanced trade-offs across datasets.

 ## Abstract

Configuration choices critically determine the behavior of generative models under competing objectives such as utility, fairness, and privacy. However, conventional tuning methods—heuristic rules and scalarized search—provide a narrow and potentially misleading view of achievable trade-offs. In fairness- and privacy-aware data generation, improving statistical fidelity can increase memorization risk or group disparities, while enforcing fairness or privacy often reduces realism, reflecting a non-convex trade-off landscape with multiple operating regimes. In this work, we introduce a framework for configuration design that couples a generative modeling pipeline with multi-objective evolutionary search to study the induced privacy–fairness–utility trade-off landscape. Rather than optimizing for a single solution, our method approximates the Pareto front over model configurations to enable systematic analysis of competing objectives. Our contribution is primarily methodological, focusing on exploration of high-dimensional configuration spaces rather than architectural innovation of the generative model itself. Results show that the utility–fairness–privacy front is strongly dataset-dependent, ranging from compact and continuous to fragmented and sparse, challenging the assumption that a single compromise suffices to characterize model behavior. Compared to heuristic baselines and scalarized stochastic search, our method reveals richer operating regimes and supports flexible decision-making for sensitive data generation.
## 🗂 Datasets

This work leverages multiple datasets, each with a dedicated tailored architecture:

- **HIV Dataset** – [HealthGymAi](https://healthgym.ai/antiviral-hiv/)
- **Heart Failure Clinical Records** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5Z89R)
- **Obesity Estimation Dataset** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5H31Z)
- **Regensburg Pediatric Appendicitis Dataset** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
- **Corporate Stress Dataset** – [Kaggle](https://www.kaggle.com/datasets/ankitpatel2100/corporate-stress-dataset-insights-into-workplace)


