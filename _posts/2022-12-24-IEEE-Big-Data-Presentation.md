---
layout: post
title: "Robust Personalized Federated Learning under Demographic Fairness Heterogeneity"
date: 2022-12-24 17:00:00 +0300
description: Published in IEEE BigData 2022
---
In this post, I will present slides from my presentation at IEEE BigData held in Osaka, Japan this year (2022) alongside my verbal explanation. My work discusses one specific technique to allow individual clients in a federated machine learning setting to be able to independently choose which fairness metric they want to enforce. A quick summary (and the abstract from my paper) is the following: 

> Personalized federated learning (PFL) gives each client in a federation the power to obtain a model tailored to their specific data distribution or task without the client forfeiting the benefits of training in a federated manner. However, the concept of demographic group fairness has not been widely studied in PFL. Further, fairness heterogeneity -- when not all clients enforce the same local fairness metric -- has not been studied at all. To fill this gap, we propose \textit{Fair Hypernetworks} (FHN), a personalized federated learning architecture based on hypernetworks that is robust to statistical (e.g., non-IID and unbalanced data) and fairness heterogeneity. We theoretically show that granting clients the ability to independently choose multiple (possibly conflicting) fairness constraints, such as demographic parity or equalized odds, does not break previously proven generalization bounds on hypernetworks used in the federated setting. Additionally, we empirically test FHN against several baselines in multiple fair federated learning settings, and we find that FHN outperforms all other federated baselines when handling clients with heterogeneous fairness metrics. We further demonstrate the scalability of FHN to show that minimal degradation to the accuracy and the fairness of the clients occurs when the federation grows in size. Additionally, we empirically validate our theoretical analysis to show FHN generalizes well to new clients. To our knowledge, our FHN architecture is the first to consider tolerance to fairness heterogeneity which gives clients the freedom to personalize the fairness metric enforced during local training.

Federated learning was proposed in 2016 by McMan et al. who desired to train a high-quality centralized model without requiring the aggregation of the distributed clients’ private data. In their work, McMan et al. proposed Federated Averaging, which remains one of the most popular federated learning architectures today. In federated averaging, one round of training is completed as follows. First, the server selects a group of clients to participate in the training round. 

<p style="align-items:center; max-width: 500px;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/2.png" />
</p>

Then the server sends the current global weights to each of the selected clients. The clients then perform a round of local training using their local data and models.
  
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/3.png" />
</p>

After training is complete <click>, the clients send their update model weights back to the server which performs a weighted aggregation <click> based on the number of data points a user has.

<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/4.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/5.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/6.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/7.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/8.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/9.png" />
</p>
<p align="center">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/10.png" />
</p>
