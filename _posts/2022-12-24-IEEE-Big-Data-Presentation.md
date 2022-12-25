---
layout: post
title: "Robust Personalized Federated Learning under Demographic Fairness Heterogeneity"
date: 2022-12-24 17:00:00 +0300
description: Published in IEEE BigData 2022
---
In this post, I will present slides from my presentation at IEEE BigData held in Osaka, Japan this year (2022) alongside my verbal explanation. My work discusses one specific technique to allow individual clients in a federated machine learning setting to be able to independently choose which fairness metric they want to enforce. A quick summary (and the abstract from my paper) is the following: 

> Personalized federated learning (PFL) gives each client in a federation the power to obtain a model tailored to their specific data distribution or task without the client forfeiting the benefits of training in a federated manner. However, the concept of demographic group fairness has not been widely studied in PFL. Further, fairness heterogeneity -- when not all clients enforce the same local fairness metric -- has not been studied at all. To fill this gap, we propose \textit{Fair Hypernetworks} (FHN), a personalized federated learning architecture based on hypernetworks that is robust to statistical (e.g., non-IID and unbalanced data) and fairness heterogeneity. We theoretically show that granting clients the ability to independently choose multiple (possibly conflicting) fairness constraints, such as demographic parity or equalized odds, does not break previously proven generalization bounds on hypernetworks used in the federated setting. Additionally, we empirically test FHN against several baselines in multiple fair federated learning settings, and we find that FHN outperforms all other federated baselines when handling clients with heterogeneous fairness metrics. We further demonstrate the scalability of FHN to show that minimal degradation to the accuracy and the fairness of the clients occurs when the federation grows in size. Additionally, we empirically validate our theoretical analysis to show FHN generalizes well to new clients. To our knowledge, our FHN architecture is the first to consider tolerance to fairness heterogeneity which gives clients the freedom to personalize the fairness metric enforced during local training.

Federated learning was proposed in 2016 by McMahan et al. who desired to train a high-quality centralized model without requiring the aggregation of the distributed clients’ private data. In their work, McMahan et al. proposed Federated Averaging, which remains one of the most popular federated learning architectures today. In federated averaging, one round of training is completed as follows. First, the server selects a group of clients to participate in the training round. 

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/2.png" />
</p>

Then the server sends the current global weights to each of the selected clients. The clients then perform a round of local training using their local data and models.
  
<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/3.png" />
</p>

After training is complete the clients send their update model weights back to the server which performs a weighted aggregation based on the number of data points a user has.

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/4.png" />
</p>

Having more data points means that the clients update is considered more important and is weighed more heavily. The aggregated weights become the new global weights, and at the next round, the clients retrieve the global weights and perform another round of training. 
 
<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/6.png" />
</p>

In federated averaging, as well as most federated learning architectures, task heterogeneity is assumed meaning that each client must implement the same architecture and optimization function. Additionally, all clients having data from the same distribution is often assumed, which is not an assumption that holds in real life scenarios. These lead to several interesting areas of research such as dealing with non-IID data and crafting personalized models.

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/7.png" />
</p>

Personalized FL has been proposed to overcome task heterogeneity and increase robustness against non-IID data. Specifically, personalized federated learning gives each client in a federation the power to obtain a model tailored to their specific data distribution, or task, without the client forfeiting the benefits of training in a federated manner, such as overcoming data limitations and continual learning. Many different approaches to personalized federated learning have been proposed, such as user clustering and collaborating, transfer learning, and multi-task and meta learning. Additionally, hypernetworks have recently been proposed as a framework to solve the personalization problem in federated learning. 

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/8.png" />
</p>

A hypernetwork simply is a network that generates the parameters for another.  For instance, on the image above, the hypernetwork is represented as h and it has parameters phi. Like a neural network, it takes in a vector as input, but instead of providing an output of classification probabilities, it instead produces the weights for each layer of another network parametrized by theta. Hypernetworks are naturally suitable for PFL as they can learn a diverse set of personalized models by conditioning on the input vector.

In personalized federated learning via hypernetworks, the authors proposed pFedHN which uses a global hypernetwork model to generate the parameters for each of the clients' local models. In pFedHN each client has a unique embedding vector stored on the global server, which is passed as input to the hypernetwork to produce the client’s personalized model parameters. This embedding vector is learned during training and contain a representation of each client that helps the hypernetwork produce custom parameters suited to the specific client. 

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/9.png" />
</p>

During each round, the client asks the hypernetwork to generate and send them the new weights, performs a round of local training, and then sends the difference between the learned weights and the weights they were sent at the start of the round back to the hypernetwork, which is represent here by delta theta i.  The optimization function is almost identical to the normal optimization function used in personalized federated learning. Specifically, for each of the n clients, the goal is to train a local model f that minimizes the loss on their data distribution. The only difference between the generic personalized federated learning optimization function and pFedHN’s is the use of the global model to generate the local parameters which is underlined in green here. We use pFedHN as the underlying base federated learning architecture in FHN. 

In the centralized setting, fair machine learning often focuses on demographic fairness which strives to make machine learning models that treat individuals from different demographic groups based on features such as race or gender similarly. There are three main approaches for generating fair machine learning algorithms. 

<p style="display: block; margin: auto; width: 50%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/10.png" />
</p>

The first is pre-processing which involves changing the training data before feeding it into an ML algorithm. The second is in-processing which aims to improve the fairness of an algorithm by adding a constraint, or a regularization term, to the existing objective function. And the final is post-processing which as the name suggests performs post-processing of the output scores of the classifier to make decisions fairer. Each of these approaches require access to the sensitive variables of each data point, making them unsuitable to be directly applicable in the federated learning setting. Additionally, in the federated learning setting, most works based around fairness focus on client parity where the accuracy of each client is required to be similar but works focusing on demographic fairness are continually gaining popularity. 

