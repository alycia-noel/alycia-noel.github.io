---
layout: post
title: "Robust Personalized Federated Learning under Demographic Fairness Heterogeneity"
date: 2022-12-24 17:00:00 +0300
description: Published in IEEE BigData 2022
use_math: true
tags: [federated learning, fairness, personalization, my work]
---
In this post, I will present slides from my presentation at IEEE BigData held in Osaka, Japan this year (2022) alongside my verbal explanation. A video of my presentation (that covers all of this material) can be found [here](https://youtu.be/f5gauh9AqLw). My work discusses one specific technique to allow individual clients in a federated machine learning setting to be able to independently choose which fairness metric they want to enforce. A quick summary (and the abstract from my paper) is the following: 

> Personalized federated learning (PFL) gives each client in a federation the power to obtain a model tailored to their specific data distribution or task without the client forfeiting the benefits of training in a federated manner. However, the concept of demographic group fairness has not been widely studied in PFL. Further, fairness heterogeneity -- when not all clients enforce the same local fairness metric -- has not been studied at all. To fill this gap, we propose _Fair Hypernetworks_ (FHN), a personalized federated learning architecture based on hypernetworks that is robust to statistical (e.g., non-IID and unbalanced data) and fairness heterogeneity. We theoretically show that granting clients the ability to independently choose multiple (possibly conflicting) fairness constraints, such as demographic parity or equalized odds, does not break previously proven generalization bounds on hypernetworks used in the federated setting. Additionally, we empirically test FHN against several baselines in multiple fair federated learning settings, and we find that FHN outperforms all other federated baselines when handling clients with heterogeneous fairness metrics. We further demonstrate the scalability of FHN to show that minimal degradation to the accuracy and the fairness of the clients occurs when the federation grows in size. Additionally, we empirically validate our theoretical analysis to show FHN generalizes well to new clients. To our knowledge, our FHN architecture is the first to consider tolerance to fairness heterogeneity which gives clients the freedom to personalize the fairness metric enforced during local training.

Federated learning was proposed in 2016 by McMahan et al. who desired to train a high-quality centralized model without requiring the aggregation of the distributed clients’ private data. In their work, McMahan et al. proposed Federated Averaging, which remains one of the most popular federated learning architectures today. In federated averaging, one round of training is completed as follows. First, the server selects a group of clients to participate in the training round. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/2.png" />
</p>

Then the server sends the current global weights to each of the selected clients. The clients then perform a round of local training using their local data and models.
  
<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/3.png" />
</p>

After training is complete the clients send their update model weights back to the server which performs a weighted aggregation based on the number of data points a user has.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/4.png" />
</p>

Having more data points means that the clients update is considered more important and is weighed more heavily. The aggregated weights become the new global weights, and at the next round, the clients retrieve the global weights and perform another round of training. 
 
<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/6.png" />
</p>

In federated averaging, as well as most federated learning architectures, task heterogeneity is assumed meaning that each client must implement the same architecture and optimization function. Additionally, all clients having data from the same distribution is often assumed, which is not an assumption that holds in real life scenarios. These lead to several interesting areas of research such as dealing with non-IID data and crafting personalized models.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/7.png" />
</p>

Personalized FL has been proposed to overcome task heterogeneity and increase robustness against non-IID data. Specifically, personalized federated learning gives each client in a federation the power to obtain a model tailored to their specific data distribution, or task, without the client forfeiting the benefits of training in a federated manner, such as overcoming data limitations and continual learning. Many different approaches to personalized federated learning have been proposed, such as user clustering and collaborating, transfer learning, and multi-task and meta learning. Additionally, hypernetworks have recently been proposed as a framework to solve the personalization problem in federated learning. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/8.png" />
</p>

A hypernetwork simply is a network that generates the parameters for another.  For instance, on the image above, the hypernetwork is represented as _h_ and it has parameters phi. Like a neural network, it takes in a vector as input, but instead of providing an output of classification probabilities, it instead produces the weights for each layer of another network parametrized by theta. Hypernetworks are naturally suitable for PFL as they can learn a diverse set of personalized models by conditioning on the input vector.

In personalized federated learning via hypernetworks, the authors proposed pFedHN which uses a global hypernetwork model to generate the parameters for each of the clients' local models. In pFedHN each client has a unique embedding vector stored on the global server, which is passed as input to the hypernetwork to produce the client’s personalized model parameters. This embedding vector is learned during training and contain a representation of each client that helps the hypernetwork produce custom parameters suited to the specific client. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/9.png" />
</p>

During each round, the client asks the hypernetwork to generate and send them the new weights, performs a round of local training, and then sends the difference between the learned weights and the weights they were sent at the start of the round back to the hypernetwork, which is represent here by _delta theta i_.  The optimization function is almost identical to the normal optimization function used in personalized federated learning. Specifically, for each of the n clients, the goal is to train a local model _f_ that minimizes the loss on their data distribution. The only difference between the generic personalized federated learning optimization function and pFedHN’s is the use of the global model to generate the local parameters which is underlined in green here. We use pFedHN as the underlying base federated learning architecture in FHN. 

In the centralized setting, fair machine learning often focuses on demographic fairness which strives to make machine learning models that treat individuals from different demographic groups based on features such as race or gender similarly. There are three main approaches for generating fair machine learning algorithms. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/10.png" />
</p>

The first is pre-processing which involves changing the training data before feeding it into an ML algorithm. The second is in-processing which aims to improve the fairness of an algorithm by adding a constraint, or a regularization term, to the existing objective function. And the final is post-processing which as the name suggests performs post-processing of the output scores of the classifier to make decisions fairer. Each of these approaches require access to the sensitive variables of each data point, making them unsuitable to be directly applicable in the federated learning setting. Additionally, in the federated learning setting, most works based around fairness focus on client parity where the accuracy of each client is required to be similar but works focusing on demographic fairness are continually gaining popularity. 

There are many definitions of fairness that have been proposed in fair machine learning research. The two we focus on in our work is demographic parity and equalized odds. Demographic parity requires that decisions are independent of the sensitive attribute. For instance, during hiring, demographic parity is achieved if applicants of all races have equal chances of being hired. On the other hand, equalized odds requires that decisions are independent of the sensitive attribute conditioned on the actual outcome. In the hiring example, equalized odds is satisfied if all qualified applicants have equal chances of getting the job regardless of race and if all unqualified applicants have an equal chance of getting the job regardless of race. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/11.png" />
</p>

It turns out that demographic parity and equalized odds cannot be achieved at the same time. This means that a machine learning algorithm can satisfy demographic parity or equalized odds. I won’t discuss the proof as to why, but a detailed explanation can be found in the work listed on the bottom of the screen. This fact plays into a main draw back of existing demographic fairness approaches in the federated learning setting which is that they require each user to enforce the same fairness metric and are often not personalizable. However, in our work, we focus on demographic fairness in the federated setting where each client can individually choose which fairness metric they want to enforce during in processing such as demographic parity or equalized odds which we term demographic fairness heterogeneity.  

In our approach to solving demographic fairness heterogeneity in the federated setting, we use an in-processing approach to enforce fairness on the local client models. Specifically, we use a constrained optimization problem based on a reduction approach presented in the paper “A reductions approach to fair classification” by Agarwal et al. In their work, the authors propose that fairness constraints like demographic parity and equalized odds, among other fairness metrics, can be described as a set of linear constraints. In other words, during the process of learning the optimal model, we are constrained such that we are below a certain fairness threshold _c_. Here, _M_ and _C_ are our fairness constraints where _M_ is a matrix and _C_ is a vector, and _mu_ is a vector of conditional moments. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/12.png" />
</p>

To solve the constrained optimization problem, the authors format the equation in saddle point form where lambda is a vector of Lagrangian multipliers for each constraint represented in the matrix _M_. Agarwal et al find the optimal solution to this equation by reducing it to a sequence of weighted classification problems that iteratively calls the model as a black box. This can get costly in a federated setting, so instead, we chose to solve the saddle point form by performing gradient ascent on the Lagrangian multiplier alongside gradient descent on the parameters during the local training phase. 
  
<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/13.png" />
</p>

Here, we list our main learning objective. While it is rather long, it can be broken up and explained through the optimization functions for pFedHN and for fairness as a linear constraint. First, consider the middle section of the equation. This is simply the optimization function for pFedHN. The remainder of the optimization function is simply the optimization function for enforcing fairness as a linear constraint. Essentially, we take the pFedHN optimization function and insert it into the fairness as a linear constraint optimization function.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/15.png" />
</p>

The overall training process can be seen as follows. First, we randomly initialize the embedding matrix which is held on the server.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/16.png" />
</p>

The server then selects one client.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/17.png" />
</p>

The the server uses the selected clients' embedding vector to generate their parameters, and then sends them to the corresponding client.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/18.png" />
</p>

The client then performs training by selecting a mini-batch of data.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/19.png" />
</p>

Using the mini-batch of data, the client performs local fairness enforced training to generate the new parameters.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/20.png" />
</p>

Finally, the client calculates the difference between their new weights, and the weights originally sent by the server, and sends the resulting value to the server. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/21.png" />
</p>

Using the client's update, the server then updates both the hypernetwork's parameters _phi_ as well as the embedding matrix _V_. This process continues for a total of _R_ rounds.

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/22.png" />
</p>

We used two different popular fairness datasets in our experimentation. The first is the adult dataset where the task is to predict if a person’s yearly income exceeds 50 thousand dollars. In order to simulate non-IID data, instead of randomly splitting the data between the clients, we split the data based on the type of employment. By this I mean that some clients only had records corresponding to government employees while other clients only had records corresponding to people who worked in the private sector. The second dataset is the COMPAS dataset where the task is to predict if a person will commit another crime withing a two-year period of their initial offence. To simulate non-IID data we split the data according to age. For times sake I will only discuss the results on the adult dataset and defer to our paper for the COMPAS results. Additionally, we chose to use gender as the sensitive attribute in all of our experiments. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/23.png" />
</p>

We compare FHN against three baselines. The first is decentralized fair learning where each user independently trains their own models in a fairness enforced manner. In other words, no collaboration occurs between the clients, but this setting has maximum privacy. The second is fair federated learning through federated averaging. In this method, we use the standard process of federated averaging, but additionally enforce the linear constraint for fairness on each client’s local model. Finally, we compare against FairFed which is an approach based on federated averaging which instead of weighting  clients based on the amount of data they have, instead adaptively modifies the aggregation weights based on how similar a clients fairness value is to the global model’s fairness value. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/24.png" />
</p>

We use four different settings in our experimentation. Specifically, we test the case where no fairness is enforced, the case where each client enforces demographic parity, the case where each client enforces equalized odds, and finally the main case where each client can choose between demographic parity and equalized odds. In addition to accuracy, we compare the architectures along equalized odds difference which and statistical parity difference. We used the first test as a baseline to see which architecture performs best without fairness being enforced. In the first three cases, FFLvFA obtained the highest accuracy and the architecture that obtained the best fairness result varied. In the last experiment, FFLvA once again had the highest accuracy, but the worst fairness values. DFL obtained the best fairness results for clients using equalized odds, mainly due to the fact that the results were exactly the same as the case of only training equalized odds since all clients trained separately. In the SPD case, FHN obtained the best overall fairness, and additionally obtained good accuracy and EOD results. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/25.png" />
</p>

In our second experiment we showed that increasing the federation size does not drastically increase the values of EOD and SPD or drastically decrease the accuracy. Specifically, we tested a range of 10 to 100 clients. In these graphs the sold line stands for the average value obtained, the color band shows the standard deviation, while the black dotted line shows the absolute maximum and minimum value obtained by any one client. While the absolute maximum and minimum increase as we introduced more clients, we still maintained an average accuracy of 80%, and EOD and SPD values close to zero which shows that FHN can scale well. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/26.png" />
</p>

For our final test, we showed that the FHN architecture was able to generalize to new clients who were not originally present during training. Specifically, we used a federation of 100 clients where 90 were included in training while the remaining 10 were. Held out to be introduced as novel clients. For each client, we draw data points according to a beta distribution where each client had varying amounts of data points, ranging for 100 to 4,000 data points. The same sampling was performed for the novel clients, but we varied the parameters of the beta distribution to simulate clients with non-IID data. The distance between a novel client’s distribution and its nearest neighbor’s distribution from the training set is reported as total variation. If the bar is above the x-axis, then the novel clients have a higher average value which is favorable for accuracy but not for EOD and SPD. Overall, there were not consistent trends as TV increased, but accuracy was always within .05, and EOD and SPD were always within .04. This shows that FHN can generalize well to new clients and novel clients achieve their desired fairness metric at almost the same degree as clients used in training. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/27.png" />
</p>

In conclusion, we proposed fair hypernetworks, which is an architecture robust to statistical and fairness heterogeneity that is based on the concept of using hypernetworks as the global model in personalized federated learning as well as using linear fairness constraints on the local model. We showed that FHN outperforms the baselines we tested against, is scalable to large federation sizes, and generalizes well to new clients, even when their data distribution is not the same as the training clients. 

<p style="display: block; margin: auto; width: 60%;">
  <img src="http://alycia-noel.github.io/assets/img/bigdata22/28.png" />
</p>

_This work was supported in part by NSF 1910284, 1946391, 2137335, and 2147375._

For more detailed analysis of our experiments and the theoretical underpinnings of this work, see the fully published paper (link coming soon). If you are a student without access to the IEEE archives, please feel free to email me and I can send a copy for free. 


