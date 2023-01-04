---
layout: post
title: Introduction to Differential Privacy
date: 2022-09-11 17:00:00 +0300
description: A beginners guide to the inner workings of differential privacy.
use_math: true
tags: [differential privacy]
---

De-identification (otherwise known as anonymization or pseudonymization) is the process of removing identifying information from a dataset. _Identifying_ information can be thought of as any piece of data which can be used to identify a unique individual in the course of daily life. For example, names, addresses, phone numbers, e-mail addresses, and birth dates. For several years, de-identification was used as the primary privacy protection when using data for a variety of tasks. However, several studies have shown that de-identification is not really private at all, and in fact, when using auxilary information the re-identification of every individual in the dataset can be performed (an attack called a _linkage attack_).

_Differential privacy_ is a formal definition of privacy (i.e., it is possible to prove that a certain dataset is differentially private). However, instead of being a propoer of data (like de-identification is), differential privacy is a property of the algorithm. That is to say that in addition to proving that a certain dataset is differentially private (which requires showing that the algorithm which produced it satisfies differential privacy), we can additionally prove that an algorithm itself satisfies differential privacy. 

Differential privacy was throughoughly explained by Dr. Cynthia Dwork and Dr. Aaron Roth in [_The Algorithmic Foundations of Differential Privacy_ ](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) (while it was initially proposed in [_Calibrating Noise to Sensitivity in Private Data Analysis_](https://link.springer.com/chapter/10.1007/11681878_14)). Dwork and Roth note the following:

> _Differential privacy_ describes a promise, made by a data holder, or curator to a data subject: "You will not be affected, adversely or otherwise, by allowing your data to be used in any study or analysis, no matter what other studies, data sets, or information sources, are available." Differential privacy addresses the paradox of learning nothing about an individual while learning useful information about a population. It ensures that the same conclusions will be reached independent of whether any individual opts into or opts out of the data set. Specifically, _it ensures that any sequence of outputs is essentially equally likely to occur, independent of the presence or absence of any individual._

### :information_source: **Definition**
A function which satisfies differential privacy is often called a _mechanism_. We say that a mechanism $F$ satisfies differential privacy if for all _neighboring_ datasets $x$ and $x'$, and all possible outputs $S$, 

<p align="center">
  $\frac{\text{Pr}[F(x) = S]}{\text{Pr}[F(x')=S]} \leq e^\epsilon$
</p>

Two datasets are considred neighbors if they are exactly the same, minus the record of a single individual. This single record could exist in one dataset, and not the other. Or, this record could exist in both dataset, but the data that makes up the record could be different. The important implication of this definition is that F's output will be the same (to a degree), with or without the data of any specific individual. The randomness build into F should be enough so that an observed output from F will not reveal which (x or x') was the input. 

The $\epsilon$ parameter in the definition is called the _privacy parameter_ or the _privacy budget_. $\epsilon$ provides a knob to tune the "amount of privacy" the definition provides. Small values of $\epsilon$ require F to provide very similar outputs when given similar inputs, and therefore provide higher levels of privacy; large values of $\epsilon$ allow less similarity in the outputs, and therefore provide less privacy. Unfortunately, setting $\epsilon$ to prevent privacy leakage is somewhat of an art and there is not one defined way to set $\epsilon$ for every situation. The general consensus is that $\epsilon$ should be around 1 or smaller, and values of $\epsilon$ above 10 probably don't do much to protect privacy - but this rule of thumb could turn out to be too conservative. 

The easiest way to achieve differential privacy (in most situations) is to add random noise to its answer. They key challenge is to add enough noise to satisfy the definition of differential privacy, but not so much noise that the answer becomes unuseful. To make this noise addition process easier, several mechanisms have been proposed that describe exactly what kind of, and how much, noise to add. The most popular are _the Laplace mechanism_, _the Gaussian mechanism_, and _the exponential mechanism_.

### The Laplace Mechanism (and Sensitivity)
#### :information_source: **Definition**
According to the Laplace mechanism, for a function $f(x)$ which returns a number, the following definition of $F(x)$ satisfies $\epsilon$-differential privacy:

<p align="center">
    $F(x) = f(x) + Lap\big(\frac{s}{\epsilon}\big)$
</p>

where $s$ is the sensitivity of $f$, and $Lap\big(\frac{s}{\epsilon}\big)$ denotes sampling from the Laplace distribution with center 0 and scale $\frac{s}{\epsilon}$.

The $L_1$ _sensitivity_ of a function $f$ is the amount $f$'s output changes when its input changes by 1. For example,

* The sensitivity of $f(x) = x$ is 1, since changing $x$ by 1 changes $f(x)$ by 1
* The sensitivity of $f(x) = x + x$ is 2, since changing $x$ by 1 changes $f(x)$ by 2
* The sensitivity of $f(x) = 10 * x$ is 10, since changing $x$ by 1 changes $f(x)$ by 10
* The sensitivity of $f(x) = x * x$ is unbounded, since the change in $f(x)$ depends on the value $x$

Some functions have pre-defined sensitivities:

| Function | Sensitivity |
| -------- | ----------- |
| Counting | 1           |
| Histogram Query | 1 | 
| Summation (no upper/lower bounds)| unbounded|
| Summation (upper/lower bounds) | upper - lower | 

But, for most functions, the sensitivity has to be estimated and possibly clipped so as to not be unbounded as queries with unbounded sensitivity cannot be directly answered with differential privacy. To do so, we can use the following equation:

<p align="center">
  $\Delta f = \max||f(x) - f(y)||$
</p>

which captures the magnitude by which a single indvidual's data can change the function $f$ _in the worst case_.


### The Gaussian Mechanism
The Gaussian mechanism is an alternative to the Laplace mechanism and instead of adding Laplacing noise, adds noise drawn from a Gaussian distribution. The Gaussian mechanism does not satisfy pure $\epsilon$-differential privacy, but instead satisfies $(\epsilon, \delta)$-differential privacy. 

:information_source: **Definition**

Approximate differential privacy, also called $(\epsilon, \delta)$-differential privacy, has a similar definition to regular $\epsilon$-differential privacy.

<p align="center">
  $\frac{\text{Pr}[F(x) = S]}{\text{Pr}[F(x')=S]} \leq e^\epsilon + \delta$
</p>

where $\delta$ is the failure probability. In other words, with probability $1 - \delta$ we will get the same guarantee as pure differential privacy, but with probability $\delta$ we get no guarantee. 

According to the Gaussian mechanism, for a function $f(x)$ which returns a number, the following definition of $F(x)$ satisfies $(\epsilon, \delta)$-differential privacy:

<p align="center">
  $F(x) = f(x) + \mathcal{N}(\sigma^2)$
</p>

where $\sigma^2=\frac{2s^2\log{(1.25/\delta)}}{\epsilon^2}$, $s$ is the $L_2$ sensitivity (a.k.a, euclidean distance), and $\mathcal{N}(\sigma^2)$ denotes sampling from the Gaussian (normal) distribution with center 0 and variance $\sigma^2$. For real-valued functions, we can use the Gaussian mechanism in exactly the same way as we do the Laplace mechanism.

### The Exponential Mechanism
The Laplace and Guassian mechanisms are focused on numerical answers, and add noise directly to the answer itself. When we want to return a precise answer with no added noise while still preserving privacy we can use the exponential mechanism. The exponential mexhanism allows selecting the best element from a set while preserving differential privacy. The analyst defines which element is the best by specifying a scoring function that outputs a score for each element in the set, and also defines the set of things to pick from. The mexhanism provides differential privacy by approximately maximizing the score of the element it returns. In other words, to satisfy differential privacy, the exponential mexhanism sometimes returns and element from the set which does not have the highest score. For a more detailed discussion of the exponential mechanism, see [this post](https://programming-dp.com/ch9.html).

## Properties of Differential Privacy
There are three important properties of differentially private mechanisms that arise from the definition of differential privacy: sequential compopsition, parallel composition, and post processing. These properties help us to design useful algorithms that satisfy differential privacy and ensure that those algorithms provide accurate answers.

### Sequential Composition
Sequential composition bounds the total privacy cost of releasing multiple results of differentially private mechanisms on the same input data. Formally, the sequential composition theorem for differential privacy says that:

> If $F_1(x)$ satisfies $\epsilon_1$-differential privacy and $F_2(x)$ satisfies $\epsilon_2$-differential privacy then the mechanism $G(x) = (F_1(x), F_2(x))$ which releases both results satisfies $\epsilon_1 + \epsilon_2$ differential privacy.

Sequential composition is a vital property of differential privacy because it enables the design of algorithms that consult the data more than once. Sequential composition is also important when multiple separate analyses are performed on a single dataset, since it allows individuals to bound the total privacy cost they incur by participating in all of these analyses. The bound on privacy cost given by sequential composition is an upper bound - the actual privacy cost of two particular differentially private releases may be smaller than this, but never larger. 

### Parallel Composition
Parallel composition can be seen as an alternative to sequential composition. It is a second way to calculate a bound on the total privacy cost of multiple data releases. Parallel composition is based on the idea of splitting your dataset into disjoint chunks and running a differentially private mechanism on each chunk separately. Since the chunks are disjoint, each individual's data appears in exactly one chunk - so even if there are $k$ chunks in total (and therefore $k$ runs of the mechanism), the mexhanism runs exactly once on the data of each individual. Formally,

> If $F(x)$ satisfies $\epsilon$-differential privacy and we split a dataset $X$ into $k$ disjoint chunks such that $x_1\cup\dots\cup x_k = X$, then the mechanism which releases all of the results $F(x_1), \dots, F(x_k)$ satisfies $\epsilon$-differential privacy. 

Parallel composition gives a much better ound than sequential composition gives. Since we run $F$ $k$ times, sequential composition would say that this procedure satisfies $k\epsilon$-differential privacy. Parallel composition allows us to say that th etotal privacy cost is just $\epsilon$.

### Post-Processing
The main idea behind postprocessing is that it is impossible to reverse the privacy protection provided by differential privacy by post-processing the data in some way. Formally, 

> If $F(X)$ satisfies $\epsilon$-differential privacy, then for any (deterministic or randomized) function $g$, $g(F(X))$ satisfies $\epsilon$-differential privacy.

This means that it is always safe to perform arbitrary computations on the output of a differentially private mechanism as there is no danger of reversing the privacy protection the mechanism has provided. In particular, it's fine to perform post-processing that might reduce the noise of the mechanism's output (e.g., replacing negative results with zeros, for queries that shouldn't return negative results). In fact, many sophisticated differentially private algorithms make use of post-processing to reduce noise and improve the accuracy of their results. 

The other implication of the post-processing property is that differential privacy provides resistance against privacy attacks based on auxiliary information. For example, the function $g$ might contain auxiliary information about elements of the dataset, and attempt to perform a linkage attack using this information. The post-processing property says that such an attack is limited in its effectiveness by the privacy parameter $\epsilon$, regardless of the auxiliary information contained in $g$.
