---
layout: post
title: Introduction to Differential Privacy
date: 2022-09-11 17:00:00 +0300
description: A beginners guide to the inner workings of differential privacy.
tags: [differential privacy]
---

De-identification (otherwise known as anonymization or pseudonymization) is the process of removing identifying information from a dataset. _Identifying_ information can be thought of as any piece of data which can be used to identify a unique individual in the course of daily life. For example, names, addresses, phone numbers, e-mail addresses, and birth dates. For several years, de-identification was used as the primary privacy protection when using data for a variety of tasks. However, several studies have shown that de-identification is not really private at all, and in fact, when using auxilary information the re-identification of every individual in the dataset can be performed (an attack called a _linkage attack_).

_Differential privacy_ is a formal definition of privacy (i.e., it is possible to prove that a certain dataset is differentially private). However, instead of being a propoer of data (like de-identification is), differential privacy is a property of the algorithm. That is to say that in addition to proving that a certain dataset is differentially private (which requires showing that the algorithm which produced it satisfies differential privacy), we can additionally prove that an algorithm itself satisfies differential privacy. 

Differential privacy was throughoughly explained by Dr. Cynthia Dwork and Dr. Aaron Roth in [_The Algorithmic Foundations of Differential Privacy_ ](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) (while it was initially proposed in [_Calibrating Noise to Sensitivity in Private Data Analysis_](https://link.springer.com/chapter/10.1007/11681878_14)). Dwork and Roth note the following:

> _Differential privacy_ describes a promise, made by a data holder, or curator to a data subject: "You will not be affected, adversely or otherwise, by allowing your data to be used in any study or analysis, no matter what other studies, data sets, or information sources, are available." Differential privacy addresses the paradox of learning nothing about an individual while learning useful information about a population. It ensures that the same conclusions will be reached independent of whether any individual opts into or opts out of the data set. Specifically, _it ensures that any sequence of outputs is essentially equally likely to occur, independent of the presence or absence of any individual._
