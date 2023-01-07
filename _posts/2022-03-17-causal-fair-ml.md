---
layout: post
title: Causal Fair Machine Learning Methods
date: 2022-03-17 17:00:00 +0300
description: An introduction to sevearl different types of causal fair machine learning methods.
tags: [fairness, machine learning]
---
Most recent fairness notions are causality-based and reflect the now widely accepted idea that using causality is necessary to appropriately address the problem of fairness. 
%This is because statistical-based machine learning criteria are observational as they depend only on the joint distribution of predictor (algorithm), marginalization attribute, features, and the final outcome. Observational criteria have severe inherent limitations that prevent them from resolving matters of fairness in a conclusive manner, mainly because fairness cannot be well assessed based only on mere correlation or association (rung one of Pearl's Ladder of Causation). Additionally, statistical-based fairness metrics do not align well with the legal process of discrimination handling since discrimination claims usually require plaintiffs to demonstrate a causal connection between the challenged decision and the marginalization feature. 
Causality-based fairness notions differ from the statistical ones in that they are not totally based on data\footnote{``Data is profoundly dumb. Data can tell you that people who took a medicine recovered faster than those who did not take it, but they can't tell you why." - Judea Pearl \citep{PearlMackenzie18}}, but consider additional knowledge about the structure of the world, in the form of a causal model. 
Causality-based fairness notions are developed mainly under two causal frameworks: the structural causal model (SCMs) and the potential outcome. SCMs assume that we know the complete causal graph, and hence, we are able to study the causal effect of any variable along many different paths. The potential outcome framework does not assume the availability of the causal graph and instead focuses on estimating the causal effects of treatment variables. In Table \ref{tab:classification}, we present the causal framework to which each causality-based fairness notion discussed in this section belongs. In this section, we we begin by giving a short insight and overview of causality-based fairness notions, followed by a brief intermission to introduce two important statistical-fairness definitions, and then we spend the remainder of the section introducing the casual-based fairness notions, minus the last section where we state the main technical pitfalls experienced by these types of metrics.
\begin{figure}[t!]
\centering
    \includegraphics[width=.7\textwidth]{ladderofcausation.jpg}
    \caption{Pearl's Ladder of Causation. The first rung, associations, only allows predictions based on passive observations. The second rung, interventions, not only relies on seeing, but also changing what is. Rung three, counterfactuals, deals with the imaginary, or what might have been.}
\label{fig:ladderofcause}
\end{figure}

In \citep{pearl2019seven}, Pearl presented the causal hierarchy through the Ladder of Causation, as shown in Fig. \ref{fig:ladderofcause}. The Ladder of Causation has the 3 rungs: association, intervention, and counterfactual. The first rung, associations, can be inferred directly from the observed data using conditional probabilities and conditional expectations (i.e. a probabilistic theory, see Section \ref{sec:philo}). The intervention rung involves not only seeing what is, but also changing what we see. Interventional questions deal with $P(y\mid do(x), z)$ which stands for ``the probability of $Y=y$, given that we intervene and set the values of $X$ to $x$ and subsequently observe event $Z=z$.'' Interventional questions cannot be answered from pure observational data alone. They can be estimated experimentally from randomized trials or analytically using causal Bayesian networks. The top rung invokes counterfactuals and deals with $P(y_x\mid x', y')$ which stands for ``the probability that event $Y=y$ would be observed had $X$ been $x$, given that we actually observed $X$ to be $x'$ and $Y$ to be $y'$.'' Such questions can be computed only when the model is based on functional relations or is structural. In Table \ref{tab:classification}, we also show the causal hierarchical level that each causality-based fairness notion aligns with. 

In the context of fair machine learning, we use $S \in \{s^+, s^-\}$ to denote the marginalization attribute, $Y \in \{y^+, y^-\}$ to denote the decision, and $\X$ to denote a set of non-marginalization attributes. The underlying mechanism of the population over the space $S\times \X \times Y$ is represented by a causal model $\mathcal{M}$, which is associated with a causal graph $\CG$. Fig. \ref{fig:cgexp} shows a causal graph that will be used to illustrate fairness notions throughout this section. With $\mathcal{M}$, we want to reason about counterfactual queries, e.g., ``what would the prediction have been for this individual if their marginalization attribute value changed?'' A historical dataset $\D$ is drawn from the population, which is used to construct a predictor $ h: \X, S \rightarrow \Y $. Note that the input of the predictor can be a subset of $\X, S$ and we use $\widehat{\Pa{}}$ to denote the set of input features of the predictor when introducing counterfactual error rate in Section \ref{sec:cer}. The causal model for the population over space $S\times \X \times \Y$ can be considered the same as $\mathcal{M}$, except that the function $f_{Y}$ is replaced with a predictor $h$. Most fairness notions involve either $Y$ or $\hat{Y}$ in their counterfactual quantity and, roughly speaking, they correspond to statistical parity (a statistical-based notion introduced below). A few fairness notions, e.g., counterfactual direct error rate \citep{zhang2018equality}, correspond to the concept of equalized odds (also explained below) and involve both $Y$ and $\hat{Y}$ in their counterfactual quantity. We also mark if a notion uses $Y$ and/or $\hat{Y}$ in Table \ref{tab:classification}.
\begin{figure}[t!]
    \centering
    \includegraphics[width=.45\textwidth]{berkelyexample.jpg}
    \caption{Causal graph of the college admission example used throughout this section. Let gender be the marginalization attribute and Female be the marginalized class. Note: for simplicity, we consider gender to be binary, but we recognize that this is not the case in real life.}
    \label{fig:cgexp}
\end{figure}
We note that for all of the fairness notions presented here, there actually exists two versions -- strict and relaxed. The strict version means there is absolutely no discrimination effect (i.e., no wiggle room), whereas the relaxed version often compares the causal effect with $\tau$, a user-defined threshold for discrimination (i.e., wiggle room). Despite having two approaches, for simplicity, we adhere to the strict version when introducing each fairness notion in the discussion below. 
%\xnote{Do you think whether it is better to move Section 5 right after Section 2? I see some discussions of causal fairness notions in Section 4. }

%\xnote{We also need to discuss where we put statistical-based fairness notions.}
\subsection{Statistical-based Fairness Notions}
Despite the claims we have made against using statistical-based fairness notions so far, we do wish to introduce two popular metrics: statistical parity and equalized odds. Our reasoning of doing so is two-fold: 1) these two statistical notions are closely tied to several causality-based fairness notions, and 2) they present a clear picture of why causality-based machine learning fairness notions are preferred over statistical ones. 

We will begin by describing statistical parity, which also goes by the names demographic parity and group fairness. As the name implies, it requires that there is an equal probability for both individuals in the marginalized and non-marginalized groups to be assigned to the positive class \citep{dwork_fairness_2011, kusner2017counterfactual}. Notationally, group fairness can be written as: 
\begin{equation}
    P(\hat{Y} = 1 \mid  S = 0) = P(\hat{Y} = 1 \mid  S  = 1)
\end{equation}
where $\hat{Y}$ is the predicted outcome and $S$ is the marginalization variable.

Barocas, Hardt, and Narayanan note that while statistical parity aligns well with how humans reason about fairness, several draw-backs exists \citep{barocas-hardt-narayanan}. Namely, that it ignores any correlation between the marginalization attributes and the target variable $Y$ which constrains the construction of a perfect prediction model. Additionally, it enables laziness. In other words, it allows situations where qualified people are carefully selected for one group (e.g., non-marginalized), while random people are selected for the other (marginalized). Further, it allows the trade of false negatives for false positives, meaning that neither of these rates are considered more important, which is false in many circumstances \citep{barocas-nips-2017}.

The fairness metric of equalized odds is also known as conditional procedure accuracy equality and disparate mistreatment. Whereas statistical parity requires that the probability of being classified as positive is the same for all groups, equalized odds requires that true and false positive rates are similar across different groups \citep{moritz_google_price_srebro}. In other words, equalized odds enforces equality among individuals who have similar outcomes. It can be written as:
\begin{equation} P(\hat{Y} = 1 \mid  Y = y \cap S = 0) = P(\hat{Y} = 1 \mid  Y = y \cap S = 1) \;\; \text{ for } \;\; y\in\{0,1\} \end{equation}
where $\hat{Y}$ is the predicted outcome, $Y$ is the actual outcome, and $S$ is the marginalization attribute.

%\subsection{Causality-based Fairness Notions}
%Since the majority of causality-based fairness notions are defined in terms of the non-observable quantities of interventions and counterfactuals, their applicability depends heavily on the identifiability of those quantities from observational data. We refer readers who are interested in learning the specifics of identifiability theory and criteria, and how they can be used to decide the applicability of causality-based fairness metrics to \citep{makhlouf2021survey}. In this section, we simply present causality-based fairness notions and discuss their relationships. 

\subsection{Total, Natural Direct, and Natural Indirect Causal Fairness}
\label{sec:cf}
We now move into our main discussion of the causality-based fairness notions, starting with a discussion of total, natural direct, and natural indirect causal fairness.
Discrimination can be viewed as the causal effect of $S$ on $Y$. Total causal fairness answers the question of if the marginalization attribute $S$ changed (e.g., changing from marginalized group $s^{-}$ to non-marginalized group $s^{+}$), how would the outcome $Y$ change on average? A straightforward strategy to answer this question is to measure the average causal effect of $S$ on $Y$ when $S$ changes from $s^{-}$ to $s^{+}$, an approach called total causal fairness.

\begin{definition} [Total Causal Fairness]\label{def:tcf}
Given the marginalization attribute $S$ and decision $Y$, we achieve total causal fairness if: \begin{equation}\TCE(s_1, s_0) = P(y_{s_1}) - P(y_{s_0}) = 0\end{equation} where $s_1, s_0 \in \{ s^+, s^-\}$.
\end{definition}
For instance, based on Fig. \ref{fig:cgexp}, TCE would report the average causal effect that being Female had on a student's outcome of admission. 

Additionally, the causal effect of $S$ on $Y$ does not only include the direct discriminatory effect, but it also includes the indirect discriminatory effect and the explainable effect. In \citep{pearl2013direct}, Pearl proposed the use of NDE and NIE to measure the direct and indirect discrimination. Recall from Definitions \ref{def:nde}, \ref{def:ide} that $\mathrm{NDE}(s_1, s_0) = P(y_{s_1, \mathbf{Z}_{s_0}}) - P(y_{s_0})$ and $\mathrm{NIE}(s_1, s_0) = P(y_{s_0, \mathbf{Z}_{s_1}}) - P(y_{s_0})$ where $\mathbf{Z}$ is the set of mediator variables. When applied to the example in Fig. \ref{fig:cgexp}, the mediator variable could be the major. 
%\xnote{If we use em font here for each attribute, we need to be consistent with attributes used in previous figures. I guess we do not need to use em font.}
$P(y_{s_1, \mathbf{Z}_{s_0}})$ in NDE is the probability of $Y=y$ had $S$ been $s_1$ and had $\mathbf{Z}$ been the value it would naturally take if $S=s_0$. In other words, based on the example, $P(y_{s_1, \mathbf{Z}_{s_0}})$ would be the probability of being admitted when changing the gender to be Male while keeping the major the same. Similarly, NIE measures the indirect effect of $S$ on $Y$. However, NIE does not distinguish between explainable and indirect discrimination. 

\subsection{Path-Specific Causal Fairness}
\label{sec:pscf}
In \citep{zhang2017causal}, Zhang et al. introduced path-specific causal fairness based on the path-specific causal effect \citep{pearl2009causality} notion presented in Definition \ref{def:psce}. Different from total, natural direct, and natural indirect causal effects, the path-specific causal effect is based on graph properties of the causal graph (where the others were based on probabilities), and characterizes the causal effect in term of specific paths. 

\begin{definition} [Path-Specific Causal Fairness]\label{def:pscf}
Given the marginalization attribute $S$, decision $Y$, and \textit{redlining attributes} $\mathbf{R}$  (i.e., a set of attributes in $\mathbf{X}$ that cannot be legally justified if used in decision-making), define $\pi_{d}$ as the path set that contains some paths from $S$ to $Y$. We achieve path-specific causal fairness if: 
\begin{equation}\mathrm{PE}_{\pi}(s_1,s_0) = P(y_{s_1 \vert \pi, s_0 \vert \overline{\pi}}) - P(s_{x_0}) =0\end{equation}
where $s_1, s_0 \in \{ s^+, s^-\}$. Specifically, define $\pi_{d}$ as the path set that contains only $S\rightarrow Y$ and define $\pi_{i}$ as the path set that contains all the causal paths from $S$ to $Y$ which pass through some redlining attributes of $\mathbf{R}$. We achieve direct causal fairness if $\mathrm{PE}_{\pi_{d}}(s_1,s_0)=0$, and indirect causal fairness if $\mathrm{PE}_{\pi_{i}}(s_1,s_0)=0$.
\end{definition}

Direct discrimination considers the causal effect transmitted along the direct path from $S$ to $Y$, i.e., $S\rightarrow Y$.
The physical meaning of $\mathit{PE}_{\pi_{d}}(s_1,s_0)$ can be explained as the expected change in decisions of individuals from marginalized group $s_0$, if the decision makers are told that these individuals were from the non-marginalized group $s_1$. When applied to the example in Fig. \ref{fig:cgexp}, it means that the expected change in admission of applicants is actually from the marginalized group (e.g., Female), when the admission office is instructed to treat the applicants as from the non-marginalized group (e.g., Male). 

Indirect discrimination considers the causal effect transmitted along all the indirect paths from $S$ to $Y$ that contain the redlining attributes. The physical meaning of $\mathit{PE}_{\pi_{i}}(s_1,s_0)$ is the expected change in decisions of individuals from marginalized group $s_0$, if the values of the redlining attributes in the profiles of these individuals were changed as if they were from the non-marginalized group $s_1$. When applied to the example in Fig. \ref{fig:cgexp}, it means the expected change in admission of the marginalized group if they had the same gender makeups shown in the major as the non-marginalized group. 

The following propositions \citep{zhang2017causal} further show two properties of the path-specific effect metrics.

\begin{proposition}
If path set $\pi$ contains all causal paths from $S$ to $Y$ and $S$ has no parent in $\mathcal{G}$, then we have:
\begin{equation}
\mathrm{PE}_{\pi}(s_1,s_0) = \mathrm{TCE}(s_1,s_0) = P(y^{+}\mid s_1)-P(y^{+}\mid s_0).
\end{equation}
\end{proposition}

$P(y^{+}\mid s_1)-P(y^{+}\mid s_0)$ is known as the \emph{risk difference} (a measure of statistical parity). Therefore, the path-specific effect metrics can be considered as an extension to the risk difference (and statistical parity) for explicitly distinguishing the discriminatory effects of direct and indirect discrimination from the total causal effect.

\begin{proposition}
For any path sets $\pi_{d}$ and $\pi_{i}$, we do not necessarily have: \begin{equation}\mathrm{PE}_{\pi_{d}}(s_1,s_0)+\mathrm{PE}_{\pi_{i}}(s_1,s_0)=\mathrm{PE}_{\pi_{d}\cup \pi_{i}}(s_1,s_0).\end{equation}
\end{proposition}
This implies that there might not be a linear connection between direct and indirect discrimination.

\subsection{Counterfactual Fairness}
In Section \ref{sec:cf} and \ref{sec:pscf}, the intervention is performed on the whole population. These metrics deal with effects on an entire population, or on the average individual from a population. But, up to this point we have not talked about ``personalized causation" -- or causation at the level of particular events of individuals \citep{PearlMackenzie18}. Counterfactuals will allow us to do so. If we infer the post-intervention distribution while conditioning on certain individuals, or groups specified by a subset of observed variables, the inferred quantity will involve two worlds simultaneously: the real world represented by causal model $\mathcal{M},$ as well as the counterfactual world $\mathcal{M}_x$. Such causal inference problems are called counterfactual inference, and the distribution of $Y_{x}$ conditioning on the real world observation $\mathbf{O}=\mathbf{o}$ is denoted by $P(y_{x}\mid  \mathbf{o})$.

In \citep{kusner2017counterfactual}, Kusner et al. defined counterfactual fairness to be the case where the outcome would have remained the same had the marginalization attribute of an individual or a group been different, and all other attributes been equal. 

\begin{definition}[Counterfactual Fairness]\label{def:cf}
	Given a factual condition $\VO = \vo$ where $\VO \subseteq \{S, \X, Y \}$, we achieve counterfactual fairness if:
	\begin{equation}\CE(s_1, s_0\mid  \vo)  = P(y_{s_1} \mid  \mathbf{o}) - P(y_{s_0} \mid  \mathbf{o}) =0\end{equation}
	where $s_1, s_0 \in \{ s^+, s^-\}$.
\end{definition}

Note that we can simply define a classifier as counterfactually fair by replacing outcome $Y$ with the predictor $\hat{Y}$ in the above equation. The meaning of counterfactual fairness can be interpreted as follows when applied to the example in Fig. \ref{fig:cgexp}. Applicants are applying for admission and a predictive model is used to make the decision $\Y$. We concern ourselves with an individual from marginalized group $s_0$ who is specified by a profile $\vo$. The probability of the individual to get a positive decision is $P(\y\mid s_0,\vo)$, which is equivalent to $P(\y_{s_0}\mid s_0,\vo)$ since the intervention makes no change to $S$'s value of that individual. Now assume the value of $S$ for the individual had been changed from $s_0$ to $s_1$. The probability of the individual to get a positive decision after the hypothetical change is given by $P(\y_{s_1}\mid s_0, \vo)$. Therefore, if the two probabilities $P(\y_{s_0}\mid s_0, \vo)$ and $P(\y_{s_1}\mid s_0, \vo)$ are identical, we can claim the individual is treated fairly as if they had been from the other group.

\subsection{Counterfactual Effects}
\label{sec:ce}

In \citep{zhang2018fairness}, Zhang and Bareinboim introduced three fine-grained measures of the transmission of change from stimulus to effect called the counterfactual direct, indirect, and spurious effects. Throughout Section \ref{sec:ce}, we use $\mathbf{W}$ to denote all the observed intermediate variables between $S$ and $Y$ and use the group with $S=s_0$ as the baseline to measure changes of the outcome. 

\begin{definition}[Counterfactual Direct Effect]
	\label{def:ctf-de}
	Given a SCM, the counterfactual direct effect (Ctf-DE) of intervention $S=s_1$ on $Y$ (with baseline $s_0$) conditioned on $S=s$ is defined as:  \begin{equation}\textrm{Ctf-DE}_{s_0,s_1}(y\mid s) = P(y_{s_1,\mathbf{W_{s_0}}}\mid s) - P(y_{s_0}\mid s).\end{equation}
\end{definition}

$Y_{s_1,\mathbf{W}_{s_0}} = y\mid S = s$ is a more involved counterfactual compared to NDE and can be read as ``the value $Y$ would be had $S$ been $s_1$, while $\mathbf{W}$ is kept at the same value that it would have attained had $S$ been $s_0$, given that $S$ was actually equal to $s$.'' In terms of Fig. \ref{fig:cgexp}, $Y_{s_1,\mathbf{W}_{s_0}} = y\mid S = s$  means the admission decision for a Female student if they had actually been Male, while keeping all intermediate variables the same, when given that the student's gender is actually $s$ (meaning Male or Female). 

\begin{definition}[Counterfactual Indirect Effect]
	\label{def:ctf-ie}
	Given a SCM, the counterfactual indirect effect (Ctf-IE) of intervention $S=s_1$ on $Y$ (with baseline $s_0$) conditioned on $S=s$ is defined as:  \begin{equation}\textrm{Ctf-IE}_{s_0,s_1}(y\mid s) = P(y_{s_0,\mathbf{W}_{s_1}}\mid s) - P(y_{s_0}\mid s).\end{equation} 
\end{definition}

Ctf-IE measures changes in the probability of the outcome $Y$ being $y$ had $S$ been $s_0$, while changing $\mathbf{W}$ to whatever level it would have naturally obtained had $S$ been $s_1$, in particular, for the individuals in which $S=s_0$. In terms of Fig. \ref{fig:cgexp}, this means the probability of admission for a Female student based on the intermediate variable values that would be obtained if they were Male (e.g., ratio of Males applying to the major).  

\begin{definition}[Counterfactual Spurious Effect ]
	\label{def:ctf-se}
	Given a SCM, the counterfactual spurious effect (Ctf-SE) of $S=s_1$ on $Y=y$ (with baseline $s_0$) is defined as: \begin{equation}\textrm{Ctf-SE}_{s_0,s_1}(y) = P(y_{s_0}\mid s_1) - P(y\mid {s_0}).\end{equation} 
\end{definition}

$\text{Ctf-SE}_{s_0,s_1}(y)$ measures the difference in the outcome $Y=y$ had $S$ been $s_0$ for the individuals that would naturally choose $S$ to be $s_0$ versus $s_1$. In other words, it measures the difference in the admission decision had the marginalization attribute been set to Female for the students that were actually Female versus Male.

\begin{proposition}
For a SCM, if $S$ has no direct (indirect) causal path connecting $Y$ in the causal graph, then $\textrm{Ctf-DE}_{s_0,s_1}(y\mid s)=0$ ($\textrm{Ctf-IE}_{s_0,s_1}(y\mid s)=0$) for any $s$, $y$; if $S$ has no back-door\footnote{A backdoor path from $X$ to $Y$ is any path starting at $X$ with a backward edge $\leftarrow$ into $X$ such as: $X \leftarrow A \rightarrow B \leftarrow C \rightarrow Y$. Backdoor paths allow information to flow from $X$ to $Y$ in a way that is not causal.} path connecting $Y$ in the causal graph, then $\textrm{Ctf-SE}_{s_0,s_1}(y) = 0$ for any $y$. 
\end{proposition}

Building on these measures, Zhang and Bareinboim derived the causal explanation formula for the disparities observed in the total variation. Recall that the total variation is simply the difference between the conditional distributions of $Y$ when observing $S$ changing from $s_0$ to $s_1$. 

\begin{definition}[Total Variation] \label{def:tv}
	The total variation (TV) of $S=s_1$ on $Y=y$ (with baseline $s_0$) is given by:
	\begin{equation}\mathrm{TV}_{s_0, s_1}(y) = P(y\mid s_1) - P(y\mid s_0). \end{equation}
\end{definition}
In regard to Fig. \ref{fig:cgexp}, the TV would be the probability of the outcome given that the student was Male minus the probability of the outcome given that the student was Female., i.e., the difference in their overall probabilites of being admitted. %\xnote{check the use of I.e.}

\begin{theorem}[Causal Explanation Formula] \label{th:expf}
For any $s_0$, $s_1$, $y$, the total variation, counterfactual spurious, direct, and indirect effects obey the following relationship: 
 \begin{equation}\mathrm{TV}_{s_0,s_1}(y) = \textrm{Ctf-SE}_{s_0,s_1}(y) + \textrm{Ctf-IE}_{s_0,s_1}(y\mid s_1) - \textrm{Ctf-DE}_{s_1,s_0}(y\mid s_1),\end{equation} 
 \begin{equation}\mathrm{TV}_{s_0,s_1}(y) = \textrm{Ctf-DE}_{s_0,s_1}(y\mid s_0) - \textrm{Ctf-SE}_{s_1,s_0}(y) - \textrm{Ctf-IE}_{s_1,s_0}(y\mid s_0).\end{equation}
 \end{theorem}

Theorem \ref{th:expf} allows the machine learning designer to quantitatively evaluate fairness and explain the total observed disparity of a decision through different discriminatory mechanisms. For example, the first formula shows that the total disparity experienced by the individuals who have naturally attained $s_1$ (relative to $s_0$, in other words, students who were naturally Male over Female) is equal to the disparity associated with spurious discrimination, plus the advantage it lost due to indirect discrimination, minus the advantage it would have gained without direct discrimination. 

\subsection{Path-Specific Counterfactual Fairness}
In \citep{wu2019pcfairness}, Wu et al. proposed path-specific counterfactual fairness (PC fairness) that covers the previously mentioned fairness notions. Letting $\Pi$ be all causal paths from $S$ to $Y$ in the causal graph and $\pi$ be a subset of $\Pi$, the path-specific counterfactual fairness metric is defined as follows.

\begin{definition}[Path-specific Counterfactual Fairness (PC Fairness)] \label{def:psctff}
Given a factual condition $\VO = \vo$ where $\VO \subseteq \{S, \X, Y \}$ and a causal path set $\pi$, we achieve the PC fairness if: 
	\begin{equation}\PCE_{\pi}(s_1, s_0\mid  \vo) = P(y_{s_1 \vert \pi, s_0 \vert \overline{\pi}}\mid \mathbf{o}) - P(y_{s_0}\mid \mathbf{o}) =0\end{equation}
where $s_1, s_0 \in \{ s^+, s^-\}$.
\end{definition}
In order to achieve path-specific counterfactual fairness in the running example, the application decision system needs to be able to discern the causal effect of the applicants gender being Female along the fair and unfair pathways, and to disregard the effect along the pathways that are unfair.
%\xnote{I cannot follow this long sentence.}

We point out that we can simply define the PC Fairness on a classifier by replacing outcome $Y$ with the predictor $\hat{Y}$ in the above equation. Previous causality-based fairness notions can be expressed as special cases of the PC fairness
based on the value of $\mathbf{O}$ (e.g., $\emptyset$ or $S,{\mathbf{X}}$) and the value of $\pi$ (e.g., $\Pi$ or $\pi_d$). Their connections are summarised in Table~\ref{tab:connection}, where $\pi_d$ contains the direct edge from $S$ to $\Y$, and $\pi_i$ is a path set that contains all causal paths passing through any redlining attributes. The notion of PC fairness also resolves new types of fairness, e.g., individual indirect fairness, which means discrimination along the indirect paths for a particular individual. Formally, individual indirect fairness can be directly defined and analyzed using PC fairness by letting $\mathbf{O}=\{S,\mathbf{X}\}$ and $\pi=\pi_{i}$.

\subsection{Proxy Fairness}
\label{sec:proxy}
In \citep{kilbertus2017avoiding}, Kilbertus et al. proposed proxy fairness. A proxy is a descendant of $S$ in the causal graph whose observable quantity is significantly correlated with $S$, but should not affect the prediction. An example of a proxy variable in our running admission case can be seen in Fig. \ref{fig:proxy}. 
\begin{figure}
    \centering
    \includegraphics[width=.45\textwidth]{proxy.jpg}
    \caption{Extension of Fig. \ref{fig:cgexp} in which we add a proxy variable: name. Name is significantly correlated with the marginalization attribute gender since a persons name is often chosen based on gender.}
    \label{fig:proxy}
\end{figure}
\begin{definition}[Proxy Discrimination]
	\label{def:proxy}
	A predictor $\hat{Y}$ exhibits no proxy discrimination based on a proxy $P$ if for all $p,p'$ we have:
	\begin{equation}
	    P(\hat{y}\mid do(P = p)) = P(\hat{Y}\mid do(P = p'))
	\end{equation}
	%no proxy discrimination based on a proxy $P$ if for all $p,p' \in \text{Dom}(P, P(\hat{Y}_p) = P(\hat{Y}_{p'}))$. 
\end{definition}

%Note that $P(\hat{Y}_p)$ is equivalent to $P(\hat{Y}\mid do(P=p))$. 
Intuitively, a predictor satisfies proxy fairness if the distribution of $\hat{Y}$ under two interventional regimes in which $P$ set to $p$ and $p'$ is the same. \citep{kilbertus2017avoiding} presented the conditions and developed procedures to remove proxy discrimination given the structural equation model. 

\subsection{Justifiable Fairness}
In \citep{salimi2019interventional}, Salimi et al. presented a pre-processing approach for removing the effect of any discriminatory causal relationship between the marginalization attribute and classifier predictions by manipulating the training data to be non-discriminatory. The repaired training data can be seen as a sample from a hypothetical fair world.

\begin{definition}[$\mathbf{K}$-fair]
	\label{def:kfair}
For a give set of variables $\mathbf{K}$, a decision function is said to be $\mathbf{K}$-fair with regards to $S$ if, for any context $\mathbf{K}=\mathbf{k}$ and any outcome $Y=y$, 
$P(y_{s_0, \mathbf{k}}) = P(y_{s_1,\mathbf{k}})$. 
\end{definition}

Note that the notion of $\mathbf{K}$-fair intervenes on both the marginalization attribute $S$ and variables $\mathbf{K}$. It is more fine-grained than proxy fairness, but it does not attempt to capture fairness at the individual level. The authors further introduced justifiable fairness for applications where the user can specify admissible (deconfounding) variables through which it is permissible for the marginalization attribute to influence the outcome. In our example from Fig. \ref{fig:cgexp}, the admissible variable is the major.

\begin{definition}[Justifiable Fairness]
	\label{def:justifiable}
A fairness application is justifiable fair if it is $\mathbf{K}$-fair with regarding to all supersets $\mathbf{K} \supseteq \mathbf{A}$ where $\mathbf{A}$ is the set of admissible variables. 
\end{definition}

Different from previous causality-based fairness notions, which require the presence of the underlying causal model, the justifiable fairness notion is based solely on the notion of intervention. The user only requires specification of a set of admissible variables and does not need to have a causal graph. The authors also introduced a sufficient condition for testing justifiable fairness that does not require access to the causal graph. However, with the presence of the causal graph, if all directed paths from $S$ to $Y$ go through an admissible attribute in $\mathbf{A}$, then the algorithm is justifiably fair. If the probability distribution is faithful to the causal graph, the converse also holds. This means that our running example is not justifiably fair since the paths from gender to admission has two paths: gender $\to$ major $\to$ admission and gender $\to$ admission.

\subsection{Counterfactual Error Rate}
\label{sec:cer}
Zhang and Bareinboim \citep{zhang2018equality} developed a causal framework to link the disparities realized through equalized odds (EO) and the causal mechanisms by which the marginalization attribute $S$ affects change in the prediction $\hat{Y}$. EO, also referred to as error rate balance, considers both the ground truth outcome $Y$ and predicted outcome $\hat{Y}$. EO achieves fairness through the balance of the misclassification rates (false positive and negative) across different demographic groups. They introduced a family of counterfactual measures that allows one to explain the misclassification disparities in terms of the direct, indirect, and spurious paths from $S$ to $\hat{Y}$ on a structural causal model. Different from all previously discussed causality-based fairness notions, counterfactual error rate considers both $Y$ and $\hat{Y}$ in their counterfactual quantity. 

\begin{definition}[Counterfactual Direct Error Rate]
	\label{def:cder}
	Given a SCM and a classifier $\hat{y}=f(\widehat{\pa{}})$ where $\widehat{\Pa{}}$ is a set of input features of the predictor, the counterfactual direct error rate ($\mathrm{ER}^d$) for a sub-population $s,y$ (with prediction $\hat{y} \ne y)$ is defined as: \begin{equation}\mathrm{ER}^d_{s_0,s_1}(\hat{y}\mid s,y) = P(\hat{y}_{s_1,y,(\widehat{\Pa{}}\backslash S)_{s_0,y}}\mid s,y) - P(\hat{y}_{s_0,y}\mid s,y).\end{equation} 
\end{definition}
For an individual with the marginalization attribute $S=s$ and the true outcome $Y=y$, the counterfactual direct error rate calculates the difference of two terms. The first term is the prediction $\hat{Y}$ had $S$ been $s_1$, while keeping all the other features $\widehat{\Pa{}}\backslash S$ at the level that they would attain had $S=s_0$ and $Y=y$, whereas the second term is the prediction $\hat{Y}$ the individual would receive had $S$ been $s_0$ and $Y$ been $y$. 

\begin{definition}[Counterfactual Indirect Error Rate]
	\label{def:cier}
	Given a SCM and a classifier $\hat{y}=f(\widehat{\pa{}})$, the counterfactual indirect error rate ($\mathrm{ER}^i$) for a sub-population $s,y$ (with prediction $\hat{y} \ne y)$ is defined as: \begin{equation}\mathrm{ER}^i_{s_0,s_1}(\hat{y}\mid s,y) = P(\hat{y}_{s_0,y,(\hat{PA}\backslash S)_{s_1,y}}\mid s,y) - P(\hat{y}_{s_0,y}\mid s,y).\end{equation} 
\end{definition}

\begin{definition}[Counterfactual Spurious Error Rate]
	\label{def:cser}
	Given a SCM and a classifier $\hat{y}=f(\widehat{\pa{}})$, the counterfactual spurious error rate ($\mathrm{ER}^s$) for a sub-population $s,y$ (with prediction $\hat{y} \ne y)$ is defined as: \begin{equation}\mathrm{ER}^s_{s_0,s_1}(\hat{y}\mid y) = P(\hat{y}_{s_0,y}\mid s_1,y) - P(\hat{y}_{s_0,y}\mid s_0,y).\end{equation} 
\end{definition}

The counterfactual spurious error rate can be read as ``for two demographics $s_0$, $s_1$ with the same true outcome $Y=y$, how would the prediction $\hat{Y}$ differ had they both been $s_0$, $y$?'' For a graphical depiction of these measures, we refer interested reader to the tutorial by Bareinboim, Zhang, and Plecko \citep{bareinboim_zhang_plecko}.

Building on these measures, Zhang and Bareinboim \citep{zhang2018equality} derived the causal explanation formula for the error rate balance. The equalized odds notion constrains the classification algorithm such that its disparate error rate is equal to zero across different demographics. 

\begin{definition}[Error Rate Balance] \label{def:erb}
	The error rate (ER) balance is given by:
	\begin{equation} \mathrm{ER}_{s_0, s_1}(\hat{y}\mid y) = P(\hat{y}\mid s_1,y) - P(\hat{y}\mid s_0,y). \end{equation}
\end{definition}

\begin{theorem}[Causal Explanation Formula of Equalized Odds] \label{th:expeo}
For any $s_0$, $s_1$, $\hat{y}$, $y$, we have the following relationship: 
 \begin{equation}\mathrm{ER}_{s_0,s_1}(\hat{y}\mid y) = \mathrm{ER}^d_{s_0,s_1}(\hat{y}\mid s_0,y) - \mathrm{ER}^i_{s_1,s_0}(\hat{y}\mid s_0,y) - \mathrm{ER}^s_{s_1,s_0}(\hat{y}\mid y).\end{equation}
\end{theorem}

The above theorem shows that the total disparate error rate can be decomposed into terms, each of which estimates the adverse impact of its corresponding discriminatory mechanism. 

\subsection{Individual Equalized Counterfactual Odds}
In \citep{pfohl2019counterfactual}, Pfohl et al. proposed the notion of individual equalized counterfactual odds that is an extension of counterfactual fairness and equalized odds. The notion is motivated by clinical risk prediction and aims to achieve equal benefit across different demographic groups. 

\begin{definition}[Individual Equalized Counterfactual Odds]\label{def:ieco}
	Given a factual condition $\VO = \vo$ where $\VO \subseteq \{\X, Y \}$, predictor $\Y$ achieves the individual equalized counterfactual odds if: 
	\begin{equation} P(\hat{y}_{s_1} \mid  \mathbf{o},y_{s_1}, s_0) - P(\hat{y}_{s_0} \mid  \mathbf{o}, y_{s_0}, s_0) =0\end{equation}
	where $s_1, s_0 \in \{ s^+, s^-\}$.
\end{definition}

The notion implies that the predictor must be counterfactually fair given the outcome $Y$ matching the counterfactual outcome $y_{s_0}$. This is different than the normal counterfactual fairness calculation in Definition \ref{def:cf}, which requires the prediction to be equal across the factual/counterfactual pairs, without caring if those pairs have the same outcome prediction. Therefore, in addition to requiring predictions to be equal across factual/counterfactual samples, those samples must also share the same value of the actual outcome $Y$. In other words, it considers the desiderata from both counterfactual fairness and equalized odds. For our running example, this is an extension of the discussion under Definition \ref{def:cf} in which we now require that $\hat{y}_{s_0} = \hat{y}_{s_1}$.

\subsection{Fair on Average Causal Effect}
\label{face}
In \citep{khademi2019fairness}, Khademi et al. introduced two definitions of group fairness: fair on average causal effect (FACE), and fair on average causal effect on the treated (FACT) based on the Rubin-Neyman potential outcomes framework. Let $Y_i(s)$ be the potential outcome of an individual data point $i$ had $S$ been $s$. 

\begin{definition}[Fair on Average Causal Effect (FACE)]
	\label{def:face}
A decision function is said to be fair, on average over all individuals in the population, with respect to $S$, if $\mathbb{E}[Y_i(s_1) - Y_i(s_0)] =0$.
\end{definition}

FACE considers the average causal effect of the marginalization attribute $S$ on the outcome $Y$ at the population level and is equivalent to the expected value of the $\TCE(s_1, s_0)$ in the structural causal model.

\begin{definition}[Fair on Average Causal Effect on the Treated (FACT)]
	\label{def:fact}
A decision function is said to be fair with respect to $S$, on average over individuals with the same value of $s_1$, if $\mathbb{E}[Y_i(s_1) - Y_i(s_0)\mid S_i =s_1] =0$.
\end{definition}
 
FACT focuses on the same effect at the group level. This is equivalent to the expected value of $ETT_{s_1,s_0}(Y)$. The authors used inverse probability weighting to estimate FACE and use matching methods to estimate FACT.  

\subsection{Equality of Effort}
In \citep{DBLP:conf/www/HuangW0W20}, Huang et al. developed a fairness notation called equality of effort. When applied to the example in Fig. \ref{fig:cgexp}, we have a dataset with $N$ individuals with attributes $(S, T, \mathbf{X}, Y)$ where $S$ denotes the marginalization attribute gender with domain values $\{ s^+, s^-\}$, $Y$ denotes a decision attribute admission with domain values $\{ y^+, y^-\}$, $T$ denotes a legitimate attribute such as test score, and $\mathbf{X}$ denotes a set of covariates. For an individual $i$ in the dataset with profile $(s_{i}, t_{i}, \mathbf{x}_{i}, y_{i})$, they may ask the counterfactual question, how much they should improve their test score such that the probability of their admission is above a threshold $\gamma$ (e.g., $80\%$). 


\begin{definition}[$\gamma$-Minimum Effort]
	\label{def:min_effort}
	For individual $i$ with value $(s_{i}, t_{i}, \mathbf{x}_{i}, y_{i})$, the minimum value of the treatment variable to achieve $\gamma$-level outcome is defined as:
	\begin{equation}
	\Psi_i (\gamma) = \argmin_{t\in T} \big\{ \mathbb{E}[Y_i(t)] \geq \gamma)    \}
	\end{equation}
	and the minimum effort to achieve $\gamma$-level outcome is $\Psi_i (\gamma)- t_{i}$.
\end{definition}

If the minimal change for individual $i$ has no difference from that of counterparts (individuals with similar profiles except the marginalization attribute), individual $i$ achieves fairness in terms of equality of effort. As $Y_i(t)$ cannot be directly observed, we can find a subset of users, denoted as $I$, each of whom has the same (or similar) characteristics ($\mathbf{x}$ and $t$) as individual $i$. $I^*$ denotes the subgroup of users in $I$ with the marginalization attribute value $s^*$ where $* \in \{+,-\}$ and $\mathbb{E}[Y_{I^*}(t)]$ denotes the expected outcome under treatment $t$ for the subgroup $I^*$.

\begin{definition}[$\gamma$-Equal Effort Fairness]
	\label{def:equ_effort_i}
	For a certain outcome level $\gamma$, the equality of effort for individual $i$ is defined as:
	\begin{equation}
	\Psi_{I^+}(\gamma) = \Psi_{I^-}(\gamma).
	\end{equation}
where $\Psi_{I^*}(\gamma) = \argmin_{t\in T} \{\mathbb{E}[Y_{I^*}(t)] \geq \gamma \}$ is the minimal effort needed to achieve $\gamma$ level of outcome variable within the subgroup $* \in \{+,-\}$.
\end{definition}

Equal effort fairness can be straightforwardly extended to the system (group) level by replacing $I$ with the whole dataset $D$ (or a particular group). Different from previous fairness notations that mainly focus on the the effect of the marginalization attribute $S$ on the decision attribute $Y$, the equality of effort instead focuses on to what extend the treatment variable $T$ should change to make the individual achieve a certain outcome level. This notation addresses the concerns whether the efforts that would need to make to achieve the same outcome level for individuals from the marginalized group and the efforts from the non-marginalized group are different. For instance, if we have two students with the same credentials minus their gender, and the Female student was required to raise their test score significantly more than the Male, then we do not achieve equal effort fairness.

\subsection{Technical Pitfalls of Causality-based Fairness}
\label{sec:pitfall}
Causality provides a conceptual and technical framework for measuring and mitigating unfairness by using the causal effect on a decision from hypothetical interventions on marginalization attributes such as gender.  Despite the benefits of causality-based notions over statistical-based ones, there have been technical challenges in applying causality for fair machine learning in practice.  One common challenge is the validity of the assumptions in causal modeling. As discussed in Section \ref{sec:cfair}, the majority of research on causal fairness is based on SCM which represents the causal relationships between variables via structural equations and a directed acyclic graph (DAG). In practice, learning structural equations and constructing the DAG model from observational data is a challenging task and often relies on strong assumptions such as the Markov property, faithfulness, and sufficiency \citep{glymour2019review}. Simply speaking, the Markov property requires that all nodes are independent of their non-descendants when conditioned on their parents; faithfulness requires all conditional independent relationships in the true underlying distribution are represented in the DAG; and sufficiency requires any pair of nodes in the DAG has one common external cause (confounder). These assumptions help narrow down the model space, however, they may not hold in the causal process or the sampling process that generates the observed data. 

Another common challenge of causality-based fairness notions based on SCMs is identifiability, i.e., whether they can be uniquely measured from observational data. As causality-based fairness notions are defined based on different types of causal effects, such as total effect on interventions, direct/indirect discrimination on path-specific effects, and counterfactual fairness on counterfactual effects, their identifiability depends on the identifiability of these causal effects. Unfortunately, in many situations these causal effects are unidentifiable. Hence identifiability is a critical barrier for causality-based fairness to be applied to real applications. In the causal inference field, researchers have studied the reasons for unidentifiability and identified the corresponding structural patterns such as the existence of the ``kite graph'', the ``w graph'', or the ``hedge graph''. We refer readers who are interested in learning the specifics of identifiability theory and criteria, and how they can be used to decide the applicability of causality-based fairness metrics to \citep{makhlouf2021survey}. We also  refer readers to \citep{wu2019pcfairness} for a summary of unidentifiable situations and approximation techniques to derive bounds of causal effects. 

The potential outcome framework does not require the causal graph. However, as discussed in Section \ref{sec:pof}, it relies on three assumptions.  SUTVA is a non-interference assumption which may not hold in many real world applications. For example, a loan officer's decision to proceed with one application may be influenced by previous applications. In this case, SUTVA is violated. When the strong ignorability assumption does not hold, there exist hidden confounders. Although we can leverage mediating features or proxies to estimate treatment effects \citep{miao2018identifying}, the lack of accuracy guarantee hinders the applicability of causal fairness. 
