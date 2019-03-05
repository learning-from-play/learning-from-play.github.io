## Abstract

We propose a self-supervised approach to learning a wide variety of manipulation skills from play data--unlabeled, unsegmented, undirected interaction with objects in an agent's environment. Learning from play offers three main advantages: 1) Collecting large amounts of play data is cheap and fast as it does not require staging the scene, resetting to an initial state, or labeling tasks, 2) It relaxes the need to have a discrete and rigid definition of skills/tasks prior to data collection. This allows the agent to focus on acquiring a continuum of manipulation skills, which can then be conditioned to perform a particular skill such as grasping. Furthermore, this data already includes ways to recover, retry or transition between different skills, which can be used to achieve a reactive closed-loop control policy, 3) It allows the robot to quickly learn a new skill from making use of pre-existing general abilities.
Our proposed approach to learning new skills from unlabeled play data decouples high-level plan prediction from low-level action prediction by: first self-supervise learning of a latent planning space, then self-supervise learning of an action model that is conditioned on a latent plan. This results in a single task-agnostic policy conditioned on a user-provided goal. This policy can perform a wide variety of tasks in the environment where playing was observed. We train a single model on 3 hours of unlabeled play data and evaluate it on 18 zero-shot manipulation tasks, simply by feeding a goal state corresponding to each task. The supervised baseline model reaches 65\% average task success, using 18 behavioral cloning policies trained on 100 demonstrations per task (1800 total). Our model completes the tasks with an average of 85\% success using a single policy in zero shots (having never been explicitly trained on these tasks) using cheap unlabeled data. When the starting position is perturbed, our model trained on play data remains robust with an accuracy of 79\% while the baseline drops to 23\%. We also qualitatively observe retry-until-success behaviors naturally emerging, and a natural organization of plan space around tasks without ever being trained with task labels.
Videos of the performed experiments are available at [site]

[site]: https://sites.google.com/view/sslmp

## Introduction

There has been significant recent progress showing that robots can be trained to be competent specialists, learning individual skills like grasping (\citet{kalashnikov2018qt}), locomotion and dexterous manipulation (\citet{haarnoja2018soft}). In this work, we focus instead on the concept of a generalist robot: a single robot capable of performing many different complex tasks without having to relearn each from scratch. This is a long standing goal in both robotics and artificial intelligence.

#### Learning From Play
\textbf{Learning From Play} is a fundamental and general method humans use to acquire a repertoire of complex skills and behaviors (<dt-cite key="wood2005play"></dt-cite>). It has been hypothesized (\citet{pellegrini2007play}, \citet{robert1981animal}, \citet{hinde1983ethology}, \citet{sutton2009ambiguity}) that play is a crucial adaptive property--that an extended period of immaturity in humans gives children the opportunity to sample their environment, learning and practicing a wide variety of strategies and behaviors in a low-risk fashion that are effective in that niche.

#### What is play?
Developmental psychologists and animal behaviorists have offered multiple definitions (\citet{burghardt2005genesis}, \citet{robert1981animal}, \citet{hinde1983ethology}, \citet{pellegrini2002children}, \citet{sutton2009ambiguity}). \citet{burghardt2005genesis}, reviewing the different disciplines, distills play down to ``a non-serious variant of functional behavior" and gives three main criteria for classifying behavior as play: 1) Self-guided. Play is spontaneous and directed entirely by the intrinsic motivation, curiosity, or boredom of the agent engaging in it. 2) Means over ends. Although play might resemble functional behavior at times, the participant is typically more concerned with the behaviors themselves than the particular outcome. In this way play is ``incompletely functional". 3) Repeated, but varied. Play involves repeated behavior, but behavior that cannot be rigidly stereotyped. In this way, play should contain multiple ways of achieving the same outcome. Finally, all forms of play are considered to \textit{follow} exploration (\citet{belsky1981exploration}). That is, before children can play with an object, they must explore it first (\citet{hutt1966exploration}), inventorying its attributes and affordances. Only after rich object knowledge has been built up to act as the bases for play does play displace exploration.


#Method

##Play data

Consider a long sequence of play data, which includes unlabeled, unsegmented sequence of states and actions:

$\mathcal{D} = \{(s_1, a_1), (s_2, a_2), \cdots, (s_T, a_T)\}$





<div class="figure">
<video class="b-lazy" data-src="assets/mp4/play_data516x360.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Play data collected from human tele-operation.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/tasks960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 140%;"></video>
<figcaption>
Figure 1: 18 tasks defined for evaluation only.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/runs960x398.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Some successful runs.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/retry960x398.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Retry behaviors emerges naturally.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/failures960x398.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Some failure cases.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/compose2960x500.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Composing 2 tasks.
</figcaption>
</div>

<div class="figure">
<img src="assets/fig/lmp_teaser5.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure 2: 
</figcaption>
</div>

Our work on learning latent plans is most related to \citet{hausman2018learning}, who present a method for reinforcement learning of closely related manipulation skills, parameterized via an explicit skill embedding space. They assume a fixed set of initial tasks at training time, with access to accompanying per task reward functions to drive policy and embedding learning.
In contrast, our method relies on unsegmented, unlabeled play data with no predefined task training distribution.
It additionally requires no reward function, and performs policy training via supervised learning, yielding orders of magnitude greater sample efficiency. Finally, they generalize to new skills by freezing the learned policy and learning a new mapping to the embedding space, whereas \lmp generalizes to new tasks simply by feeding a new current and goal state pair to the trained plan proposal network.

Our learning method for latent plans is self-supervised, and relates to other works in self-supervised representation learning from sequences \citet{wang2015unsupervised,misra2016shuffle,Sermanet2017TCN}.
% Our learning method for latent plans is self-supervised, and relates to other works in self-supervised representation learning from sequences (\citet{wang2015unsupervised,misra2016shuffle,Sermanet2017TCN}).

<div class="figure">
<img src="assets/fig/lmp_inference4.svg" style="margin: 0; width: 50%;"/>
<figcaption>
Figure 3: 
</figcaption>
</div>

Lastly, our work is related to prior research on few-shot learning of skills from demonstrations (\citet{finn2017one,wang2017robust,DBLP:journals/corr/JamesDJ17,DBLP:journals/corr/abs-1806-10166,DBLP:journals/corr/DuanASHSSAZ17}).
While our method does not require demonstrations to perform new tasks -- only the goal state -- it can readily incorporate demonstrations simply by treating each subsequent frame as a goal. In contrast to prior work on few-shot learning from demonstration that require a meta-training phase (\citet{finn2017one}), our method does not require any expensive task-specific demonstrations for training or a predefined task distribution, only non-specific play data. In contrast to prior work that uses reinforcement learning (\citet{DBLP:journals/corr/abs-1810-05017}), it does not require any reward function or costly RL phase.

<div class="figure">
<img src="assets/fig/tsne.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure : 
</figcaption>
</div>
