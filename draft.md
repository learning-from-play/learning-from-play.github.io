## Abstract

We propose a self-supervised approach to learning a wide variety of manipulation skills from play data--unlabeled, unsegmented, undirected interaction with objects in an agent's environment. Learning from play offers three main advantages: 1) Collecting large amounts of play data is cheap and fast as it does not require staging the scene, resetting to an initial state, or labeling tasks, 2) It relaxes the need to have a discrete and rigid definition of skills/tasks prior to data collection. This allows the agent to focus on acquiring a continuum of manipulation skills, which can then be conditioned to perform a particular skill such as grasping. Furthermore, this data already includes ways to recover, retry or transition between different skills, which can be used to achieve a reactive closed-loop control policy, 3) It allows the robot to quickly learn a new skill from making use of pre-existing general abilities.
Our proposed approach to learning new skills from unlabeled play data decouples high-level plan prediction from low-level action prediction by: first self-supervise learning of a latent planning space, then self-supervise learning of an action model that is conditioned on a latent plan. This results in a single task-agnostic policy conditioned on a user-provided goal. This policy can perform a wide variety of tasks in the environment where playing was observed. We train a single model on 3 hours of unlabeled play data and evaluate it on 18 zero-shot manipulation tasks, simply by feeding a goal state corresponding to each task. The supervised baseline model reaches 65\% average task success, using 18 behavioral cloning policies trained on 100 demonstrations per task (1800 total). Our model completes the tasks with an average of 85\% success using a single policy in zero shots (having never been explicitly trained on these tasks) using cheap unlabeled data. When the starting position is perturbed, our model trained on play data remains robust with an accuracy of 79\% while the baseline drops to 23\%. We also qualitatively observe retry-until-success behaviors naturally emerging, and a natural organization of plan space around tasks without ever being trained with task labels.
Videos of the performed experiments are available at [site]

[site]: https://sites.google.com/view/sslmp

## Introduction

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/play_data640x360.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Play data collected from human tele-operation.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/tasks960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 140%;"></video>
<figcaption>
Figure 1: caption
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/runs960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Some successful runs.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/retry960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Retry behaviors emerges naturally.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/failures960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Some failure cases.
</figcaption>
</div>

<div class="figure">
<video class="b-lazy" data-src="assets/mp4/compose2960x540.mp4" type="video/mp4" autoplay muted playsinline loop style="display: block; width: 100%;"></video>
<figcaption>
Figure 1: Composing 2 tasks.
</figcaption>
</div>

<div class="figure">
<img src="assets/fig/lmp_teaser4.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure 2: 
</figcaption>
</div>

<div class="figure">
<img src="assets/fig/lmp_inference4.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure 3: 
</figcaption>
</div>

<div class="figure">
<img src="assets/fig/tsne.svg" style="margin: 0; width: 80%;"/>
<figcaption>
Figure : 
</figcaption>
</div>
