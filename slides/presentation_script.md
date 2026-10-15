# Presentation Script: Towards a Universal Feature Extractor for Atari Games

**Target length:** about 25 minutes of talking, leaving about 5 minutes for questions.
**How to use this:** each slide has a **cue** (one line to remember), a **Report:** pointer and the
**spoken text**. Learn the cues by heart and use the text for practice; don't read it word for word.
Lines starting with **Takeaway** are the one sentence the audience should remember from that slide.
Sentences in *[optional: …]* add detail; say them if you are on time, skip them if you are behind.
The spoken text uses "I" where it sounds natural; the slides and the report stay in the passive voice.

**Report pointers** give the section, table or figure, and page in the current `report_1/main.pdf`
(19 pages). If the report changes, page numbers may shift; section numbers stay the same.

**If you are short on time**, drop all *[optional]* sentences and shorten slides 8, 12, 14 and 17 to
one sentence each (saves about 3–4 minutes).

**Background for you:** the section after the cheat sheet explains the six points you asked about
(one network, copying experts, reusing features, why the autoencoders, the downstream table, the
9,216-dimensional feature) in more depth.

---

## Part 1: Introduction (≈ 2 min)

### Slide 1: Title (0:20)
**Cue:** name, title, the one question, supervisor.
**Report:** title page

> Good morning, everyone. My name is Sajid Ali Sadruddin, and I will present my research project,
> "Towards a Universal Feature Extractor for Atari Games", with the subtitle "Reconstruction, FSQ and
> Joint-Embedding Prediction". The question behind it is simple: can one encoder, trained once, help
> an agent play many different Atari games? The project was supervised by Oleh Shkalikov.

### Slide 2: Outline (0:15)
**Cue:** five parts.
**Report:** §1 Introduction · §2 Related Work · §3 Methodology · §4 Results · §5 Conclusion

> The talk has five parts: the motivation, related work, the method, the results, and the
> conclusion.

### Slide 3: Motivation and Research Questions (1:30)
**Cue:** same input everywhere, yet every agent learns its own CNN → one shared encoder? Two choices: objective and usage.
**Report:** §1 Introduction, p. 1–2 (RQ1–RQ4 at the end of p. 2)

> An RL agent for Atari has two parts: an **encoder**, which turns the raw pixels into a short list
> of numbers, the features, and a **policy**, which uses these features to choose an action.
>
> Every Atari game gives the agent exactly the same type of input: four stacked grey-scale frames of
> 84 by 84 pixels. Still, the standard agent trains its own small encoder, the NatureCNN, from
> scratch for every game, and it learns only from the reward.
>
> This is wasteful, because many games look alike: small sprites, balls, walls and scrolling
> backgrounds. So the idea is: pre-train **one** encoder on many games, without rewards, and reuse
> it for PPO on every game. *[optional: If this works, the encoder is trained once, every new game
> starts from it, and all games share one latent space.]*
>
> This raises two design choices. The **objective**: should the encoder learn by reconstructing the
> pixels, or by predicting the next latent state? And the **usage**: do we keep the encoder frozen,
> or fine-tune it together with the policy?
>
> This leads to four research questions. RQ1: which objective gives the most useful features for
> PPO? RQ2: is it better to freeze or to fine-tune the encoder? RQ3: do the features work on games
> never seen in pre-training? And RQ4: what does it cost in GPU-hours and energy, and can we inspect
> the features, for example by decoding them back into images?

---

## Part 2: Related Work (≈ 3.5 min)

### Slide 4: Related Work: Same Problem, One Model for Many Games (1:30)
**Cue:** one agent for all games · one student copying many experts · reuse one game's features for the next.
**Report:** §2 Related Work, *Multi-Game Agents and Transfer*, p. 2 (all six works of the slide are cited there)

> Let me start with others who worked on the same goal: one model for many Atari games. The table
> has three groups, and the right column says how this work differs.
>
> **One agent, many games.** Normally, every game gets its own neural network. PopArt,
> Multi-Game Decision Transformers and Scaled Q-Learning instead train **one network**, one single
> set of weights, that plays 40 to 57 games: whatever game it sees, the same network chooses the
> action. But they share the whole agent, including the part that decides, and they learn from
> rewards. Here, only the encoder is shared, and pre-training uses no rewards.
>
> **Learning from experts.** Policy Distillation and Actor-Mimic first train a separate expert for
> every game. Then one student network learns to copy them: on a Pong frame it must pick the action
> the Pong expert would pick, on a Breakout frame the action of the Breakout expert. Here, the
> experts only provide the data; nothing is trained to copy their decisions.
>
> **Transfer to a new game.** A network trained on one game has already learned to see: edges,
> sprites, a ball. Progressive Networks and Farebrother et al. reuse these features as the starting
> point for the next game, or the next mode of the same game, instead of learning to see from zero.
> Here, one encoder learns from eleven games at once, not from one game at a time.
>
> **Takeaway:** others share the whole agent, copy experts, or transfer game by game; here, only the
> encoder is shared, it is trained without rewards, and it learns from eleven games at once.

### Slide 5: Related Work: From Images and Videos to Games (1:00)
**Cue:** each method was built for images, videos or robots; here it is adapted to games.
**Report:** §2 Related Work, *Autoencoders* and *Joint-Embedding Predictive Architectures*, p. 2; *Multi-Task World Models*, p. 3

> Now the methods this work builds on. None of them was built for games. The middle column shows
> what each method was built for, the right column what was changed in this work.
>
> The VAE, VQ-VAE and FSQ were built to generate images, video and audio, where the goal is a
> faithful reconstruction. Here, only their encoder is reused, as the feature extractor for PPO, and
> the loss focuses on the few pixels that move.
>
> I-JEPA learns from images and V-JEPA from videos. Both hide regions of the input and predict their
> representations, and neither knows about actions. Here, the JEPA predicts the **next** frame stack,
> given the action and the game, and one model is trained on eleven games.
>
> TD-MPC2 trains one world model for 80 robot-control tasks and plans with it. Two of its ideas, the
> task embedding and SimNorm, are reused here, but there is no planning.
>
> **Takeaway:** images and videos have no actions; in games, the agent's actions change what happens
> next.
>
> One more point is important for the rest of the talk: the agent here is always model-free, as the
> next slide shows.

### Slide 6: Related Work: Model-Based vs. Model-Free RL (1:00)
**Cue:** world model trains the encoder only, PPO chooses the actions.
**Report:** §2 Related Work, *Multi-Task World Models*, p. 3; last sentence of §1, p. 2

> The most important point about this project: it is **model-free**.
>
> The top lane shows model-based RL, for example Dreamer, IRIS, MuZero, TD-MPC2 or V-JEPA 2-AC. An
> encoder feeds a world model that predicts the next state and the reward, and the agent uses it to
> plan or to imagine. The world model chooses the actions.
>
> The lower lane shows this work, in two steps. In step one, the encoder is pre-trained on expert
> data from eleven games, without rewards, together with a decoder or a JEPA predictor. Then the
> decoder or predictor is thrown away. In step two, only the encoder weights are copied into a normal
> PPO agent, frozen or fine-tuned, which plays each game and learns from the reward.
>
> **Takeaway:** the world model only trains the encoder; PPO chooses the actions.

---

## Part 3: Methodology (≈ 8 min)

### Slide 7: Data Collection (0:50)
**Cue:** experts from HuggingFace → frame dataset and transition dataset; 11 + 4 games.
**Report:** §3.2 Data Collection, p. 3; Fig. 1, p. 4

> Now the method, starting with the data. A random policy rarely gets past the first screen, so
> pre-trained expert agents from the HuggingFace Hub played the games, one per game. Their actions
> were sampled from their policies, so the data also contains some sub-optimal play.
>
> Part (a) shows the two datasets. The **frame dataset** has 50 thousand frame stacks per game for
> the autoencoders: 550 thousand from the eleven pre-training games and 200 thousand from the four
> held-out games. The **transition dataset** has state, action, next state and an end-of-episode flag
> for the V-JEPA: 250 thousand per game, 2.75 million in total, plus a validation split.
>
> Part (b) shows the processing. Consecutive stacks share three of their four frames, so the next
> state is already 75 percent identical to the current one. Transitions that end an episode are
> masked out, game-specific action numbers are mapped to one common set of 18 actions, and moving
> pixels are marked for the FSQ loss. Eleven games are used for pre-training, four are held out for
> testing.

### Slide 8: The Atari Games (0:45)
**Cue:** same shape, very different content, tiny moving sprites, different action sets.
**Report:** §3.3 The Atari Games, p. 3–4; Fig. 2 and Table 1, p. 5

> These are all fifteen games: the frame on the left, the change to the next frame on the right; the
> four held-out games are marked with a dagger. The input shape is always the same, but the content
> is very different: from 1 percent black pixels in Pong to 95 percent in Asteroids.
>
> In most games less than 4 percent of the pixels move; only BeamRider and Riverraid move large parts
> of the screen. And the games have 4 to 18 actions, where the same action number means a different
> button in each game.
>
> **Takeaway:** what matters are a few small sprites. This is why the models give extra weight to
> moving pixels.

### Slide 9: Autoencoders (2:00)
**Cue:** rebuild the frame → can't ignore objects; VAE (baseline) → VQ-VAE (grid + codebook) → FSQ (grid, no codebook); moving pixels 85 %.
**Report:** §3.4 Autoencoders, p. 5–6 (*Standard VAE*, *Residual VQ-VAE*, *FSQ-VAE*, Eq. 1, p. 6); the diagram itself is only in the slides

> The top of the slide shows the general principle. An autoencoder squeezes the frame through an
> encoder into a small bottleneck, and a decoder rebuilds the frame from it. The loss is the
> difference between the input and the reconstruction. After training, only the encoder is kept for
> PPO.
>
> Why autoencoders? To rebuild the frame, the encoder cannot ignore anything visible, including
> small objects. And the decoder shows us what the encoder keeps.
>
> The three boxes below are the three bottlenecks, from coarse to fine. Each one fixes a weakness of
> the previous one.
>
> The **Standard VAE** is the baseline. Its encoder predicts a mean mu and a spread sigma, and the
> latent z is sampled from that Gaussian. It uses exactly the NatureCNN layers, so any gain comes
> from pre-training alone. But it squeezes the whole frame into one vector of 256 numbers, and one
> cell of its feature map covers 12 by 12 pixels, larger than a ball.
>
> The **Residual VQ-VAE** keeps a 21 by 21 grid, so each cell covers only 4 by 4 pixels and the
> position of every object is kept. Each cell is replaced by the nearest of 512 learned code vectors,
> the codebook, like a vocabulary of visual patterns. That fits Atari, which is built from a few
> repeated sprites, and discrete codes already work well in Atari world models such as DreamerV2 and
> IRIS. The weakness: the codebook must be learned, and some codes are never used.
>
> The **FSQ-VAE** keeps the same grid but replaces the learned codebook with simple rounding: each of
> four channels is rounded to the nearest of a few fixed levels, 8, 5, 5 and 3, which gives 600
> possible values per cell. There is no codebook to learn, so there are no unused codes and fewer
> settings to tune, and the model is small: 0.84 million parameters. *[optional: Its decoder also
> uses upsampling instead of transposed convolutions, which avoids checkerboard artefacts on the flat
> Atari backgrounds.]*
>
> Finally, small moving objects were often lost in the reconstructions, so moving pixels get a higher
> weight in the loss: in the FSQ-VAE they always count for 85 percent of the loss, however few of
> them there are.

### Slide 10: V-JEPA World Model (1:15)
**Cue:** predict the next *latent*, not pixels; EMA target without gradients; keep only the context encoder.
**Report:** §3.5 Action- and Game-Conditioned V-JEPA, p. 6–7; Fig. 3 and Eq. 2, p. 7

> The second family is the JEPA. It does not rebuild pixels; it predicts the **representation** of
> the next state. The hope was that it does not waste capacity on pixel details, and, because the
> predictor also gets the action, that the features encode what the agent controls.
>
> The frame stack is cut into 36 patches of 14 by 14 pixels, each turned into a token of 256 numbers.
> The context encoder, a six-layer transformer, encodes the current state. The target encoder
> encodes the next state; it is a slowly updated copy of the context encoder and gets no gradients,
> which helps against collapse. The predictor, three transformer layers, gets the context tokens
> plus an action token and a game token, and must predict the target tokens. Transitions at the end
> of an episode are masked out of the loss. A small inverse head guesses the action from two
> consecutive states.
>
> **Takeaway:** the model predicts the latent of the next frame stack, not its pixels, and only the
> context encoder is used for PPO.

### Slide 11: V-JEPA: Design Choices (0:50)
**Cue:** four TD-MPC2 ideas to train one model on many games; table = who learns from which loss.
**Report:** §3.5, p. 7 (four design choices); Table 2 and *Training Step*, p. 8

> To train one model on many games, four ideas from TD-MPC2 were used. First, one common set of 18
> actions, because the same button number means something different in each game. Second, a
> learnable embedding per game, replaced by an "unknown" embedding a quarter of the time, so the
> model also has something to use on new games. Third, SimNorm: the latent is split into groups of
> 8 values with a softmax on each group, which keeps the features bounded and makes collapse harder.
> Fourth, the inverse head, with a small weight of 0.01 in the total loss.
>
> The table shows which part learns from which loss. The context encoder, 4.95 million parameters,
> learns from both losses. The predictor learns only from the JEPA loss, the inverse head only from
> the inverse loss. The target encoder is only a moving average. *[optional: This matters later: the
> inverse loss, which should make the model care about actions, never reaches the predictor.]*

### Slide 12: Experimental Setup: Pre-Training (0:45)
**Cue:** autoencoders < 3 M params, about 1–2 h; V-JEPA 12.5 M params, 15.6 h, batch 512; checkpoint at epoch 92.
**Report:** §3.6 Experimental Setup, *Pre-Training*, p. 8; Table 3, p. 9

> This table shows how each encoder was pre-trained. The three autoencoders saw the 550 thousand
> frame stacks, the V-JEPA the 2.75 million transitions. The autoencoders are small, below 3 million
> parameters, and each trained in under two hours on one H100. The V-JEPA has 12.5 million
> parameters and trained for 92 epochs, about 15.6 hours.
>
> The architectures follow the original papers: VQ-VAE, FSQ, I-JEPA and TD-MPC2. The JEPA papers
> train much larger models on many GPUs, so for the 12.5 million parameter model on one GPU a
> smaller batch of 512 and a lower learning rate were used.
>
> For the V-JEPA, the checkpoint with the lowest total validation loss, the JEPA loss plus 0.01 times
> the inverse loss, was used; that was epoch 92. A small sweep over the inverse-loss weight confirmed
> 0.01.

### Slide 13: Experimental Setup: PPO and Evaluation (0:50)
**Cue:** scratch vs frozen vs fine-tuned; RL-Zoo PPO, 10 M steps; HNS: 0 = random, 1 = human; one seed.
**Report:** §3.6, *PPO Training and Evaluation*, p. 8–9

> Three settings are compared. The baseline is the NatureCNN, trained from scratch. **Frozen** means
> the pre-trained encoder is locked and only the actor and critic learn. **Fine-tuned** means the
> pre-trained encoder is only the starting point and keeps learning together with the policy.
>
> PPO uses the standard RL-Zoo Atari settings for 10 million steps, with the same 512-unit heads for
> every encoder, so only the encoder differs.
>
> The metric is the mean return over the last million steps. Scores differ a lot between games, so
> they are turned into the human-normalised score: 0 means random play, 1 means human level. Each
> configuration was run with one seed, and all numbers come from the full Weights & Biases logs.

### Slide 14: Scope of the Work (0:30)
**Cue:** 15 games, 5 encoders, 168 runs, 1.68 billion steps.
**Report:** §3.6, last paragraph, p. 9 (168 runs); §4.3 (diagnostics), p. 13; Table 9, p. 16

> In numbers: fifteen games, eleven plus four held out; 3.8 million samples from the expert agents;
> five pre-trained encoders, counting both VQ-VAE versions; six tests of the world model; and 168 PPO
> runs of 10 million steps each, so 1.68 billion agent steps in total. The code is about 10 thousand
> lines of Python.
>
> The runs split into 15 for the NatureCNN, 30 each for the three autoencoders, frozen and
> fine-tuned on all fifteen games, and 63 for the V-JEPA, including the ablations. Everything ran
> from May to September 2026 on the H100 GPUs of the ZIH Capella cluster.

---

## Part 4: Results (≈ 9 min)

### Slide 15: Pre-Training Results (1:00)
**Cue:** FSQ rebuilds sprites best; nobody generalises; JEPA loss rose while its rank went 1.4 → 57.
**Report:** §4.1 Pre-Training, p. 10 (the reconstruction image is only in the slides)

> Now the results, first the pre-training. The figure shows rebuilt frames of seen and held-out
> games. The FSQ-VAE had the lowest error, and three times less error on moving pixels than the
> VQ-VAE, so the weighted loss and the new decoder work for small sprites. On the held-out games, all
> autoencoders were much worse, four to eleven times higher error: they fit the training games.
>
> For the V-JEPA, something surprising happened: its validation loss went **up** after epoch two,
> while its features became richer. The effective rank, roughly how many of the 256 dimensions are
> actually used, grew from 1.4 to 57. The low early loss came from almost collapsed features, where
> every state looks the same and prediction is trivial. *[optional: The inverse accuracy also rose
> from 23 to 58 percent, against 13 percent chance.]*
>
> **Takeaway:** for a JEPA, a low loss does not mean good features.

### Slide 16: Downstream RL Performance (2:15)
**Cue:** how to read it → median: FSQ fine-tuned best · mean: NatureCNN (two games) · other VAEs lower · frozen low · held-out: NatureCNN · JEPA ≈ 0.
**Report:** §4.2 Downstream Reinforcement Learning, p. 10–13; Table 4, p. 11

> This is the main result, so let me first explain how to read the table. Each column is one agent:
> the NatureCNN from scratch, then each pre-trained encoder twice, frozen and fine-tuned, called
> "joint". Every number is a human-normalised score: 0 is random, 1 is human level. The first row is
> the median over the 11 pre-training games, the second row the mean, the third the median over the
> 4 new games.
>
> First row: the fine-tuned FSQ-VAE has the highest median, 1.07 against 0.74. In simple terms, it
> reached human level on six of the eleven games, the NatureCNN on five. It beat the NatureCNN on
> Breakout, Enduro, Qbert and Riverraid, and nearly tied on Pong and Seaquest. Even the fine-tuned
> VQ-VAE, at 0.85, was slightly above the NatureCNN.
>
> Second row: in the mean, the NatureCNN is ahead, 2.51 against 2.02. The mean is pulled up by a few
> games with very high scores. On DemonAttack and RoadRunner the NatureCNN scored about twice as
> much, and these two games alone explain the gap.
>
> The other autoencoders: the Standard VAE was the lowest on every game, and the VQ-VAE was below the
> FSQ everywhere except MsPacman, furthest behind on games with small, fast sprites.
>
> The frozen columns are all much lower: the frozen FSQ learned something in every game but reached
> less than a third of the fine-tuned median, and fine-tuning beat freezing on 13 of 15 games,
> Breakout by a factor of twelve.
>
> Third row: on new games, the NatureCNN trained directly on them stays ahead on three of four. And
> the V-JEPA is near zero everywhere, frozen or fine-tuned.
>
> **Takeaway:** pre-training helps as a starting point, but not as a frozen replacement.

### Slide 17: Learning Curves (0:35)
**Cue:** green solid = FSQ fine-tuned, close to blue; dashed = frozen, low.
**Report:** Fig. 4, p. 12

> The learning curves show the same picture over time, one panel per game. Blue is the NatureCNN,
> solid green the fine-tuned FSQ, which follows it closely or is above it. The dashed lines, the
> frozen encoders, rise slowly and stay low; the V-JEPA curves stay near the bottom.

### Slide 18: Sample Efficiency (0:45)
**Cue:** faster on 4 games by 1.4–2.6 M steps; frozen reaches 80 % on only 2 of 11.
**Report:** §4.2, *Sample Efficiency*, p. 10; Table 5, p. 12

> This table asks: how fast does each agent learn? It counts the million steps an agent needs until
> it first covers 50 or 80 percent of the NatureCNN's learning progress; a dash means never. The
> fine-tuned FSQ, FSQ-J, reached the 80 percent level 1.4 to 2.6 million steps earlier on Enduro,
> MsPacman, Qbert and Riverraid, and the 50 percent level faster on five games. The frozen FSQ,
> FSQ-F, reached the 80 percent level on only 2 of 11 games.
>
> **Takeaway:** a pre-trained start can save millions of steps; a frozen encoder rarely gets there.

### Slide 19: V-JEPA: Can the Features Be Fixed? (1:00)
**Cue:** 9,216 numbers, 18× the NatureCNN → pooled to 256 + 5× entropy bonus → no help → content, not size.
**Report:** §4.2, *V-JEPA*, p. 11–12; Table 6, p. 13; *Using the Encoder in PPO*, p. 8

> Why did the V-JEPA fail? The first hypothesis was the size of its feature. It outputs 36 tokens of
> 256 numbers, so 9,216 numbers per state: 18 times more than the NatureCNN. With a frozen encoder,
> only the policy heads learn, and their first layer alone needs about 4.7 million weights to read
> this feature, learned from the reward alone.
>
> So all 22 seen-game runs were repeated with the 36 tokens averaged into 256 numbers, which is also
> how the JEPA papers usually summarise their tokens. These runs also used separate encoders for the
> actor and the critic. And the frozen runs were repeated once more with a five times larger entropy
> bonus, which makes the policy explore more.
>
> The table shows that neither helped: the median went from 0.045 with the flat feature to 0.021
> pooled, and 0.007 pooled and fine-tuned, where the policy became almost deterministic. With the
> larger entropy bonus, the policy stayed random and the critic stopped learning: its explained
> variance dropped to 0.03.
>
> **Takeaway:** the problem is probably not the size of the feature, but the information in it.

### Slide 20: V-JEPA: World-Model Diagnostics (1:15)
**Cue:** copying looks good → baselines; no collapse, real dynamics, but the action barely matters; latents sorted by game.
**Report:** §4.3 World-Model Diagnostics, p. 13–14; Table 7, p. 14; Fig. 5b, p. 15

> So what is inside the model? The next state shares three of four frames with the current one, so
> simply **copying** the current latent already looks like a good prediction. The model was
> therefore compared with copying, with an untrained model, and with deliberately wrong actions, on
> 64 sequences of five steps.
>
> Three findings. First, no collapse: the effective rank is 52.5, against 2.5 for an untrained model.
> Second, it learned dynamics: it beats copying, and the gap grows from 0.013 after one step to 0.046
> after five steps, when input and target no longer share any frames. Third, the key: **the action
> barely matters**. A wrong action changes the prediction by only 0.006.
>
> The game token gets 1.8 times the uniform attention, but the encoder already recognises the game
> from the pixels: the t-SNE on the right shows that the latents are sorted by game. And on the
> held-out games, the model is worse than copying.

### Slide 21: Why the V-JEPA Fails with PPO (1:15)
**Cue:** predicts what usually happens, not what the agent changes.
**Report:** §4.4 Why the V-JEPA Failed with PPO, p. 14–15; Table 8, p. 14; Fig. 5a, p. 15

> The PCA on the left shows, for five states, the predicted next latent for every possible action.
> All the dots land on almost the same spot: the predictor hardly distinguishes between actions.
>
> The observations on the right: the predictor mostly ignores the action; the latents barely change
> from one step to the next, with a cosine similarity of 0.974; the latent space is sorted by game;
> and the model is worse on held-out games.
>
> My interpretation: a JEPA only has to predict what is **predictable**, so it can ignore details that
> have little effect on the latent error, like a small ball. But in Atari, exactly these details
> decide the right action. On top of that, 14 by 14 patches are coarse for small objects, and the
> expert data shows few alternative actions. So the critic could still read how good a state is, but
> the policy could not find the right action.
>
> **Takeaway:** the V-JEPA learned what usually happens in a game, not what the agent's actions
> change.

### Slide 22: Compute Cost and Energy (0:55)
**Cue:** pre-training ≈ one PPO run, paid once; ≈ 520 GPU-hours, 82 kWh in total; GPU mostly idle (125–210 W).
**Report:** §4.5 Compute Cost, Energy and Interpretability, p. 15–16; Table 9, p. 16

> For RQ4: the cost per run, measured from the GPU power logged in Weights & Biases, one H100 per
> run. Pre-training the FSQ-VAE took 1.9 GPU-hours and 0.7 kilowatt-hours, about one PPO run, and it
> is paid once for all games. The V-JEPA took 15.6 GPU-hours and 2 kilowatt-hours. A PPO run took 2.2
> to 4.5 GPU-hours, depending on the encoder. All reported experiments took about 520 GPU-hours and
> 82 kilowatt-hours.
>
> The GPU was mostly idle during PPO, because the Atari emulator runs on the CPU: it drew only 125 to
> 210 watts of its 700. *[optional: The rows marked with a star were not logged in Weights & Biases
> and are approximated.]*
>
> For interpretability, the autoencoders win: their decoder turns features back into images. The
> V-JEPA can only be inspected indirectly, with PCA, t-SNE and probes.

---

## Part 5: Conclusion (≈ 2.25 min)

### Slide 23: Conclusion (1:25)
**Cue:** one answer per RQ, then the one-sentence answer.
**Report:** §5 Conclusion, p. 16–17

> To answer the research questions. RQ1: the FSQ-VAE gave the most useful features when fine-tuned;
> it beat the Standard VAE on every game and the VQ-VAE on all but one. RQ2: fine-tuning worked much
> better than freezing. The fine-tuned FSQ had a higher median than the NatureCNN, 1.07 against
> 0.74, and learned faster on four games, but it had the better final score on only four of eleven.
> RQ3: there was no overall gain on new games; the FSQ was better only on Alien, and the NatureCNN
> led on the other three. RQ4: pre-training costs about one PPO run and is shared by all games;
> everything together took about 520 GPU-hours and 82 kilowatt-hours, and the FSQ decoder shows what
> the features contain.
>
> The V-JEPA learned how games usually evolve, but hardly what the agent's actions change.
>
> **So my answer is: a shared encoder is a useful starting point, but it cannot yet replace the
> NatureCNN when it is frozen.**

### Slide 24: Limitations and Future Work (0:45)
**Cue:** one seed, PPO not re-tuned; next: 3 seeds, smaller patches + action-relevant loss, probes first, FSQ world model, more games.
**Report:** §5.1 Limitations and Future Work, p. 17

> The main limitations are one seed per run and PPO settings that were not re-tuned for the
> pre-trained encoders. Next, I would repeat the main comparison with three seeds; give the V-JEPA
> smaller patches and a loss where the action matters more, such as multi-step prediction or a
> reward head; check the features first by predicting object positions, before running expensive
> PPO; and use the FSQ grid inside a world model and pre-train on more games.

### Slide 25: Thank You (0:10)

> Thank you for your attention. I am happy to take your questions.

---

## Key Numbers Cheat Sheet

| What | Number |
|---|---|
| Games | 11 pre-training + 4 held out (Alien, Asteroids, Atlantis, IceHockey) |
| Data | 50 k frame stacks/game (autoencoders); 250 k transitions/game, 2.75 M total (V-JEPA) |
| Median HNS, seen | NatureCNN 0.74 · FSQ fine-tuned **1.07** · FSQ frozen 0.31 · Res-VQ ft. 0.85 · Std-VAE ft. 0.36 · V-JEPA 0.05 / 0.04 |
| Mean HNS, seen | NatureCNN **2.51** · FSQ fine-tuned 2.02 |
| Human level (HNS ≥ 1) | FSQ fine-tuned on 6 of 11 games · NatureCNN on 5 of 11 |
| Mean gap explained by | DemonAttack (HNS ≈ 7.0 vs 3.7) and RoadRunner (≈ 5.5 vs 2.6) |
| Median HNS, held-out | NatureCNN **0.50** · FSQ fine-tuned 0.45 · FSQ frozen 0.21 |
| FSQ ft. beats NatureCNN | Breakout, Enduro, Qbert, Riverraid (≈ tie on Pong, Seaquest) |
| Fine-tuning vs. freezing (FSQ) | better on 13 of 15 games (Breakout 12×, DemonAttack 7×) |
| Sample efficiency | 80% level reached 1.4–2.6 M steps earlier on 4 games |
| FSQ | grid 4×21×21 = 1,764 values; levels [8,5,5,3] = 600 per cell; moving pixels 85% of loss |
| VQ-VAE | 21×21 grid, K = 512 codes, d = 64 |
| V-JEPA | 12.5 M params; 14×14 patches → 36 tokens × 256; EMA τ = 0.996; λ = 0.01; batch 512; 92 epochs, checkpoint epoch 92; 15.6 h |
| V-JEPA ablation (median HNS) | flat frozen 0.045 · pooled frozen 0.021 · pooled fine-tuned 0.007 · pooled + entropy 0.05: 0.015 |
| Scope | 15 games · 3.8 M expert samples · 5 encoders · 168 PPO runs (15 / 30 / 30 / 30 / 63) · 1.68 B steps · ≈ 10 k lines of Python |
| JEPA diagnostics | eff. rank 52.5 (untrained 2.5); wrong action Δcos 0.006; inverse acc. 55.9% (chance 12.9%) |
| Cost | FSQ pre-training 1.9 GPU-h / 0.7 kWh; V-JEPA 15.6 GPU-h / 2.0 kWh; PPO 2.2–4.5 GPU-h / 0.3–0.8 kWh per run; total ≈ 520 GPU-h, 82 kWh |
| Speed | steps/s: NatureCNN 976 · V-JEPA frozen 854 · V-JEPA fine-tuned 677 |

---

# Background for You: The Six Points Explained

These are for your own understanding. Each ends with a short version you can say if someone asks.

## 1. "One network plays 40–57 games": what does *one network* mean?

A neural network is a fixed architecture plus one set of learned numbers, the **weights**. The
usual Atari setup trains a **separate** network for every game: the Pong agent has its own weights,
the Breakout agent has other weights, and the Pong agent cannot play Breakout at all.

"One network" means a **single set of weights for all games**. You give it a frame from any of the
40–57 games, and the same network outputs the action. It was trained on all games together, so
the same weights had to learn all of them.

- **PopArt** (Hessel et al., 2019): one agent trained with RL on all 57 Atari games at once. The
  main difficulty is that rewards have very different sizes between games (Pong gives ±1, other games
  give thousands of points), so PopArt automatically normalises the learning targets of each game so
  that no game dominates the learning.
- **Multi-Game Decision Transformer** (Lee et al., 2022): one transformer trained offline on
  recorded gameplay of up to 46 games, and it adapts quickly to new games by fine-tuning.
- **Scaled Q-Learning** (Kumar et al., 2023): one network trained offline on 40 games.

**How this differs from your work:** they share *everything*, including the part that decides the
action, and they use rewards. You share only the encoder (the part that "sees"), train it without
rewards, and still train a separate policy for each game.

> *Short version:* "One network means one set of weights for all games: the same model looks at
> Pong or Breakout and chooses the action. They share the whole agent and use rewards; I share only
> the encoder and use no rewards."

## 2. "One network copies the experts of several games": what does that mean?

This is called **policy distillation**, and it works like one student learning from several
teachers.

1. First, a separate **expert** (teacher) network is trained for each game, for example a DQN that
   is very good at Pong, another for Breakout, and so on.
2. Then a single **student** network is trained by supervised learning: it is shown frames from all
   games, and for each frame its target is the teacher's output. On a Pong frame it should choose
   what the Pong expert would choose; on a Breakout frame what the Breakout expert would choose.
3. In the end, one student network can play all these games, close to the experts' level.

**Actor-Mimic** (Parisotto et al., 2016) does exactly this and then goes one step further: it uses
the student's learned features as the starting point for a **new** game, which made learning faster.

**How this differs from your work:** you also use experts, but only to **collect data** (the frames
they produce while playing). Your encoder never learns to copy their decisions. It learns only by
rebuilding the frames, or by predicting the next state. Copying experts is "task-specific"
supervision; your pre-training is self-supervised.

> *Short version:* "They train one student network to imitate the expert of every game: on a Pong
> frame it must choose what the Pong expert would choose. I use the experts only to record
> gameplay; my encoder never copies their decisions."

## 3. "Reuse the features of one game for the next": in simple terms

**Features** are what the early layers of a network have learned to detect: edges, shapes,
sprites, the ball, the player. A network trained on Pong has learned to "see" a moving ball.

**Reusing the features** means: when you start learning a new game, you don't start with a blank
network that has to learn to see from zero. You start from the network that was already trained on
another game, so it can already detect shapes and movement, and only has to learn what is new.

- **Fine-tuning:** copy the trained network as the starting point for the new game.
- **Progressive Networks** (Rusu et al., 2016): keep the old game's network fixed, add a new
  network for the new game, and connect them so the new network can read the old features. This way
  nothing that was learned for the old game is forgotten.
- **Farebrother et al. (2018)** showed that a DQN trained on one version (mode) of a game
  over-specialises to it, but its features can be fine-tuned for other versions of the same game.

**How this differs from your work:** they transfer from one game to the next, step by step. You
train one encoder on eleven games **at once**, so it should contain what is common to all of them.

> *Short version:* "A network trained on one game already knows how to see the screen: shapes,
> sprites, motion. Transfer means starting the next game from those weights instead of from zero.
> They do this game by game; I train one encoder on eleven games at once."

## 4. Why these autoencoders, including FSQ? What advantage did you expect for this task?

**Why reconstruction at all?**
- To rebuild a frame, the encoder **cannot ignore anything visible**, including small sprites. A
  predictive model like the JEPA may skip details; a reconstruction model is forced to keep them.
- The decoder lets you **look inside**: you can see what the features contain (RQ4).
- It is the most established self-supervised objective, so it is a solid reference point.

**Why three variants?** They form a ladder; each step fixes a weakness of the previous one:

| Model | What you expected it to give | Its weakness |
|---|---|---|
| **Standard VAE** | Fairest baseline: same layers as the NatureCNN, so any difference comes from pre-training alone. Compact 256-number feature. | One vector for the whole frame; one cell = 12×12 pixels, bigger than a ball; the KL term blurs details. |
| **Residual VQ-VAE** | **Spatial 21×21 grid**: each cell = 4×4 pixels, so object positions are kept. **Discrete codes** = a "vocabulary" of 512 visual patterns, natural for Atari, which is built from a few repeated sprites and colours; discrete latents already work in Atari world models (DreamerV2, IRIS). | The codebook must be learned; some codes are never used (codebook collapse); extra losses and settings. |
| **FSQ-VAE** | Same advantages as the VQ-VAE (grid + discrete values) **without a codebook**: each channel is simply rounded to fixed levels, so all codes are used, there is nothing to learn in the bottleneck, fewer settings, stable training. Small and cheap (0.84 M parameters, 1.9 GPU-hours). | Rounding limits how finely each value can be stored; still a pixel objective, so it spends effort on the background. |

**Plus the loss on moving pixels:** in most games under 4 % of the pixels move, so a normal loss
can almost ignore the ball. The FSQ-VAE's loss always gives moving pixels 85 % of the weight.

**Did it work?** Yes, along the ladder: Standard VAE < Residual VQ-VAE < FSQ-VAE on almost every
game, and the FSQ-VAE had 3× less error on moving pixels than the VQ-VAE.

> *Short version:* "Reconstruction forces the encoder to keep every visible object, and the
> decoder lets me see what it keeps. I went from coarse to fine: the VAE is the fair baseline with
> NatureCNN layers; the VQ-VAE adds a spatial grid and a discrete vocabulary that fits Atari's
> repeated sprites; FSQ keeps those advantages but removes the learned codebook, so there are no
> unused codes and fewer settings. The results followed the same order."

## 5. The Downstream RL Performance slide, in detail

**What the slide measures:** after pre-training, every encoder is put into a PPO agent, the agent
plays each game for 10 million steps, and the score at the end of training is recorded.

**How to read the table:**
- **Columns** = agents. *NatureCNN scratch* is the baseline. Then each pre-trained encoder appears
  twice: *frzn* (frozen: encoder locked, only the policy learns) and *joint* (fine-tuned: the encoder
  keeps learning together with the policy).
- **Numbers** = human-normalised score (HNS): 0 = random play, 1 = human level, above 1 = better
  than a human. This makes Pong (scores −21 to 21) comparable with RoadRunner (scores in the tens of
  thousands).
- **Row 1, median over 11 seen games:** sort the 11 game scores and take the middle one (the 6th).
  Half the games are above it. It is robust: one extreme game cannot move it.
- **Row 2, mean over 11 seen games:** the average. One game with a huge score pulls it up.
- **Row 3, median over 4 held-out games:** the same, but on games the encoder never saw in
  pre-training.

**What it tells you, line by line:**
1. **Fine-tuned FSQ has the best median (1.07 vs 0.74).** Concretely: the fine-tuned FSQ reaches
   human level (≥ 1) on 6 of 11 games, the NatureCNN on 5. It wins on Breakout, Enduro, Qbert and
   Riverraid, and ties on Pong and Seaquest. Even the fine-tuned VQ-VAE (0.85) is above the
   NatureCNN.
2. **The NatureCNN has the best mean (2.51 vs 2.02)**, because of a few games with very large
   normalised scores. On DemonAttack (HNS ≈ 7.0 vs 3.7) and RoadRunner (≈ 5.5 vs 2.6) the NatureCNN
   scored about twice as much, and these two games alone are larger than the whole gap in the mean.
   (Breakout has the largest HNS of all, about 11, because the human reference score is low, but there
   both agents are similar.)
3. **Frozen encoders are far behind.** Frozen FSQ: 0.31, less than a third of fine-tuned. Fine-tuning
   beat freezing on 13 of 15 games, e.g. Breakout 12× better. So the pre-trained features alone are
   not enough; they need the reward to adapt.
4. **The ladder holds:** Standard VAE lowest, then VQ-VAE, then FSQ, frozen and fine-tuned.
5. **Held-out games (row 3):** the NatureCNN trained directly on these games (0.50) stays slightly
   ahead of the fine-tuned FSQ (0.45); the FSQ was better only on Alien. Pre-training on 11 games did
   not give an advantage on new games.
6. **V-JEPA ≈ 0.05 everywhere:** PPO could barely use its features, frozen or fine-tuned.

**The one-sentence meaning:** pre-training helps as a **starting point** (fine-tuned), but not as a
**frozen replacement** for the NatureCNN, and not yet for new games.

> *Why report the median if the NatureCNN wins the mean?* Both are shown honestly. The median says
> "on a typical game"; the mean is dominated by two or three games with very large normalised scores.
> With one seed, the median is the more robust summary.

## 6. "The 9,216-dimensional feature is too big for PPO": why did we suspect this?

**Where the number comes from:** the V-JEPA encoder outputs 36 tokens (one per 14×14 patch), each
with 256 numbers, and they were flattened into one feature: 36 × 256 = **9,216 numbers**. For
comparison: NatureCNN 512 numbers, FSQ-VAE 1,764.

**Why that could be a problem:**
1. **The policy has to learn to read it from the reward alone.** With a frozen encoder, only the
   actor and critic heads learn. Their first layer (9,216 → 512) alone has about **4.7 million
   weights**, almost three times the size of the whole NatureCNN (1.68 M). Learning that many weights
   from a weak, noisy reward signal is slow.
2. **Most of the 9,216 numbers are redundant.** Neighbouring patches and consecutive frames give very
   similar tokens (the features barely change from step to step), so the useful information is
   hidden in a few directions among thousands.
3. **The JEPA papers themselves pool the tokens.** In I-JEPA and V-JEPA, the tokens are usually
   averaged or pooled into one compact vector before a classifier reads them. So pooling to 256 was
   the natural thing to try.
4. **A symptom pointed there:** the critic could predict the returns well (explained variance up to
   0.97), but the policy did not improve, which looked like the policy could not find the few useful
   directions in a very large input.

**What was tested:** the 36 tokens were averaged into **256 numbers** (all 22 seen-game runs, frozen
and fine-tuned), and the frozen runs were also repeated with a **5× larger entropy bonus** so that
the policy explores more instead of settling early.

**Result:** neither helped (median HNS 0.045 flat → 0.021 pooled → 0.015 with more exploration). So
the size was not the problem; the **content** was. This led to the diagnostics, which showed that the
predictor ignores the action.

> *Short version:* "9,216 numbers is 18 times the NatureCNN's feature, and a frozen policy would need
> almost 5 million weights just to read it, learned only from the reward. The JEPA papers also pool
> their tokens. So pooling to 256 was the obvious first test. It didn't help, which showed the
> problem is what the feature contains, not its size."

---

# Hard Questions and Answers

Answer in two steps: give the short answer first, then one supporting fact. If you don't know,
say so and say how you would find out. That is better than guessing.

## A. Choice of architectures

**Q1. Why exactly these four models? Why not contrastive methods like CURL or ATC, or masked autoencoders?**
> The goal was to compare the two main families of self-supervised objectives under identical
> conditions: reconstruction and latent prediction. Within reconstruction, the three autoencoders form
> a ladder: a coarse continuous vector (VAE), a discrete spatial grid with a learned codebook
> (VQ-VAE), and a discrete grid with fixed rounding (FSQ). The V-JEPA represents the
> reconstruction-free, predictive family, which is currently the most discussed direction for world
> models. Contrastive methods like ATC, and a broad set of objectives in the Atari Pre-training
> Benchmark, were already studied on Atari, so the new part here was the discrete bottlenecks and the
> action-conditioned JEPA. Adding ATC as a further baseline is a sensible next step.

**Report:** §3.4 first paragraph (why these model families), p. 5. ATC and Atari-PB are cited in neither the report nor the slides

**Q2. Why FSQ instead of the standard VQ-VAE?**
> FSQ keeps the advantages of a discrete grid but avoids the known problems of vector quantisation:
> codebook collapse, unused codes, and the extra commitment and codebook losses. Each channel is
> simply rounded to a fixed number of levels, so there is nothing to learn in the bottleneck and fewer
> hyper-parameters. In practice it also gave the lowest reconstruction error, three times lower error
> on moving pixels than the VQ-VAE, and the best PPO results.

**Report:** §3.4 *FSQ-VAE*, p. 6; reconstruction numbers in §4.1, p. 10

**Q3. Why the levels [8, 5, 5, 3]?**
> The FSQ paper recommends level sets for different codebook sizes, for example [8, 6, 5] for about
> 2^8 codes and [8, 5, 5, 5] for about 2^10 codes. [8, 5, 5, 3] gives 600 possible values per cell,
> which lies between the two and is close to the 512 codes of the VQ-VAE, so the two discrete models
> have a similar capacity. The levels were not tuned further.
> *(Check that this matches your own reason for choosing these levels.)*

**Report:** §3.4 *FSQ-VAE*, p. 6; Table 3, *Reference setting*, row *Bottleneck / target*, p. 9

**Q4. Why a spatial grid (21×21) instead of a single vector?**
> Atari objects are tiny. In the NatureCNN, one cell of the final 7×7 map covers 12×12 pixels, larger
> than a ball. A 21×21 grid keeps where objects are, which matters for control. The results support
> this: the Standard VAE with a single vector was the weakest model on every game.

**Report:** §3.4 *Standard VAE* and *Residual VQ-VAE*, p. 6; §4.2 *Standard VAE and Residual VQ-VAE*, p. 11

**Q5. Why did you choose a JEPA at all?**
> Two reasons. First, it does not need to reconstruct pixels, so in principle it does not waste
> capacity on irrelevant background. Second, by conditioning the predictor on the action, it should
> learn what the agent controls, which is exactly what a policy needs. JEPAs are also a central
> proposal for world models (LeCun's position paper, I-JEPA, V-JEPA). The project tested whether that
> promise holds for Atari control.

**Report:** §3.5, first paragraph, p. 6

**Q5b. How is your JEPA different from I-JEPA and V-JEPA?**
> I-JEPA and V-JEPA learn from images and videos. They hide regions of the input and predict the
> representations of the hidden regions; there are no actions. In this project, nothing is masked:
> the model predicts the representation of the **next** frame stack, and the predictor also gets the
> action and the game as extra tokens. To train one model on eleven games, all actions are mapped to
> one set of 18 Atari actions, and each game gets an embedding, both ideas from TD-MPC2. An inverse
> head that predicts the action was added as well.

**Report:** §2 *Joint-Embedding Predictive Architectures*, p. 2; §3.5, p. 6–7; Fig. 3, p. 7

**Q6. Why a transformer for the JEPA and a CNN for the autoencoders? Isn't that unfair?**
> The JEPA follows the I-JEPA and V-JEPA papers, which use vision transformers; the token structure
> also makes it easy to add the action and game tokens. The autoencoders follow their own papers,
> which use CNNs. So each model is used in its standard form. A fully controlled comparison with the
> same backbone would be cleaner and is a fair point for future work.

**Report:** §3.6 *Pre-Training* (architectures follow the references), p. 8; Table 3, p. 9

**Q7. Why 14×14 patches? Aren't they too coarse?**
> 84 divided by 14 gives a 6×6 grid of 36 tokens, which keeps the transformer affordable; patch size
> 14 is also standard in the JEPA papers. But yes, the diagnostics suggest that this is too coarse for
> small sprites, and smaller patches are the first thing I would change.

**Report:** §3.5 (14×14 patches, 36 tokens), p. 6; Table 8 row *Latents change slowly*, p. 14; §5.1, p. 17

**Q8. Why the EMA target with stop-gradient instead of VICReg or a contrastive loss?**
> It is the standard anti-collapse mechanism of I-JEPA, V-JEPA and BYOL, and it needs no negatives or
> extra regularisation weights. The diagnostics show it worked: no collapse, effective rank 52.5. A
> VICReg-style regulariser is an alternative but would not fix the main problem, which was the action
> being ignored, not collapse.

**Report:** §3.5 (EMA target encoder), p. 6; *Training Step*, p. 8; Table 7 (effective rank), p. 14

**Q9. Why the game embedding, if the encoder can already recognise the game from pixels?**
> The embedding was taken from TD-MPC2 to give the model explicit task information and an "unknown"
> embedding for new games. It turned out to be partly redundant: the latents cluster by game even
> without it. That is one of the findings.

**Report:** §3.5 (game embedding), p. 7; §4.3 (latents cluster by game), p. 13–14; Fig. 5b, p. 15

**Q10. Why SimNorm?**
> SimNorm splits the latent into groups and applies a softmax to each group. This keeps the latents
> bounded and makes collapse to a constant harder. It comes from TD-MPC2, which trains one model on
> many tasks, the same situation as here.

**Report:** §3.5 (SimNorm), p. 7

## B. Fairness of the comparison

**Q10b. The Atari Pre-training Benchmark already compares pre-training objectives on 50 games. What is new here?**
> Atari-PB is the closest study, and it found that task-agnostic objectives, such as learning
> objects and temporal dynamics, generalise better. This project is complementary: it compares
> discrete bottlenecks (VQ-VAE, FSQ) with an action- and game-conditioned JEPA, uses a standard PPO
> agent for 10 M steps, evaluates every encoder both frozen and fine-tuned, and measures compute
> cost and energy. Its finding that the JEPA learned dynamics but not what the actions change fits
> well with Atari-PB's distinction between task-agnostic and task-relevant features.

**Report:** Atari-PB is cited in neither the report nor the slides; closest is §2 *Multi-Game Agents and Transfer*, p. 2

**Q10c. Actor-Mimic already trained one network on several games and transferred it to new games. What is different?**
> Actor-Mimic trains the network to imitate the policies of expert agents, so it learns from the
> experts' actions, which is task-specific supervision. Here, the experts only provide the data: the
> encoder learns without rewards and without copying any policy, from reconstruction or latent
> prediction, and PPO learns the policy for each game. The question here is whether such
> self-supervised features are enough on their own.

**Report:** §2 *Multi-Game Agents and Transfer*, p. 2

**Q11. Is the FSQ gain really from pre-training, or just from a different architecture with a finer grid?**
> That is a fair question, and the honest answer is that the two effects cannot be fully separated,
> because there is no run of the FSQ encoder trained from scratch. Two things suggest pre-training
> matters. First, the fine-tuned FSQ learned faster early on, on four games, which is typical of a
> good initialisation. Second, the frozen FSQ encoder, which never saw a reward, still learned in
> every game, so the features contain useful information. Training the FSQ architecture from scratch
> is the missing control I would add first.

**Report:** not discussed in the report (missing control); supporting evidence in Table 5, p. 12 and §4.2 *FSQ-VAE, Frozen*, p. 11

**Q12. The Standard VAE uses the same layers as the NatureCNN but scores lower even when fine-tuned (0.36 vs 0.74). Did pre-training hurt?**
> Yes, in that case it seems so. The VAE compresses everything into 256 numbers with a KL penalty,
> which produces smooth, blurry features, and starting PPO from there was worse than starting from
> random weights. This shows that pre-training is not automatically helpful; the objective and the
> bottleneck matter. (Note: these Std-VAE scores were only recorded in rounded form.)

**Report:** §3.4 *Standard VAE*, p. 6; §4.2 *Standard VAE and Residual VQ-VAE*, p. 11; Table 4, p. 11

**Q13. The encoders have different feature sizes (512, 1,764, 9,216). Does that explain the results?**
> Not alone. The frozen FSQ with 1,764 values was weaker than the 512-value NatureCNN, and the
> V-JEPA did not improve when the 9,216 values were pooled to 256. So the size of the feature is not
> the main factor; its content is.

**Report:** §3.4 (FSQ feature, 1,764 values), p. 6; *Using the Encoder in PPO* (9,216 / 256), p. 8; Table 6, p. 13

**Q14. PPO hyper-parameters were tuned for the NatureCNN. Isn't that unfair to the pre-trained encoders?**
> Yes, that is a limitation. The standard RL-Zoo settings were used for all agents so that the
> comparison has no tuning advantage on either side. For example, fine-tuning a transformer with the
> same learning rate as a small CNN was probably too aggressive for the V-JEPA: on Pong it reached +8
> and then collapsed. A lower encoder learning rate is a likely improvement.

**Report:** §3.6 *PPO Training and Evaluation*, p. 8; §4.2 *V-JEPA* (Pong +8.4 then −15), p. 11; §4.4 last paragraph, p. 15; §5.1, p. 17

**Q15. You used only one seed. How can you trust the results?**
> Single-seed RL results are noisy, so small differences should not be over-interpreted; the report
> says this explicitly. The large effects are robust across games: fine-tuning beat freezing on 13 of
> 15 games, and the V-JEPA was near zero everywhere. Close comparisons, like FSQ versus NatureCNN on
> Pong or Seaquest, would need more seeds. Repeating the main comparison with three seeds is the
> first item of future work.

**Report:** §3.6, last sentence, p. 9; §5.1, p. 17

**Q16. Why the median HNS and not the mean, or the IQM?**
> The mean is dominated by a few games with very large normalised scores, such as DemonAttack. The
> median is robust to such outliers. Both are reported: FSQ fine-tuned wins on the median, NatureCNN
> on the mean, and the slides say this openly. With more seeds, the interquartile mean recommended by
> Agarwal et al. would be the better choice.

**Report:** §3.6 *PPO Training and Evaluation* (HNS, median), p. 8; Table 4, p. 11

**Q17. Why use the training return instead of separate evaluation episodes?**
> The mean return over the last 10% of training is the usual measure in the PPO and RL-Zoo
> literature, so it is comparable to published baselines. For the V-JEPA ablation, the final policy
> was also tested on 30 episodes, and the conclusions were the same.

**Report:** §3.6 (last 10% of training), p. 8; Table 6 (*test* column, 30 episodes), p. 13

## C. Data

**Q18. Why expert data and not random or exploratory data?**
> Random play rarely leaves the first screen, so it would not show the later parts of the games. The
> expert data covers the whole game. The downside is low action diversity: in RoadRunner the expert
> uses one action 67% of the time. That probably hurt the V-JEPA, because it rarely saw what other
> actions would change. A mix of expert and exploratory data would be better.

**Report:** §3.2 Data Collection, p. 3; Table 1 (*Top act. %*), p. 5; Table 8 row *Little action variety in the data*, p. 14

**Q19. Isn't there a distribution shift? The PPO agent early in training sees very different states from the expert.**
> Yes. Early in training, the PPO agent sees states the expert never visits, for example losing
> early. This is one reason why frozen encoders struggle and why fine-tuning helps: it lets the
> encoder adapt to the agent's own state distribution.

**Report:** not discussed in the report

**Q20. Why these four held-out games?**
> They cover different genres (maze, free-roaming shooter, fixed defence, sports) that are not in
> the training set, and pre-trained experts were available for them. They are also visually quite
> different, for example Asteroids is 95% black, so they are a hard test of generalisation.

**Report:** §3.3 and Table 1 (game types), p. 3–5

## D. Results and interpretation

**Q21. Why do frozen encoders work so much worse than fine-tuned ones?**
> A frozen encoder must already contain every small detail that matters in every game. A reward-free
> objective does not force that: reconstruction spends capacity on large areas like the background,
> and the details that decide the action, like a bullet, are only a few pixels. Fine-tuning lets the
> reward signal reshape the features.

**Report:** §4.2 *FSQ-VAE, Frozen*, p. 11; §5 Conclusion (RQ2), p. 16

**Q22. Why did nothing generalise to the held-out games?**
> Eleven games is a small and specific training distribution, and the held-out games look different.
> The reconstruction errors on held-out games were four to eleven times higher, and the V-JEPA was
> worse than copying there. A universal extractor probably needs many more games, closer to the 40+
> games used in multi-game agents.

**Report:** §4.1 (held-out errors 4–11× higher), p. 10; §4.3 (held-out worse than copying), p. 14; §5 (RQ3), p. 16

**Q23. The V-JEPA's validation loss went up during training. Why did you still trust the model?**
> Because a low JEPA loss can mean collapse: if all latents are almost the same, prediction is
> trivial. The effective rank showed that the early, low-loss model was nearly collapsed (rank 1.4).
> As training continued, the rank grew to 57 and the inverse-dynamics accuracy rose from 23% to 58%.
> So the checkpoint was selected on the total loss including the inverse term, which kept improving.

**Report:** §4.1 Pre-Training, p. 10; §3.6 *Pre-Training* (checkpoint choice), p. 8

**Q24. What is the effective rank?**
> It measures how many dimensions of the latent space are actually used. It is computed from the
> singular values of a batch of latents: normalise them to a distribution, take the entropy, and
> exponentiate. A value near 1 means all latents point in one direction, so the space has collapsed.

**Report:** §4.1 (explained in words), p. 10; Table 7, p. 14

**Q25. Why did the predictor ignore the action?**
> Three likely reasons. First, consecutive frame stacks share three of four frames, so the next state
> is mostly predictable without the action. Second, the expert data contains few alternative actions,
> so there is little signal about what other actions would change. Third, the inverse-dynamics loss
> trains the encoder but not the predictor, because it uses the target latent and not the
> prediction. Multi-step prediction, where the effects of actions add up, or a reward head would give
> the action more weight.

**Report:** §4.4, p. 14–15; Table 8, p. 14; Table 2 and *Training Step* (inverse loss does not train the predictor), p. 8

**Q26. The frozen V-JEPA critic had high explained variance. Why could the policy still not learn?**
> The critic only needs to know how good a state is, which is a smooth property, like how far the
> game has progressed. The policy needs to know which action is best, which often depends on a few
> pixels, like the exact ball position. The V-JEPA features contained the first kind of information
> but not the second.

**Report:** §4.2 *V-JEPA*, p. 11; Table 8 row *State, not action, information*, p. 14; §4.4, p. 15

**Q27. Is the V-JEPA result a failure of JEPAs in general, or of your implementation?**
> It is a result for this setting: a small model, one-step prediction, coarse patches and expert data.
> The model itself behaved as intended: no collapse and real dynamics. So the result does not show
> that JEPAs cannot work for control, but it shows that the standard one-step JEPA objective does not
> automatically produce action-relevant features. This matches Sobal et al. (2022), who found that
> JEPAs focus on slowly changing features (this paper is not cited in the report).

**Report:** §4.3, p. 13–14; §4.4, p. 14–15

## E. Cost and interpretability

**Q28. How was the energy measured, and what is missing?**
> W&B recorded the GPU power every few seconds, and the energy is the integral over each run. It
> covers the GPU only, not the CPU, memory or cooling. Since the Atari emulator runs on the CPU, the
> true total is higher. Some runs were not logged; their values are approximated from similar runs
> and marked as approximate.

**Report:** §4.5, first paragraph, p. 15; Table 9 and its caption (approximate values), p. 16

**Q29. The GPU was only 10–22% utilised. Isn't that wasteful?**
> Yes. PPO on Atari is limited by the CPU emulator, not by the GPU. Running more environments in
> parallel, or several runs per GPU, would use the hardware much better. This is a useful practical
> lesson.

**Report:** §4.5 *Energy*, p. 16

**Q30. Is pre-training worth its cost?**
> For the FSQ-VAE, pre-training cost about as much as one PPO run (1.9 GPU-hours), it is paid once,
> and the encoder can be reused for all games. On four games it saved 1.4 to 2.6 million steps. So
> as an initialisation it is cheap. For the V-JEPA, 15.6 GPU-hours did not pay off.

**Report:** §4.5 *Compute Time*, p. 15; Table 9, p. 16; Table 5, p. 12

**Q31. What do you mean by "interpretable"?**
> For the autoencoders, the decoder turns a feature back into an image, so you can directly see what
> the encoder kept and what it lost, for example a missing ball. The V-JEPA has no decoder, so its
> latents can only be studied indirectly, with PCA, t-SNE and the diagnostics.

**Report:** §4.5 *Interpretability*, p. 16

## F. Big-picture questions

**Q32. Why not use the world model for planning, like Dreamer or MuZero?**
> The research question was about representations: can a reward-free pre-trained encoder replace
> the NatureCNN in a standard model-free agent? Keeping the agent model-free isolates the effect of
> the representation. Using the learned model for planning is a different question and a natural
> next step, for example using the FSQ grid as the state of a world model.

**Report:** §2 *Multi-Task World Models*, p. 3; §5.1 (FSQ grid in a world model), p. 17

**Q33. Why PPO and not DQN or a more sample-efficient algorithm?**
> PPO with the RL-Zoo settings is a standard and robust baseline for Atari, it runs many environments
> in parallel, and the same settings work for every game. That makes it a good fixed test bed for
> comparing encoders.

**Report:** §3.6 *PPO Training and Evaluation*, p. 8

**Q34. So, is a universal feature extractor for Atari possible?**
> Partly. A shared encoder already works as a good starting point: fine-tuned, it matched or beat the
> NatureCNN on six of eleven games and learned faster on four. But as a frozen, plug-in replacement,
> it does not work yet. That is why the title says "towards".

**Report:** §5 Conclusion, p. 16–17

**Q35. If you had three more months, what would you do?**
> First, three seeds and an FSQ-from-scratch control, to make the main result solid. Second, a V-JEPA
> with smaller patches and multi-step prediction or a reward head, so that actions matter. Third,
> check the features with linear probes for object positions before running expensive PPO. And
> finally, pre-train on many more games.

**Report:** §5.1 Limitations and Future Work, p. 17

**Q36. What was the most important thing you learned?**
> That a good-looking loss curve is not enough. For the V-JEPA, the loss rose while the model got
> better, and a low loss could even mean collapse. Simple checks, like comparing with copying the
> current latent or giving wrong actions, were necessary to understand what the model actually
> learned.

**Report:** §5 Conclusion, third paragraph, p. 17; §4.1, p. 10; §4.3, p. 13–14
