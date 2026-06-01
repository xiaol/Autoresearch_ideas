# Neural Geometry, Manifold Steering, and RWKV

TTS provider: MiniMax

Model: speech-2.8-hd

Host A voice: English_captivating_female1

Host B voice: English_CaptivatingStoryteller

Source article: https://www.goodfire.ai/research/the-world-inside-neural-networks

Source paper: arXiv:2605.05115, Steering Along Manifolds to Control Neural Networks

## 0:00-0:40

A: (breath) Neural networks are not just lookup tables. Inside a trained model, hidden activations often arrange themselves into geometry that mirrors structure in the world. <#0.4#> Goodfire's article, The World Inside Neural Networks, shows this with concepts like days, months, colors, locations, and other ordered ideas.

B: (chuckle) So the claim is that the model has an internal map, not just isolated facts.

A: Exactly. This video explains the original manifold-steering idea, then shows our RWKV reproduction.

## 0:40-1:20

A: (breath) First we need two spaces. Activation space is where the model's hidden vectors live. For one prompt, at one layer, the model produces a long vector. <#0.3#> That vector is a point in activation space.

B: (chuckle) And behavior space?

A: Behavior space is what the model does after we patch that point. In our experiments, behavior means the output probability distribution over concept labels, like Monday through Sunday, or January through December. So behavior space is not the whole layer output. It is the final answer pattern induced by the patched hidden state.

## 1:20-1:55

A: Now look at cyclic concepts. Monday, Tuesday, Wednesday, and the rest are not random labels. <#0.3#> They live on a loop. If the model understands the cycle, the hidden states for those labels may also form a loop-like curve.

B: (breath) Then a straight line between Monday and Thursday might cut through the middle of the loop.

A: Right. That straight chord can pass through hidden states the model rarely uses for this concept. Manifold steering asks for a different path: move along the learned curve.

## 1:55-2:34

A: (breath) The original method is simple in outline. First, collect activations from many prompts. Second, group them by concept label and estimate the concept centroids. Third, fit a smooth manifold through those centroids. Fourth, patch hidden states along either the curved manifold path or the straight-line baseline. <#0.4#> Finally, measure what the model outputs.

B: (chuckle) So we are not only drawing a pretty curve.

A: Exactly. The causal question is whether moving on the activation manifold also moves naturally in behavior space.

## 2:34-3:11

A: (breath) Here is how to read the GIFs. The connected labeled points are the model's natural concept outputs, projected into behavior space. The black square is manifold steering. It follows the fitted activation curve. The gray dot is linear steering. It uses the straight chord between the same start and end hidden states.

B: (breath) If the square tracks the labeled curve, and the dot cuts away from it, the curved intervention is preserving more natural output behavior.

A: That is the key visual test.

## 3:11-3:47

A: This is our RWKV weekday run. RWKV is state based, so the patch location is not a transformer residual stream in the exact architectural sense. <#0.4#> We patch the last-token block output after the selected RWKV block's residual updates. That is the closest analogue to the transformer block output used in the original work.

B: (chuckle) The square and dot share endpoints here?

A: Yes. We corrected the reproduction so both methods start and end at identical hidden states.

## 3:47-4:21

A: (breath) Now compare a small transformer, Qwen3.5 0.8B, under the same matched-endpoint protocol. This is useful because the original work was transformer-centered, while our question was whether a state model can show the same activation-to-behavior geometry.

B: So the comparison is architecture first: RWKV versus transformer, with the task and endpoint rule held fixed. <#0.3#>

A: Exactly.

## 4:21-4:55

A: We also tested months. Months are another cyclic concept, but with twelve points instead of seven. <#0.3#> The important detail is the same: the square is the manifold path, the gray dot is the straight path, and both runs now share identical endpoints.

B: (breath) If the endpoint is not matched, the final behavior point could differ just because the target hidden state differed.

A: Correct. Endpoint matching removes that confound.

## 4:55-5:37

A: (breath) The table summarizes the matched runs. For weekdays, RWKV has a manifold isometry correlation of 0.949, compared with 0.891 for the linear distance. Qwen has 0.857 for manifold distance and 0.537 for linear distance.

B: (chuckle) What does that correlation mean?

A: It compares distances in activation geometry with distances in behavior geometry. A high value means that as we move farther along the internal manifold, the model's output distribution also moves farther along the behavior manifold.

## 5:37-6:21

A: (breath) For months, RWKV gives a very high manifold correlation, 0.989, while the linear version is only 0.210. Qwen is also stronger for manifold distance than linear distance, 0.915 versus 0.744.

B: (sighs) But the table also says the tiny models are at chance accuracy.

A: (sighs) Yes. This is not a claim that these small models solve weekday or month arithmetic. It is a geometry and intervention sanity check. The structure can be visible before the model is highly competent at the explicit task.

## 6:21-7:01

A: (breath) The RWKV-specific lesson is about where the original transformer idea transfers and where it needs care. In a transformer, patching usually targets the residual stream around a layer. In RWKV, information flows through recurrent state and block outputs, so we patch the last-token block output after time-mix and channel-mix residual updates.

B: And matched endpoints were crucial. <#0.4#>

A: Yes. Without endpoint matching, a behavior-space difference could be caused by the endpoints, not by the path between them.

## 7:01-7:41

A: (breath) The practical takeaway is this. Linear steering asks: which direction should I push the activation? Manifold steering asks: which path stays inside the model's natural concept geometry?

B: (chuckle) And our extra result is that this question is not transformer-only.

A: (breath) The first RWKV version suggests the same activation-to-behavior test can be applied to state-based models. The next step is to scale the models and use stronger tasks, so the geometry result and the capability result can be studied together.
