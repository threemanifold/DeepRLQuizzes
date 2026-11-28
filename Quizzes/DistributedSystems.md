# Distributed Training and Inference — Questions

## 1. Replicated data parallelism

Train a **7B-parameter model** on **8 GPUs with 80 GB each**, using replicated data parallelism and Adam. Each parameter requires:

| State | Bytes |
|---|---:|
| BF16 weight | 2 |
| BF16 gradient | 2 |
| FP32 master weight | 4 |
| Two FP32 optimizer moments | 8 |

Each GPU processes **2 sequences per microbatch**, accumulating **4 microbatches per optimizer update**. Use decimal GB and ignore activations and temporary buffers.

1. What does each GPU store and what data does it process? What must GPUs communicate? **Name the collective**, explain its reduction, and explain why replicas remain synchronized.
2. Calculate the memory required per GPU. Does the model fit, and why does adding replicas help or fail to help? Assuming sufficient memory, what is the effective global batch size?
3. Compare synchronizing the accumulated gradient buffer after every microbatch with synchronizing only after all four. Assuming the collective averages across GPUs and each microbatch contributes its mean gradient, derive whether both strategies produce the same update. What scaling is needed to obtain the mean gradient over the full global batch?
4. Calculate the gradient tensor size and the total tensor sizes submitted to synchronization per GPU per update for each strategy. How does communication volume compare? Distinguish these tensor sizes from actual network traffic.

Assume one optimizer update after accumulation and ignore floating-point rounding differences.

## 2. Sharding optimizer state: ZeRO Stage 1

Return to the **7B model on 8 GPUs with 80 GB each**. Distribute the **FP32 master weights and Adam moments** evenly across GPUs, keeping BF16 model weights and gradients fully replicated. Use the same bytes per parameter as above and ignore activations and temporary buffers.

1. How much memory does each GPU now need? Does the configuration fit?
2. After synchronizing gradients, each GPU updates only the parameters whose optimizer state it owns. How can every GPU obtain the complete updated BF16 model? **Name the collective** needed.
3. As the number of GPUs increases, which memory costs shrink and which remain constant? Write the per-GPU model-state memory as a function of GPU count and find its limit as that count grows.

## 3. ZeRO Stage 2: gradient sharding and communication costs

Keep the **7B model and 8 GPUs**. Shard the **BF16 gradients** alongside the FP32 master weights and Adam moments. BF16 model weights remain replicated. Ignore activations and temporary buffers.

1. Calculate the per-GPU model-state memory and the saving compared with Stage 1. What memory remains replicated?
2. Each GPU needs only the globally averaged gradient shard for the parameters it updates. **Name the collective** that produces this directly, replacing the full gradient all-reduce. Which collective restores the complete BF16 weights after the optimizer update?
3. Compare two communication patterns: **our Stage 1 implementation** uses a full gradient all-reduce followed by an updated-weight all-gather; **Stage 2** uses the gradient collective you identified followed by the weight all-gather. For ring implementations, derive the bytes sent per GPU for each collective and the total per stage. Assume one gradient synchronization per optimizer update and BF16 communication for gradients and weights. Count sends only, excluding receives and protocol overhead.
4. Does gradient sharding necessarily increase communication? Give the communication-volume ratio for these two specified implementations.

## 4. ZeRO Stage 3: sharding parameters

Keep the **7B model and 8 GPUs**, but now shard **all model states**, including BF16 weights. Gather each layer's complete weights for its forward pass and discard the gathered copy immediately afterward; repeat for its backward pass. Keep the local shards. Assume no prefetching or caching.

1. Calculate the persistent model-state memory per GPU. Why can peak memory exceed this even without activations? Explain how gathering one layer at a time differs from reconstructing the full model simultaneously.
2. **Name the collectives** for gathering weights and synchronizing gradients. Must GPUs reconstruct the complete model immediately after updating their parameter shards?
3. Calculate the total bytes sent per GPU per optimizer update, and compare with Stage 2. Use ring collectives, one forward/backward pass per update, BF16 weights and gradients, and one weight gather per parameter in each pass. Count sends only, excluding protocol overhead.
4. Instead, retain every layer's gathered weights from its forward pass until its backward pass. What communication is saved? How much weight memory accumulates by the end of forward, and how does this affect the peak-memory advantage over Stage 2? Ignore duplicate buffers and other temporary storage for this comparison.

## 5. ZeRO Stage 3: overlapping communication and computation

Consider a forward pass through **14 layers**, each with **1 GB of BF16 weights**, sharded across **8 GPUs**. Each layer takes **10 ms** to compute once its weights are gathered. Use ring all-gather and an effective sending bandwidth of **125 GB/s**; communication time is bytes sent divided by bandwidth.

Only one all-gather runs at a time. Communication and computation can overlap perfectly without slowing each other down. Release gathered weights after their layer finishes computing. Initially, no layer's complete weights have been gathered.

1. How long does one layer's all-gather take?
2. Calculate total forward time without prefetching, then with one-layer-ahead prefetching: while computing layer i, gather layer i+1. Include the initial gather.
3. How many complete layer-weight buffers must coexist with prefetching? What additional memory is required compared with no prefetching?
4. If bandwidth halves, can computation still hide all communication after startup? Calculate the new prefetched forward time and identify the bottleneck.
5. Derive the general total-time formula for L identical layers with gather time g and compute time c, using one-layer-ahead prefetching. Account for startup and completion, and explain when communication is fully hidden after startup.

## 6. ZeRO Stage 3 with gradient accumulation

Return to the **7B model on 8 GPUs**, accumulating **4 microbatches per optimizer update**. For every microbatch, gather weights layer by layer for forward and again for backward, releasing them after use. Retain no gathered weights between passes or microbatches.

Compare two hypothetical implementations:

- **A:** reduce-scatter gradients after every microbatch and accumulate the resulting gradient shards.
- **B:** accumulate unreduced local gradients and perform one reduce-scatter after all four microbatches.

Assume ring collectives, BF16 weights and gradients, and correct gradient normalization. Count bytes sent only.

1. Calculate total per-GPU communication per optimizer update for each strategy, first for K microbatches, then for K = 4. Include both weight and gradient traffic.
2. How much memory does each strategy need for its retained gradient accumulation buffer? Ignore temporary buffers.
3. In replicated data parallelism, delaying synchronization across four microbatches reduced communication fourfold. Why does it not achieve that reduction here? What is the ratio of B's communication to A's as K becomes large?

## 7. Sharding within nodes, replication across nodes

Use **2 nodes with 8 GPUs each**, fast connections within each node and a slower connection between nodes. Keep the **7B model** and previous memory assumptions. Shard all model states within each group of 8 GPUs; the two nodes hold matching copies of those shards. Every GPU processes a different, equal-sized microbatch.

1. Calculate persistent model-state memory per GPU and compare with sharding across all 16 GPUs.
2. Weight all-gathers and gradient reduce-scatters happen within each node. After the local gradient reduce-scatter, what additional communication makes each shard's gradient reflect data from all 16 GPUs? **Name the collective and its participating GPUs.**
3. With one gradient synchronization per optimizer update, calculate cross-node bytes sent per GPU and per node. Assume BF16 gradients and ring collectives; count sends only.
4. Why might this arrangement train faster than sharding across all 16 GPUs, despite its higher memory requirement? Which repeated communications stay off the slower cross-node network?

## 8. Tensor parallelism: splitting an MLP

Consider the MLP design from [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053):

\[
Y=\operatorname{GELU}(XW_1)W_2,
\quad X\in\mathbb R^{2048\times4096},
\quad W_1\in\mathbb R^{4096\times16384},
\quad W_2\in\mathbb R^{16384\times4096}.
\]

Distribute computation across **8 GPUs**. Each starts with the same X and must finish with complete Y. GELU acts elementwise. Ignore biases, dropout, and optimizer states.

1. How should W_1 and W_2 be partitioned so each GPU computes with its own weight shards, with no communication between the two matrix multiplications? Give local weight and intermediate activation shapes.
2. Why can GELU run locally with this partition? What goes wrong if W_1 is instead partitioned along its input dimension and GELU is applied before combining partial results? What communication would that alternative require before GELU and after the second multiplication?
3. **Name the collective** needed after the second multiplication in the original partition. Should it sum or average, and why?
4. With BF16 weights and communicated activations, calculate weight memory per GPU and bytes sent per GPU during the forward pass using a ring collective. Express both in MiB, where 1 MiB = 2^20 bytes.

## 9. Tensor parallelism through backpropagation

Keep the MLP, dimensions, and **8 GPUs** from Question 8. On GPU r:

\[
Z_r=XW_{1,r},\qquad H_r=\operatorname{GELU}(Z_r),
\qquad Y=\sum_{r=1}^{8}H_rW_{2,r}.
\]

Every GPU receives the same upstream gradient \(G=\partial L/\partial Y\), representing **one logical loss**. There is no data parallelism in this example.

1. Derive the local gradients for W_{2,r}, H_r, Z_r, and W_{1,r}. Can these all be computed without communication? Give their shapes.
2. Derive GPU r's contribution to the input gradient. How is the complete input gradient obtained on every GPU? **Name the collective and its reduction.** Do weight gradients also need synchronization across this tensor-parallel group?
3. Assuming BF16 communication and ring collectives, calculate bytes sent per GPU during backward alone, then during the complete forward-backward pass. Which tensor is communicated in each direction, and is it a slice of the complete result or a full-shaped partial sum?

## 10. Combining tensor and data parallelism

Run the MLP from Question 8 on **16 GPUs across 2 nodes**, with ranks 0-7 on one node and 8-15 on the other. Use **TP degree 4 and DP degree 4**, with contiguous TP groups: ranks 0-3, 4-7, and so on. There is no ZeRO or pipeline parallelism.

Each TP group processes **2 sequences of 1024 tokens per microbatch**, accumulating **4 microbatches per optimizer update**. Different data replicas process different sequences. Communicate activations and weight gradients in BF16.

1. List the TP and DP groups. Which GPUs hold matching copies of the same weight shard?
2. What is the effective global batch size in sequences? Why does the TP degree not multiply it?
3. Which tensors are synchronized within TP groups, and which within DP groups? **Name the collectives and reductions.** Which synchronizations can wait until gradient accumulation finishes?
4. For this MLP alone, calculate bytes sent per GPU per optimizer update, separately for TP and DP. Assume ring collectives and one DP gradient synchronization per update; ignore protocol overhead.
5. Extend to L identical MLP blocks. How do total communication volume and collective-call count scale for TP and DP? First pack all DP gradients into one collective per update, then consider fixed-size DP gradient buckets. Explain the difference between TP's layer dependencies and DP's ability to combine gradient communication across layers.

## 11. Pipeline parallelism and bubbles

Investigate the approach in [GPipe](https://arxiv.org/abs/1811.06965). Split a model into **4 consecutive stages**, each on a different GPU. A training batch contains **8 microbatches**. Each stage takes **3 ms for forward and 6 ms for backward**, per microbatch.

All microbatches finish forward before any backward computation begins. Within each phase, pipeline the microbatches across stages. Ignore communication time, recomputation, and optimizer overhead. There is no TP or DP.

1. What is communicated between stages during forward and backward? Are these collective operations or point-to-point transfers?
2. Calculate complete forward-backward time and each GPU's utilization, defined as time computing divided by total elapsed time. Where does idle time arise?
3. Generalize to p balanced stages, m microbatches, forward time f, and backward time b. Derive total time, utilization, and the fraction of time lost to pipeline bubbles.
4. To reproduce one synchronous optimizer update over the entire batch, when should stages update their weights? Why would updating after each microbatch change the training algorithm? Account for the fact that all forwards have already used the old weights.

## 12. 1F1B: activation memory versus pipeline bubbles

Investigate the synchronous 1F1B schedule discussed in [Megatron-LM (2021)](https://arxiv.org/abs/2104.04473). Keep **4 stages and 8 microbatches**, but each forward or backward operation now takes **1 ms**. Each stage retains **1 GB of saved activations per microbatch**, from completion of its forward until completion of its backward. Ignore recomputation, communication time, and temporary buffers.

Stages 0, 1, 2, 3 first perform **3, 2, 1, 0 forwards**, respectively. Each then alternates one new forward with one backward for its oldest outstanding microbatch, waiting for dependencies as needed. After its final forward, each stage completes its remaining backwards.

1. What is peak saved-activation memory on each stage? Compare with performing all forwards before any backwards.
2. Calculate total forward-backward time and GPU utilization. Does this schedule reduce the pipeline bubble, or primarily something else?
3. Does one-forward, one-backward imply one optimizer update per microbatch? When should weights change to preserve synchronous batch training?
4. If every boundary activation and activation gradient occupies **16 MiB**, calculate total bytes transferred across all three stage boundaries per optimizer update, counting each transfer once. Does switching to 1F1B change that volume?

## 13. Choosing microbatch size: bubbles versus computation efficiency

Keep **4 pipeline stages**, synchronous **1F1B**, and a fixed batch of **64 sequences per optimizer update**, without TP or DP. Assume these measured per-stage times for one microbatch:

| Sequences per microbatch | Forward | Backward |
|---:|---:|---:|
| 1 | 0.8 ms | 1.6 ms |
| 2 | 1.0 ms | 2.0 ms |
| 4 | 1.6 ms | 3.2 ms |
| 8 | 3.0 ms | 6.0 ms |

Saved activations occupy **0.5 GB per sequence per outstanding microbatch** on each stage. Every GPU has **6 GB available for saved activations** after reserving other memory. Ignore communication time and temporary buffers.

1. For each microbatch size, calculate the number of microbatches, bubble fraction, and total iteration time. Which is fastest without the memory constraint?
2. Calculate peak saved-activation memory on the first stage. Which choices fit, and which feasible choice is fastest?
3. Why does minimizing the bubble fraction not necessarily minimize iteration time?
4. Suppose boundary tensor sizes scale linearly with sequences per microbatch. At this fixed batch size, how does changing microbatch size affect total communication bytes versus the number of point-to-point messages?

## 14. Pipeline imbalance: when more microbatches cannot fix utilization

Keep **4 stages and 8 microbatches**, with these per-microbatch computation times:

| Stage | Forward | Backward |
|---|---:|---:|
| 0 | 1 ms | 2 ms |
| 1 | 2 ms | 4 ms |
| 2 | 4 ms | 8 ms |
| 3 | 1 ms | 2 ms |

Use the all-forwards-then-all-backwards schedule. Each stage starts an operation as soon as its dependencies and GPU are available. Intermediate outputs can be queued without capacity limits, so a stage need not wait for its neighbor to accept an output before continuing. Ignore communication, recomputation, and optimizer overhead.

1. Calculate total forward-backward time and each stage's utilization. Derive utilization from useful computation divided by elapsed time rather than tracing every idle interval. Why can the balanced-stage formula not be applied directly?
2. Generalize total time to m microbatches with stage-specific forward times f_i and backward times b_i. Allow the forward and backward bottlenecks to occur on different stages.
3. As m becomes arbitrarily large, what happens to each stage's utilization and the average across all four GPUs? Can more microbatches eliminate all idle time?
4. Suppose repartitioning gives every stage **2 ms forward and 4 ms backward**, preserving total computation. What is the new iteration time for eight microbatches? Why might assigning equal layer counts or parameter counts fail to achieve this balance?

## 15. Combining TP, PP, and DP, then adding full sharding

Train a **70B-parameter model on 32 GPUs**, using **TP = 4, PP = 4, DP = 2**. Split parameters evenly across PP stages and TP ranks. Identify GPUs by

\[
\text{rank}=16d+4p+t,
\]

where d is 0 or 1, and p and t are each 0, 1, 2, or 3. Each data replica processes **8 microbatches of 2 sequences** per optimizer update. Initially, model states are replicated across DP replicas. Use **16 bytes per parameter** for weights, gradients, master weights, and Adam moments; ignore activations and temporary buffers.

1. For rank 5, list its TP, PP, and DP groups. Which ranks hold copies of the same parameter shard?
2. What is the effective global batch size? Why do TP and PP degrees not multiply it?
3. Calculate persistent model-state memory per GPU.
4. Describe rank 5's communication within its TP group, across its DP group, and with neighboring PP stages. **Name the collectives or point-to-point operations.** Use matching TP coordinates across PP stages and give the ranks involved in forward and backward transfers. Does a PP group imply a group-wide collective?
5. Replace replicated DP with FSDP-style full sharding across each DP pair, sharding parameters, gradients, and optimizer states. What is the new persistent memory per GPU? Which communication operations are added or replace the original DP synchronization, and does global batch size change?
6. Suppose adjacent stages' TP groups occupy different nodes and each TP rank holds the same complete **16 MiB boundary activation**. Compare cross-node bytes sent when all four matching pairs send complete copies; one pair sends a complete copy followed by a destination broadcast; or each pair sends a distinct quarter followed by a destination all-gather. What local communication is needed, and why might the latter two approaches differ in performance? Does the same reasoning apply to replicated activation gradients in backward? See the scatter-gather optimization in [Megatron-LM (2021)](https://arxiv.org/abs/2104.04473).

## 16. Full sharding with TP and PP: memory versus communication

Keep the **70B model, TP = 4, PP = 4, DP = 2**, and **8 microbatches per optimizer update**. Compare three hypothetical strategies along the DP axis:

- **A, replicated:** accumulate full local weight gradients, then perform one DP all-reduce per update.
- **B, fully sharded:** gather each layer's TP weight shard before every forward and backward, releasing it afterward. Reduce-scatter gradients after every microbatch and accumulate gradient shards.
- **C, retain gathered weights:** keep optimizer states and gradients sharded as in B, but gather each weight shard only once per update and retain it through all eight microbatches. After the optimizer update, discard gathered copies.

Use BF16 weight and gradient communication and ring collectives. Count bytes sent per GPU along the DP axis only, summing across that GPU's assigned layers. Use our previous 16 bytes per parameter for model-state memory.

1. Calculate communication per optimizer update for A, B, and C. Why does full sharding introduce communication repeated across microbatches?
2. Calculate retained model-state memory for each strategy. For C, gathered weights replace their corresponding local shards; avoid double-counting. Ignore temporary buffers and momentary layer materialization in B.
3. If **40 GB per GPU is available for retained model states**, with temporary buffers budgeted separately, which strategies fit? Which would you favor if DP-axis communication were the bottleneck?

## 17. Placing TP, PP, and DP across nodes

The aim is to compare how GPU placement changes cross-node traffic, while distinguishing communication volume from exposed communication time.

Train the **70B model with TP = 4, PP = 4, DP = 2** on **four nodes of eight GPUs**. Identify a TP group by its data-replica coordinate d and pipeline-stage coordinate p, where d is 0 or 1 and p is 0, 1, 2, or 3. Each TP group occupies four GPUs on one node. Compare:

| Node | Configuration A: consecutive stages together | Configuration B: matching stages together |
|---|---|---|
| 0 | (d=0, p=0), (d=0, p=1) | (d=0, p=0), (d=1, p=0) |
| 1 | (d=0, p=2), (d=0, p=3) | (d=0, p=1), (d=1, p=1) |
| 2 | (d=1, p=0), (d=1, p=1) | (d=0, p=2), (d=1, p=2) |
| 3 | (d=1, p=2), (d=1, p=3) | (d=0, p=3), (d=1, p=3) |

Each replica processes **8 microbatches per optimizer update**. At every PP boundary, the complete activation tensor is **0.25 GB per microbatch**, and its backward activation gradient has the same size. The four matching TP-rank pairs send distinct quarters of each boundary tensor, followed by a local all-gather when needed; there are no redundant full-tensor cross-node copies. BF16 weight gradients use **2 bytes per parameter**. Accumulate gradients across microbatches and synchronize them once per update using ring all-reduce within each DP pair. Ignore protocol overhead and all intra-node traffic; use decimal GB.

Count **aggregate bytes sent across nodes over all GPUs**, counting each transmission once. Opposite-direction sends both count, but do not count the receive of a transmission again.

1. For each placement, which TP, PP, and DP communications cross nodes? Identify the cross-node PP boundaries in each replica.
2. Calculate total cross-node PP traffic, DP traffic, and their sum per optimizer update for each placement. For DP, distinguish the ring all-reduce bytes sent per participant from the total over both replicas. Explain why two-way traffic is not receive double-counting.
3. Replace 8 microbatches with K, holding microbatch size and boundary tensor size fixed. Derive total cross-node traffic for each placement and find the crossover. Which placement sends fewer bytes on either side? Why would the calculation change if the total batch size were held fixed while K changed?
4. Does sending fewer cross-node bytes guarantee a faster iteration? Explain how DP gradient buckets can overlap synchronization with the final accumulated backward pass, and contrast this with a single all-reduce launched after backward finishes. How can PP transfers overlap with independent work on other microbatches, and how can TP communication overlap with independent computation or computation on other chunks? Discuss how dependencies, traffic concentrated on particular links, and shared network bandwidth affect the time communication adds to an iteration.

## 18. Gradient buckets: overlap, queuing, and a shared NIC

The aim is to turn communication volumes into an iteration timeline, accounting for when gradients become ready and which transfers share bandwidth.

The **final backward pass after gradient accumulation takes 40 ms**, starting at time zero. Four gradient buckets become ready for DP synchronization:

| Bucket | Ready at | Outgoing synchronization traffic |
|---|---:|---:|
| 1 | 10 ms | 2 GB |
| 2 | 20 ms | 2 GB |
| 3 | 30 ms | 2 GB |
| 4 | 40 ms | 2 GB |

The node's NIC provides **100 GB/s effective aggregate sending bandwidth**. Traffic figures already include the collective algorithm's communication overhead; do not apply another all-reduce traffic multiplier. A synchronization completes once its stated traffic has been sent. Other collective participants are ready when needed, and no other network resources limit progress. Use decimal GB.

Within a DP group, bucket synchronizations run sequentially in bucket order. They can overlap perfectly with backward computation without slowing it or changing bucket readiness times. Ignore startup latency, optimizer time, and receive-side limits. An optimizer can start only after its backward pass and all required synchronizations finish.

1. First use just one GPU on this node, participating in one DP group. If it waits until backward finishes and synchronizes all **8 GB together**, when can its optimizer start? Give both elapsed time from the beginning of backward and communication time exposed after backward.
2. Instead, start each bucket as soon as it is ready and the NIC is free. Give every bucket's transfer start and finish times. When can the optimizer start, and how much communication time remains exposed? Why does a bucket becoming ready not necessarily mean it can start transmitting?
3. Now two GPUs on the node belong to **two independent DP groups** and share this NIC. For example, two nodes each hold a model replica split across GPUs X and Y: X synchronizes with remote X, while Y synchronizes a different parameter shard with remote Y. Both local GPUs have the readiness schedule above and use bucketed synchronization. Whenever both groups are communicating, each gets **50 GB/s**; a group communicating alone gets **100 GB/s**. Give the transfer timeline for each group, when their optimizers can start, and the exposed communication time. Check the result using total outgoing bytes divided by aggregate NIC bandwidth.
4. Why would assigning each local GPU an independent 100 GB/s connection give the wrong result? Distinguish the exercise's sequential bucket scheduling within each group from concurrent requests sharing a NIC: must all NIC requests universally execute one at a time, or can they progress concurrently while sharing a fixed bandwidth budget?

## 19. Choosing gradient bucket size: overlap versus startup cost

A backward pass lasts **40 ms**, producing **8 GB of gradients uniformly over time**. Split the gradients into m equal buckets, so bucket j becomes ready at time **40j/m ms**, for j = 1, ..., m. All DP participants have the same readiness schedule.

The NIC provides **400 GB/s**, and **each bucket synchronization has a fixed 2 ms startup cost**, during which no payload transfers. Assume outgoing synchronization traffic equals the bucket size; use decimal GB. Each synchronization occupies the communication resource during both startup and payload transfer, and synchronizations execute sequentially in bucket order. A bucket starts as soon as it is ready and the previous synchronization has finished.

Communication overlaps perfectly with computation without slowing it or changing bucket readiness times. The optimizer can start once backward and all synchronizations finish. Ignore optimizer time and all other overheads.

1. For **m = 1, 4, and 16**, calculate each bucket's synchronization duration, including startup, and when the optimizer can start. Account for both bucket readiness and queuing behind previous synchronizations.
2. Why can both one large bucket and many tiny buckets perform worse than an intermediate choice? Identify the competing costs. Explain why a communication queue can extend beyond backward even while communication overlaps with computation.
3. Derive optimizer start time as a function of m, distinguishing whether a synchronization finishes before the next bucket becomes ready or communication accumulates a queue. Find the integer m that minimizes this time, and give the remaining exposed communication time at the optimum. Explain the relationship between bucket production intervals and synchronization duration there.

## 20. Uneven replica progress: collective latency versus network time

Two DP replicas, **A and B**, each run on one GPU. Their three gradient buckets become ready at these times, measured from the beginning of backward:

| Bucket | Replica A | Replica B |
|---|---:|---:|
| 1 | 10 ms | 10 ms |
| 2 | 20 ms | 35 ms |
| 3 | 30 ms | 40 ms |

Backward finishes when the last bucket becomes ready on each replica. Each bucket's all-reduce takes **8 ms once both replicas are ready**. Collectives execute sequentially in bucket order. For this simplified model, no useful transfer starts before both replicas are ready; actual collective implementations can make partial progress earlier. Communication overlaps perfectly with computation without slowing it or changing readiness times. An optimizer update can begin once both backward passes and all required synchronizations finish. Ignore optimizer time and other overheads.

1. When does each all-reduce start useful communication and finish? When can the optimizer update begin?
2. Replica A measures bucket 2's collective from when it can locally launch it--its bucket is ready and the previous collective has finished--until completion. What duration does it observe? Separate waiting for another replica from actual communication time. Why would attributing the entire observed duration to network transfer be misleading?
3. Suppose A becomes twice as fast: its buckets are ready at **5, 10, and 15 ms**, while B is unchanged. Does the optimizer start earlier? Explain which dependencies determine the result.
4. Instead, keep A's original timings and improve B's bucket 2 readiness from **35 to 20 ms**, leaving its final bucket at **40 ms**. When can the optimizer start? Explain how improving an intermediate bucket's readiness can reduce iteration time even when neither replica's total backward time changes. Identify the queuing delay that disappears.

## 21. Activation checkpointing: memory, recomputation, and FSDP

A model has **16 consecutive blocks**. Each block takes **2 ms forward** and **4 ms backward**. In this simplified model, each block's backward needs only its input activation, occupying **1 GB**, plus its weights. Count the model's initial input among saved activations. Exclude the final output, activation gradients, and temporary computation buffers. There is **8 GB available for saved activations**.

Without checkpointing, retain every block's input until its backward finishes.

With checkpointing, divide the model into segments of **k blocks**, where k divides 16. During the original forward, retain only each segment's input. Before backward through a segment, rerun its **entire forward**, retaining its internal block inputs; count its already-stored segment input only once. Finish the segment's backward and free its saved activations before processing the preceding segment. For the main calculation, always rerun all k forwards, even if some could be avoided. Ignore communication initially.

1. Without checkpointing, calculate peak saved-activation memory and total forward-backward time.
2. Derive peak saved-activation memory as a function of k. Distinguish the number of retained segment inputs from the number of internal inputs reconstructed within one segment. Evaluate **k = 2, 4, and 8** and identify which choices fit. Why can both very short and very long segments use more memory?
3. Under the stipulated full-segment recomputation rule, calculate total forward-backward time and its percentage increase over the baseline. Does retaining a segment input, by itself, provide the inputs needed for backward through all blocks in that segment?
4. Add **FSDP**: each block requires a **1 ms weight all-gather before every execution**, including original forward, recomputed forward, and backward. Release gathered weights after each execution and assume no communication overlap. Count all-gathers and calculate total time with and without checkpointing. Ignore other communication for this comparison.
5. Now allow early stopping of recomputation, retaining the assumption that each block's backward needs only its input and weights. For a four-block segment starting from saved activation a_0, identify which forwards produce all inputs needed by backward. Must the segment's final forward be rerun? Does the answer change for the model's final block? Derive the optimized number of recomputed forwards for general k and recalculate compute time and time including FSDP gathers at **k = 4**. Why should this input-only model not be assumed to describe every real block's backward requirements?

## 22. Sequence parallelism and the bridge to context parallelism

Consider **Y = GELU(X W_1) W_2**, with:

- **X:** 2048 token rows by 4096 hidden features.
- **W_1:** 4096 by 16384.
- **W_2:** 16384 by 4096.

Use **4 GPUs**, BF16 activations, and tensor parallelism: split W_1 by output features and W_2 by matching input features. There is no DP or PP. In ordinary TP, every GPU starts with complete X and finishes with complete Y.

For Megatron-style sequence parallelism, keep the same TP weight shards, but let each GPU start with **512 distinct token rows of X**, containing all 4096 features, and finish with the corresponding 512 rows of Y. The token rows are positions of one sequence, not independent batch samples. Ignore biases and dropout; LayerNorm, where discussed, normalizes each token over its hidden features.

1. Calculate input activation memory per GPU with replicated X and with sequence-sharded X. Can LayerNorm run locally on the sharded layout, and why? Use MiB, where 1 MiB = 2^20 bytes.
2. Which collective must precede the first matrix multiplication so each TP weight shard processes every token? Specify whether activations or weights are gathered. After the second multiplication, which collective both sums full-shaped partial results and distributes the output by token rows? Give local input, gathered input, local intermediate, partial output, and final output shapes.
3. Compare forward bytes sent per GPU for ordinary TP and the sequence-parallel layout using ring collectives. Count sends only and distinguish the contributions of each collective. Does reducing replicated boundary activations necessarily increase communication volume? Does it mean complete X is never temporarily materialized, or every intermediate tensor becomes four times smaller?
4. For a separate **context-parallel** setup, use **4 GPUs with replicated weights**, each owning 512 distinct positions of the same sequence throughout the model. Why can tokenwise MLPs process local rows, while full bidirectional self-attention cannot attend only to local tokens? Which remotely owned tensors are needed for local queries? How do the information requirements change with causal attention and contiguous sequence chunks?
5. Explain how context parallelism differs from data parallelism and tensor parallelism. Why does context parallelism require attention communication during forward and backward, and why must contributions to replicated weight gradients also be combined?

## 23. Inference: prefill, decoding, and batching

Consider a **7B-parameter model** with **BF16 weights (2 bytes per parameter)** on one GPU with **1 TB/s memory bandwidth** and **100 TFLOP/s compute throughput**. Approximate computation as **2N FLOPs per processed token**, where N is parameter count. Use decimal units. Ignore attention's additional computation, KV-cache traffic, and other overheads.

Each decoding step generates one token per sequence. Assume it reads all model weights from GPU memory once, reusing them across the batch. Assume ideal overlap of memory transfers and computation, so ideal step time is the maximum of their separate lower bounds.

1. What distinguishes prefill from autoregressive decoding? Why can prompt positions be processed together within a layer, whereas generating successive tokens for one sequence requires successive steps? What does prefill compute besides the KV cache?
2. Using token representations as columns of X and projection matrices W_Q and W_K, write the matrix expression for all query-key scores. Where are scaling, the causal mask, and softmax applied? Specify the mask entries and softmax axis. Explain why causal masking does not require sequential processing of prompt positions, and why layers still have dependencies.
3. For decoding batch sizes B = 1 and B = 128, calculate weight bytes read per step, FLOPs per step, memory-transfer time, and compute time. Using the ideal-overlap assumption, calculate step time, aggregate tokens per second, and time between tokens for each sequence.
4. Derive the batch size where memory-transfer and compute times are equal. Describe how aggregate throughput and per-sequence time between tokens change below and above this crossover. Explain how batching amortizes weight reads.
5. What overlap assumption justifies taking the maximum of memory-transfer and compute times rather than their sum? Explain how loading weight tiles can overlap with computation on other tiles, without requiring complete layers to execute independently.

## 24. Distributed inference: tensor parallelism versus inference replicas

Serve a **70B-parameter model with BF16 weights (2 bytes per parameter)** on **8 GPUs with 80 GB each**. Compare:

| Configuration | Inference replicas | TP per replica |
|---|---:|---:|
| A | 1 | 8 |
| B | 2 | 4 |

Reserve **10 GB per GPU** for everything except weights and KV cache. At the context length considered, each request needs **2 GB of KV cache across the complete model**. Assume weights and KV cache shard evenly across TP ranks. Balance requests equally between B's replicas. Use decimal GB.

All requests have finished prefill and remain active throughout measurement. Ignore cache growth and request queuing. Use these **hypothetical measured decoding step times**, already including TP communication:

\[
t_8(b)=8+0.20b\text{ ms},\qquad t_4(b)=10+0.25b\text{ ms},
\]

where b is the number of requests **in that replica's batch**. Each step generates one token per request.

1. Calculate weight memory and available KV-cache memory **per GPU** in A and B. Calculate KV-cache bytes per request per GPU, using the relevant TP degree. What is the maximum number of simultaneous requests per replica and across each deployment? How does duplicating model weights affect aggregate KV-cache capacity?
2. With **64 active requests total**, calculate decoding step time, time between tokens for each request, and aggregate tokens per second for A and B. Explain why step time and time between tokens coincide under these assumptions. Which layout performs better in throughput and latency?
3. Repeat with **8 active requests total**. Write both deployment step times in terms of total active requests R and derive the crossover, assuming R can be split equally for B. Why can the preferred layout change with load? Do the supplied timing formulas alone establish which costs are memory-bound or compute-bound? Explain why two replicas do not provide twice the total GPU resources of A.
4. B has two model copies processing independent requests. Does inference require the gradient all-reduces used in training DP? Distinguish communication between replicas from activation communication within a TP replica.
5. Why can ordinary TP shard a model's KV cache without sequence or context parallelism? Explain head-wise sharding, where each rank can store every token position for a subset of attention heads. Are raw attention-head outputs summed or concatenated in standard multi-head attention? Explain how partitioning the subsequent output projection lets TP ranks compute full-shaped partial contributions and combine them with an all-reduce sum, without first gathering all head outputs. Ignore projection bias.
