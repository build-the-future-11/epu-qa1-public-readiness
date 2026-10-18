# EPU-QA1: query-conditioned retrieval intervention

Version: 2026-10-15, specified before implementation or new data access.
Scope: a bounded, exposed-data development study of the native EPU failure.
This is not a claim of a novel neural architecture or an accuracy result.

## Introduction and question

The retained EPU report describes an iterative classifier much worse than its
direct nearest-prototype control. Existing EPU recurrence has no explicit query
in its fixed-point equation. We propose a fixed query anchor as a controlled
intervention because it bounds movement away from the query and makes the
fixed point query-conditioned. Does that intervention recover useful decisions,
and does it improve on simply retaining the direct predictor?

The contribution sought is a source-bound failure analysis, paired intervention
and an explicit boundary between stable retrieval and useful classification.
It can be informative if the candidate fails. No claim is made that this is the
first analysis of such behavior, a new class of Hopfield networks, a biological
model, or a hardware result. The actual cause of the old digits outcome is still
unmeasured; aggregate near-chance accuracy alone does not establish collapse.

H1 (mechanistic): anchoring reduces changed decisions and concentration of final
winners relative to the original rule on a fixed bank and fixed queries. Winner
concentration is descriptive; multiple inputs can legitimately share a class.
H2 (utility): the fixed candidate exceeds direct retrieval by at least 1 percentage
point in mean accuracy over the three declared corruptions. This is a development
screen, not a powered or confirmatory claim. If H1 holds but H2 fails, stability
has not established incremental predictive value.

## Prior art and novelty boundary

Search conducted 2026-10-15 using modern Hopfield/query regularization,
input-driven dynamics, fixed context, sparse retrieval and nonparametric retrieval
queries. Sources below are primary papers. The review is targeted rather than a
claim of exhaustive coverage. HF markdown endpoints failed; original sources were
used. No unavailable full text is represented as reviewed.

| Source and reviewed material | Relevant finding | Consequence for EPU |
|---|---|---|
| [Ramsauer et al., Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217), abstract; full PDF fetch exceeded tool limit | Softmax retrieval, attention equivalence and several attractor regimes are established. | Neither softmax recurrence nor an averaging-attractor explanation is novel. Capacity results do not establish this classifier's accuracy. |
| [Millidge et al., ICML 2022](https://proceedings.mlr.press/v162/millidge22a.html), abstract and framework summary | Similarity, separation and projection organize associative-memory models; alternative metrics are studied. | Metric changes need matched direct controls. Unit-norm squared Euclidean distance and dot-product rankings are equivalent. |
| [Santos et al., JMLR 2025](https://jmlr.org/papers/volume26/24-1961/24-1961.pdf), introduction, sections 3.1–3.2, proposition 2 and appendix B.1 | Sparse/structured transforms and post-transformations fit a broad energy framework; softmax mixtures can require temperature care. | Sparsity and normalization are established alternatives, not automatic fixes. Energy descent and memory retrieval require stated assumptions. |
| [Hu et al., ICML 2025](https://arxiv.org/html/2404.03900v2), introduction, discussion, appendix B | Nonparametric interpretations include sparse and approximate retrieval. The paper explicitly limits superiority claims and discusses convergence assumptions. | Approximation/efficiency branches need numerical and cost comparisons; convergence does not guarantee a better classifier. |
| [Betteti et al., Input-driven dynamics, Science Advances 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12017325/), indexed abstract/introduction/equations 4–6; direct full page blocked by captcha | Inputs and initialization have different roles. Input-dependent synapses and continuous external drive have already been studied; constant drive can also distort stored attractors. | Persistent input itself is not novel. Our additive fixed-query blend is a simpler intervention, not their synaptic-plasticity mechanism or a reproduction of their results. |
| [Modern Hopfield Networks for graph embedding, 2022](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2022.1044709/full), indexed method section | Context can remain fixed while target state iterates. | Fixed context already exists. No first-use claim for anchoring is justified. |

The new branch tests a local unresolved question. Broad EPU novelty, efficacy,
hardware and neuroscience gates remain open. Independent literature review is
still needed before any publication-level novelty assertion.

## Complete model contract

Inputs are a bank M of k centered memory vectors in R^d, associated integer class
labels, and a centered query q in the closed Euclidean unit ball. k<=32 and
2<=d<=128. Memory vectors also lie in that ball. Zero vectors and duplicates are
allowed and reported; nonfinite, dimensionally invalid or out-of-domain inputs
raise errors. Centering tolerance and norm tolerance are 1e-10.

Training consists only of arithmetic means of raw training images per observed
class, followed by row centering and L2 normalization. Raw pixels are divided by
16 before means are formed. No development label contributes to a prototype.
The divisor for normalization is max(norm,1e-12); zero/constant images stay zero.
Classes are sorted ascending and absent training classes cause evaluation failure.
There are no learned weights, optimizer, loss, gradient, gamma term or hidden
calibration parameters. A serialized checkpoint is the bank, labels and config.

Initialize z_0=q. For each of T steps:

    s_i = beta <m_i,z_t>
    w_i = exp(s_i-max(s)) / sum_j exp(s_j-max(s))
    r_t = sum_i w_i m_i
    z_(t+1) = alpha z_t + (1-alpha)[(1-lambda)q + lambda r_t]

The candidate is alpha=0.6, beta=6, lambda=0.1, T=8, inherited from the previous
draft without a data-driven sweep. No state renormalization, clipping or adaptive
stopping is applied. The input assumptions make clipping unnecessary. Decode with
argmax_i <m_i,z_T>, breaking exact ties by bank order. Report exact and near ties
(gap<=1e-12), zero norms and prediction changes. Softmax weights are not calibrated
class probabilities. Summation uses Python float/math.fsum; original-source
parity is numerical, not bitwise, and near ties must remain visible.

Model API may accept alpha in [0,1), beta in [1e-12,100], lambda in [0,1], T in [0,64]
for mathematical controls; the experiment accepts only its fixed named variants.
The E5 numerical amendment narrows the public beta domain to avoid
nonfinite energy from division by a subnormal positive beta. Default beta=6 and
the fixed experimental configuration are unchanged. This lower bound guarantees
neither accurate energy differences near the bound nor empirical performance.
Normalization retains every E4 output that passes the centered-unit-ball check.
Only when that output fails does it center reference-subtracted residuals and
normalize again, then validate the result. See NUMERICAL_AMENDMENT.md for the
retained failure fixtures and the limits of this separately versioned repair.
Time O(Tkd), bank storage O(kd), working state O(k+d), optional trace O(T(k+d)).
The direct control uses the same bank and decoder in O(kd). Serial CPU execution
uses no accelerator or numerical-library thread pool.

## Conditional mathematical evidence

Let D=max_ij ||m_i-m_j|| and L=beta D^2/4. The reconstruction derivative is beta
times weighted memory covariance. For any unit direction, variance of its
projection is bounded by D^2/4, hence ||Dr||<=L. The update derivative norm is at
most c=alpha+(1-alpha)lambda L. For the fixed candidate D<=2 gives L<=6, c<=0.84.
This is a global contraction in exact arithmetic; no spectral-norm estimate or
floating-point test is being substituted for the proof.

Define E_q(z)=||z||^2/2-(1-lambda)<q,z>-(lambda/beta)logsumexp(beta Mz).
Its gradient is z-(1-lambda)q-lambda r(z). Its Hessian is I-lambda Dr, bounded
below by (1-lambda L)I. Therefore the candidate has a unique minimizer and fixed
point. Convexity of logsumexp and the exact quadratic expansion give, with
eta=1-alpha, E_q(z+)-E_q(z)<=-(1/eta-1/2)||z+-z||^2. For alpha=.6 this is
-2||z+-z||^2. These energy guarantees do not assert class correctness.

The closed unit ball and centered subspace are invariant under the convex update.
Since ||r-q||<=2, induction gives ||z_t-q||<=2 lambda(1-alpha^t). A direct winner
a is preserved if every initial margin <m_a-m_b,q> exceeds
2 lambda(1-alpha^t)||m_a-m_b||. This also preserves confident wrong answers.

For two queries, their fixed points obey

    (1-lambda)/(1+lambda L) ||dq|| <= ||dz*||
        <= (1-lambda)/(1-lambda L) ||dq||, if lambda L<1.

Subtract the fixed-point equations and apply the triangle/reverse-triangle
inequalities to prove both bounds. Thus lambda<1 makes the query-to-fixed-point
map injective under contraction. The discrete class decoder is not injective.
These results repair query dependence, not the statistical representation.

With scores s=Mz and Gram matrix G=MM^T the exact evolution is
s+=alpha s+(1-alpha)[(1-lambda)Mq+lambda G softmax(beta s)]. For any two memory
indices, subtract these equations to attribute the change in their margin.
Uniform weights have a closed form and two equal-norm memories preserve the
sign of their initial score difference. Those controls were derived in the
retained 15 October review; no universal rank-collapse claim is justified.

## Adversarial flaw register

| Flaw / cause | Severity | Repair in this branch | Required evidence |
|---|---|---|---|
| Iteration loses to direct retrieval; cause not traced | Critical | Native-source traces and fixed anchor intervention | Real paired trajectories and outcomes; still unrun |
| Stable wrong answers; stability is not utility | Critical | Direct baseline and paired changed-correct/changed-wrong counts | H2 screen, future fresh confirmation |
| Input anchoring and sparse attention are prior art | Critical | Narrow contribution to failure analysis; remove architecture-novelty language | Independent publication review if pursued |
| Query-free stationary equation | High | Persistent original query in every step | Proven contraction/query dependence; tested implementation |
| One mean per class erases modes | High | Stronger full-bank kNN controls, report their memory cost | If kNN wins, acknowledge representation bottleneck; no silent new bank |
| Exposed benchmark cannot confirm a tuned successor | Critical | Label every current run exposed development; refuse confirmation mode | New provenance/split/power-reviewed confirmatory protocol |
| Writer dependence and overlapping old splits | High | One deterministic new development split; no independent-seed claims | Descriptive paired analysis, no population p-values |
| Exact duplicate leakage | High | Group equal raw images before splitting | Disjoint feature hashes and row IDs checked at runtime |
| Noise seeds can become hidden tuning | High | One named noise schedule and common corruptions | All three noises retained; no favorable-subset promotion |
| Temperature/sparsity/feature tuning confounded | High | No grid in QA1; fixed controls only | Any future change gets its own version and budget |
| Extra iterations incur overhead | High | One-step and direct repeated-budget controls; exact dot-operation counts and timing | Compare useful gain per cost; no hardware efficiency claim |
| Floating ties, malformed values, silent shape truncation | High | Strict domain validation, stable softmax, explicit ties | Edge, metamorphic and native-reference tests |
| Gamma is unused; RTL differs | High | New model has no gamma; old sources untouched | Separate specified RTL equivalence study, not this branch |
| No current resource headroom | Blocking | Existing 512 MiB/pageout/swap guard before data and periodically | Fresh passing samples; do not bypass or close other work |
| Experiment can overwrite old evidence | High | Exclusive run directories, hashes, retained refusals, fixed attempt ledger | Failure-path and output-preservation checks |
| Same-agent tests are not independent reproduction | High | Label checks engineering-only | External/independent reproduction remains open |
| Universal/generalization/perfection claim | Unsupported | Restrict scope to this source, bank and development corpus | Broader evidence cannot be inferred from this package |

Multiple mechanisms were considered: anchoring, changing memory representation,
changing similarity, sparse selection and adaptive stopping. Anchoring is selected
because it changes one explicit term with an analyzable bound. Other changes are
deferred rather than bundled into an uninterpretable performance search.
