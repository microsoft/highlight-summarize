import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# LLM Control Region Formalization

This file formalizes the security analysis of Highlight & Summarize (H&S) pipelines
for LLM applications, proving that H&S exponentially reduces an attacker's control
over LLM outputs.

## Main Results

- `controlRegion_card_bound` (Lemma): For any LLM L and threshold β, the control region
  satisfies |C_β(L)| ≤ |P|/β where P is the input space.

- `hs_security_theorem` (Theorem): Consider an LLM L : Σ^K → Σ* and an H&S pipeline
  L ∘ h_D on a document of length Len. Assuming |C_β(L)| ≥ α|P| for constants α,β ∈ (0,1],
  then |C_β(L ∘ h_D)| / |C_β(L)| = O(K·Len · |Σ|^{-K}).

## References

Based on the theoretical framework in "Highlight & Summarize: Defending
Against Prompt Injection Attacks in LLM Applications".
-/

open Finset BigOperators

/-! ## Basic Definitions -/

/-- The input space P = ⋃_{i=1}^K Σ^i consists of all token sequences
    with length between 1 and K (the maximum context length). -/
def InputSpace (Token : Type*) (K : ℕ) := { l : List Token // 1 ≤ l.length ∧ l.length ≤ K }

/-- An LLM as a map L : P → Δ(O) from inputs to probability distributions over outputs.
    We model this as a sub-distribution (probabilities sum to ≤ 1) to handle
    cases where the LLM may fail to produce valid output. -/
structure LLM (Token : Type*) (K : ℕ) (O : Type*) [Fintype O] where
  /-- P(L(s) = o): probability of output o given input s -/
  prob : InputSpace Token K → O → ℝ
  prob_nonneg : ∀ s o, 0 ≤ prob s o
  prob_le_one : ∀ s o, prob s o ≤ 1
  prob_sum_le_one : ∀ s, ∑ o : O, prob s o ≤ 1

/-! ## Control Region Definition -/

/-- The β-control region C_β(L) = {o ∈ O | ∃ p ∈ P : P(L(p) = o) ≥ β}.
    This is the set of outputs an attacker can reliably produce (with probability ≥ β)
    by choosing an appropriate input from P. Defined as a Finset using classical logic. -/
noncomputable def ControlRegion {Token : Type*} {K : ℕ} {O : Type*} [Fintype O]
    (L : LLM Token K O) (β : ℝ) (P : Finset (InputSpace Token K)) : Finset O :=
  Finset.univ.filter (fun o => ∃ p ∈ P, β ≤ L.prob p o)

/-! ## Proposition: Control Region Size Bound -/

/-- For a single input s, at most 1/β outputs can have probability ≥ β.
    This follows from the fact that probabilities sum to at most 1. -/
lemma outputs_per_input_bound {Token : Type*} {K : ℕ} {O : Type*} [Fintype O]
    (L : LLM Token K O) (β : ℝ) (hβ_pos : 0 < β) (s : InputSpace Token K)
    (C : Finset O) (hC : ∀ o ∈ C, β ≤ L.prob s o) :
    (C.card : ℝ) ≤ 1 / β := by
  by_contra h_neg
  push_neg at h_neg
  -- If |C| > 1/β, then sum of probs over C > 1, contradicting prob_sum_le_one
  have h_sum_C : β * C.card ≤ ∑ o ∈ C, L.prob s o := by
    have h1 : ∑ _o ∈ C, β = β * C.card := by simp [Finset.sum_const, mul_comm]
    rw [← h1]
    apply Finset.sum_le_sum
    intro o ho
    exact hC o ho
  have h_sum_all : ∑ o ∈ C, L.prob s o ≤ ∑ o : O, L.prob s o := by
    apply Finset.sum_le_univ_sum_of_nonneg
    intro o
    exact L.prob_nonneg s o
  have h_le_one : ∑ o : O, L.prob s o ≤ 1 := L.prob_sum_le_one s
  have h_beta_card : β * C.card > 1 := by
    have hcg : (C.card : ℝ) > 1 / β := h_neg
    have h1 : β * C.card > β * (1 / β) := mul_lt_mul_of_pos_left hcg hβ_pos
    have h2 : β * (1 / β) = 1 := mul_one_div_cancel (ne_of_gt hβ_pos)
    rw [h2] at h1
    exact h1
  -- Now we have: β * |C| > 1, but sum of probs ≤ 1, contradiction
  have h_contra : β * C.card ≤ 1 := by
    calc β * C.card ≤ ∑ o ∈ C, L.prob s o := h_sum_C
      _ ≤ ∑ o : O, L.prob s o := h_sum_all
      _ ≤ 1 := h_le_one
  linarith

/-- **Lemma**: The control region size is bounded by |C_β(L)| ≤ |P|/β.

    *Proof sketch*: Each output o ∈ C has a witness input p with prob(p,o) ≥ β.
    Each input witnesses at most 1/β outputs (by `outputs_per_input_bound`).
    Using a witness function w : C → P and counting fibers, we get |C| ≤ |P|/β. -/
theorem controlRegion_card_bound {Token : Type*} {K : ℕ} {O : Type*} [Fintype O]
    (L : LLM Token K O) (β : ℝ) (hβ_pos : 0 < β)
    (P : Finset (InputSpace Token K)) :
    ((ControlRegion L β P).card : ℝ) ≤ P.card / β := by
  classical
  set C := ControlRegion L β P with hC_def
  by_cases hC_empty : C.card = 0
  · simp only [hC_empty, Nat.cast_zero]; positivity
  · by_cases hP_empty : P.card = 0
    · -- If P is empty, C must be empty (contradiction)
      simp only [Finset.card_eq_zero] at hP_empty
      exfalso
      obtain ⟨o, ho⟩ := Finset.card_pos.mp (Nat.pos_of_ne_zero hC_empty)
      rw [hC_def, ControlRegion, Finset.mem_filter] at ho
      obtain ⟨p, hp, _⟩ := ho.2
      rw [hP_empty] at hp
      simp at hp
    · -- Main case: use witness function and fiber counting
      -- For each o ∈ C, there exists a witness p ∈ P with prob(p,o) ≥ β
      have hWitness : ∀ o ∈ C, ∃ p ∈ P, β ≤ L.prob p o := fun o ho => by
        rw [hC_def, ControlRegion, Finset.mem_filter] at ho
        exact ho.2
      -- P is nonempty, so we can pick a default element
      have hP_nonempty : P.Nonempty := Finset.card_pos.mp (Nat.pos_of_ne_zero hP_empty)
      obtain ⟨p₀, _⟩ := hP_nonempty
      -- Define witness function: w(o) = some witness for o (or p₀ if o ∉ C)
      let w : O → InputSpace Token K := fun o =>
        if h : o ∈ C then Classical.choose (hWitness o h) else p₀
      -- For o ∈ C, w(o) ∈ P and prob(w(o), o) ≥ β
      have hw_mem : ∀ o ∈ C, w o ∈ P := fun o ho => by
        simp only [w, ho, dif_pos]
        exact (Classical.choose_spec (hWitness o ho)).1
      have hw_prob : ∀ o ∈ C, β ≤ L.prob (w o) o := fun o ho => by
        simp only [w, ho, dif_pos]
        exact (Classical.choose_spec (hWitness o ho)).2
      -- Define the fiber: outputs in C that chose p as their witness
      let fiber : InputSpace Token K → Finset O := fun p => C.filter (fun o => w o = p)
      -- Each fiber has size ≤ 1/β by outputs_per_input_bound
      have hFiberBound : ∀ p ∈ P, ((fiber p).card : ℝ) ≤ 1 / β := fun p _hp => by
        apply outputs_per_input_bound L β hβ_pos p
        intro o ho
        simp only [fiber, Finset.mem_filter] at ho
        have heq : w o = p := ho.2
        have hprob : β ≤ L.prob (w o) o := hw_prob o ho.1
        rw [heq] at hprob
        exact hprob
      -- C is covered by the union of fibers over P
      have hCover : C ⊆ P.biUnion fiber := fun o ho => by
        simp only [Finset.mem_biUnion, fiber, Finset.mem_filter]
        exact ⟨w o, hw_mem o ho, ho, rfl⟩
      -- |C| ≤ |⋃_{p∈P} fiber(p)| ≤ ∑_{p∈P} |fiber(p)| ≤ |P| / β
      have h1 : (C.card : ℝ) ≤ (P.biUnion fiber).card := by
        exact Nat.cast_le.mpr (Finset.card_le_card hCover)
      have h2 : ((P.biUnion fiber).card : ℝ) ≤ ∑ p ∈ P, ((fiber p).card : ℝ) := by
        have := Finset.card_biUnion_le (s := P) (t := fiber)
        calc ((P.biUnion fiber).card : ℝ)
            ≤ (∑ p ∈ P, (fiber p).card : ℕ) := Nat.cast_le.mpr this
          _ = ∑ p ∈ P, ((fiber p).card : ℝ) := Nat.cast_sum P (fun p => (fiber p).card)
      have h3 : ∑ p ∈ P, ((fiber p).card : ℝ) ≤ ∑ _p ∈ P, (1 / β) := by
        apply Finset.sum_le_sum
        intro p hp
        exact hFiberBound p hp
      have h4 : ∑ _p ∈ P, (1 / β : ℝ) = P.card / β := by
        simp only [Finset.sum_const, nsmul_eq_mul]
        rw [mul_one_div]
      calc (C.card : ℝ)
          ≤ (P.biUnion fiber).card := h1
        _ ≤ ∑ p ∈ P, ((fiber p).card : ℝ) := h2
        _ ≤ ∑ _p ∈ P, (1 / β) := h3
        _ = P.card / β := h4

/-! ## Highlight & Summarize Security Theorem

The H&S pipeline restricts LLM inputs to contiguous substrings of a document D.
A document of length Len has O(Len²) contiguous substrings. More precisely, the number
of substrings with length between 1 and K is at most K·Len (achieved when Len ≥ K).
This is compared to Θ(|Σ|^K) possible arbitrary inputs, exponentially reducing the control region.

**Coverage Assumption**: We assume the LLM has basic coverage, meaning
|C_β(L)| ≥ α|P| for some α,β ∈ (0,1]. This captures that some proportion
of outputs can be obtained by prompting.
-/

/-- The coverage assumption: the β-control region contains at least α fraction of
    what the upper bound allows. This formalizes that the LLM is "expressive enough"
    that some proportion of its potential outputs can be obtained by prompting.
    Specifically: |C_β(L)| ≥ α|P|. -/
def CoverageAssumption {Token : Type*} {K : ℕ} {O : Type*} [Fintype O]
    (L : LLM Token K O) (α β : ℝ) (P : Finset (InputSpace Token K)) : Prop :=
  α * P.card ≤ (ControlRegion L β P).card

/-- **Main Security Theorem**: Consider an LLM L : Σ^K → Σ* and an H&S pipeline L ∘ h_D
    operating on a document of length Len. Assume there are constants α,β ∈ (0,1] such
    that |C_β(L)| ≥ α|P|. Then:

    |C_β(L ∘ h_D)| / |C_β(L)| ≤ K·Len / (αβ · |P|)

    Since |P| = Θ(|Σ|^K) and αβ is a constant, this gives
    |C_β(L ∘ h_D)| / |C_β(L)| = O(K·Len · |Σ|^{-K}),
    showing exponential reduction in the attacker's control when |Σ| > 1. -/
theorem hs_security_theorem {Token : Type*} {K : ℕ} {O : Type*} [Fintype O]
    (L : LLM Token K O) (α β : ℝ) (Len : ℕ)
    (hα_pos : 0 < α) (hβ_pos : 0 < β)
    (P_highlight : Finset (InputSpace Token K))
    (P_full : Finset (InputSpace Token K))
    (h_hs_bound : P_highlight.card ≤ K * Len)
    (h_coverage : CoverageAssumption L α β P_full)
    (hP_full_pos : 0 < P_full.card) :
    ((ControlRegion L β P_highlight).card : ℝ) / (ControlRegion L β P_full).card
      ≤ (K * Len : ℕ) / (α * β * P_full.card) := by
  -- First, bound |C_β(L, P_highlight)| ≤ K·Len/β using the control region bound
  have h_hs_bound' : ((ControlRegion L β P_highlight).card : ℝ) ≤ (K * Len : ℕ) / β := by
    have h1 : ((ControlRegion L β P_highlight).card : ℝ) ≤ P_highlight.card / β :=
      controlRegion_card_bound L β hβ_pos P_highlight
    have h2 : (P_highlight.card : ℝ) ≤ (K * Len : ℕ) := Nat.cast_le.mpr h_hs_bound
    calc ((ControlRegion L β P_highlight).card : ℝ)
        ≤ P_highlight.card / β := h1
      _ ≤ (K * Len : ℕ) / β := div_le_div_of_nonneg_right h2 (le_of_lt hβ_pos)
  -- From coverage assumption: |C_β(L, P_full)| ≥ α|P_full|
  have h_coverage' : α * P_full.card ≤ (ControlRegion L β P_full).card := h_coverage
  have hαP_pos : (0 : ℝ) < α * P_full.card := mul_pos hα_pos (Nat.cast_pos.mpr hP_full_pos)
  -- |C_hs|/|C_full| ≤ |C_hs|/(α|P_full|) ≤ (K·Len/β)/(α|P_full|) = K·Len/(αβ|P_full|)
  calc ((ControlRegion L β P_highlight).card : ℝ) / (ControlRegion L β P_full).card
      ≤ (ControlRegion L β P_highlight).card / (α * P_full.card) := by
        apply div_le_div_of_nonneg_left (Nat.cast_nonneg _) hαP_pos h_coverage'
    _ ≤ ((K * Len : ℕ) / β) / (α * P_full.card) := by
        apply div_le_div_of_nonneg_right h_hs_bound' (le_of_lt hαP_pos)
    _ = (K * Len : ℕ) / (β * (α * P_full.card)) := by rw [div_div]
    _ = (K * Len : ℕ) / (α * β * P_full.card) := by congr 1; ring
