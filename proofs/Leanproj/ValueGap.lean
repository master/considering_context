import Mathlib

open Finset

namespace ValueGap

variable {n : ℕ} [NeZero n]

structure IsStoch (p : Fin n → ℝ) : Prop where
  nonneg  : ∀ j, 0 ≤ p j
  sum_one : ∑ j, p j = 1

noncomputable def tv (p q : Fin n → ℝ) : ℝ := (1 / 2) * ∑ j, |p j - q j|

/-- **Value gap from marginal averaging** -/
theorem value_gap
    (γ Rmax ε : ℝ) (r Vθ Vbar : Fin n → ℝ) (Pθ Pbar : Fin n → Fin n → ℝ)
    (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) (hR : 0 ≤ Rmax)
    (hStθ : ∀ i, IsStoch (Pθ i)) (hStbar : ∀ i, IsStoch (Pbar i))
    (hBθ   : ∀ i, Vθ i   = r i + γ * ∑ j, Pθ i j   * Vθ j)
    (hBbar : ∀ i, Vbar i = r i + γ * ∑ j, Pbar i j * Vbar j)
    (hVbar : ∀ i, 0 ≤ Vbar i ∧ Vbar i ≤ Rmax / (1 - γ))
    (hε    : ∀ i, tv (Pθ i) (Pbar i) ≤ ε) :
    ∀ i, |Vθ i - Vbar i| ≤ γ * Rmax / (1 - γ) ^ 2 * ε := by
  have hpos : 0 < 1 - γ := by linarith
  have hne  : (1 - γ) ≠ 0 := ne_of_gt hpos
  set c := Rmax / (2 * (1 - γ)) with hcdef
  have hcnn : 0 ≤ c := by rw [hcdef]; positivity
  have h2c : Rmax / (1 - γ) = 2 * c := by rw [hcdef]; field_simp; ring
  -- pick a state maximizing the value gap
  obtain ⟨i₀, hi₀⟩ := Finite.exists_max (fun i => |Vθ i - Vbar i|)
  set D := |Vθ i₀ - Vbar i₀| with hDdef
  -- (1) telescoping identity, rewards cancel
  have hdecomp : ∀ i, Vθ i - Vbar i
      = γ * (∑ j, Pθ i j * (Vθ j - Vbar j))
      + γ * (∑ j, (Pθ i j - Pbar i j) * Vbar j) := by
    intro i
    rw [hBθ i, hBbar i]
    have e1 : ∑ j, Pθ i j * (Vθ j - Vbar j)
        = (∑ j, Pθ i j * Vθ j) - ∑ j, Pθ i j * Vbar j := by
      rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
    have e2 : ∑ j, (Pθ i j - Pbar i j) * Vbar j
        = (∑ j, Pθ i j * Vbar j) - ∑ j, Pbar i j * Vbar j := by
      rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
    rw [e1, e2]; ring
  -- (2) transition term is a sup-norm non-expansion
  have hT1 : ∀ i, |∑ j, Pθ i j * (Vθ j - Vbar j)| ≤ D := by
    intro i
    calc |∑ j, Pθ i j * (Vθ j - Vbar j)|
        ≤ ∑ j, |Pθ i j * (Vθ j - Vbar j)|      := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, Pθ i j * |Vθ j - Vbar j|        := Finset.sum_congr rfl (fun j _ => by
            rw [abs_mul, abs_of_nonneg ((hStθ i).nonneg j)])
      _ ≤ ∑ j, Pθ i j * D                      := Finset.sum_le_sum (fun j _ =>
            mul_le_mul_of_nonneg_left (hi₀ j) ((hStθ i).nonneg j))
      _ = (∑ j, Pθ i j) * D                    := by rw [Finset.sum_mul]
      _ = D                                    := by rw [(hStθ i).sum_one, one_mul]
  -- (3) mass-zero ⇒ recenter by the midrange c
  have hmass : ∀ i, ∑ j, (Pθ i j - Pbar i j) = 0 := fun i => by
    rw [Finset.sum_sub_distrib, (hStθ i).sum_one, (hStbar i).sum_one, sub_self]
  have hrecenter : ∀ i, ∑ j, (Pθ i j - Pbar i j) * Vbar j
      = ∑ j, (Pθ i j - Pbar i j) * (Vbar j - c) := fun i => by
    have h : ∑ j, (Pθ i j - Pbar i j) * (Vbar j - c)
        = ∑ j, ((Pθ i j - Pbar i j) * Vbar j - (Pθ i j - Pbar i j) * c) :=
      Finset.sum_congr rfl (fun j _ => by ring)
    rw [h, Finset.sum_sub_distrib, ← Finset.sum_mul, hmass i, zero_mul, sub_zero]
  have hcbnd : ∀ j, |Vbar j - c| ≤ c := by
    intro j; obtain ⟨hlo, hhi⟩ := hVbar j
    have hhi' : Vbar j ≤ 2 * c := by rw [← h2c]; exact hhi
    rw [abs_le]; constructor <;> linarith
  -- (4) Hölder + ‖·‖₁ = 2·tv, then Pinsker bound ε
  have hT2 : ∀ i, |∑ j, (Pθ i j - Pbar i j) * Vbar j| ≤ Rmax / (1 - γ) * ε := by
    intro i
    rw [hrecenter i]
    calc |∑ j, (Pθ i j - Pbar i j) * (Vbar j - c)|
        ≤ ∑ j, |(Pθ i j - Pbar i j) * (Vbar j - c)| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, |Pθ i j - Pbar i j| * |Vbar j - c|   := Finset.sum_congr rfl (fun j _ => by
            rw [abs_mul])
      _ ≤ ∑ j, |Pθ i j - Pbar i j| * c              := Finset.sum_le_sum (fun j _ =>
            mul_le_mul_of_nonneg_left (hcbnd j) (abs_nonneg _))
      _ = (∑ j, |Pθ i j - Pbar i j|) * c            := by rw [Finset.sum_mul]
      _ = (2 * tv (Pθ i) (Pbar i)) * c              := by simp only [tv]; ring
      _ ≤ (2 * ε) * c                               := by gcongr; exact hε i
      _ = Rmax / (1 - γ) * ε                        := by rw [h2c]; ring
  -- (5) per-state bound, then close the contraction at the maximizer
  have hstep : ∀ i, |Vθ i - Vbar i| ≤ γ * D + γ * (Rmax / (1 - γ) * ε) := by
    intro i
    rw [hdecomp i]
    calc |γ * (∑ j, Pθ i j * (Vθ j - Vbar j)) + γ * (∑ j, (Pθ i j - Pbar i j) * Vbar j)|
        ≤ |γ * (∑ j, Pθ i j * (Vθ j - Vbar j))| + |γ * (∑ j, (Pθ i j - Pbar i j) * Vbar j)| :=
          abs_add _ _
      _ = γ * |∑ j, Pθ i j * (Vθ j - Vbar j)| + γ * |∑ j, (Pθ i j - Pbar i j) * Vbar j| := by
          simp only [abs_mul, abs_of_nonneg hγ0]
      _ ≤ γ * D + γ * (Rmax / (1 - γ) * ε) := by gcongr; exacts [hT1 i, hT2 i]
  have hD : D ≤ γ * Rmax / (1 - γ) ^ 2 * ε := by
    have hs := hstep i₀                         -- D ≤ γ*D + γ*(Rmax/(1-γ)*ε)
    have hDle : D ≤ γ * (Rmax / (1 - γ) * ε) / (1 - γ) := by
      rw [le_div_iff₀ hpos]; nlinarith [hs]     -- `le_div_iff` on older Mathlib
    calc D ≤ γ * (Rmax / (1 - γ) * ε) / (1 - γ) := hDle
      _ = γ * Rmax / (1 - γ) ^ 2 * ε            := by field_simp; ring
  exact fun i => le_trans (hi₀ i) hD

end ValueGap
