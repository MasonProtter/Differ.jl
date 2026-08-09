# Forward-over-reverse composition: overrides DifferForwards' coupling-point hooks
# (`_is_foreign_mode_carrier`, `_foreign_mode_primal_ir`, `_foreign_selfsim_shadow_type`,
# `_foreign_selfsim_mirror_field`) and DifferReverse's (`_forward_entry_name`,
# `_is_foreign_forward_carrier`, `_foreign_has_hand_frule`) so that differentiating a
# `rev_gradient`/`value_and_gradient!`/`Tape`-pullback call under forward mode works. Loaded only
# when both `DifferForwards` and `DifferReverse` are present (weak-dep extension).
#
# Every name used here is brought in via `import` (never a bare `using`), including ones only
# ever *called*, not extended: mixing `import X: name` with a separate bare `using X` on the same
# name is what breaks method-extension detection (see the migration progress notes, stage 3/4 —
# "must be explicitly imported to be extended" vs. the worse, silent case where a name never
# brought into scope any other way becomes a brand-new disconnected local binding instead of an
# error). `primal`/`tangent` are deliberately imported from `DifferForwards` only, never from
# `DifferReverse` too — they're two different generic functions with the same name (one for
# `Dual`, one for `CoDual`), and this file (rules_ad_runtime.jl below) only ever needs the
# `Dual` one; importing both would make the bare name ambiguous.
module DifferForwardsOverReverseExt

import DifferForwards: DifferForwards, Dual, primal, tangent, frule!!, has_hand_frule,
    _is_foreign_mode_carrier, _foreign_mode_primal_ir,
    _foreign_selfsim_shadow_type, _foreign_selfsim_mirror_field,
    NoTangent, tangent_type, build_tangent

import DifferReverse: DifferReverse, Stack, SingletonStack, CommsCell, Tape,
    is_reverse_fwds_impl, is_reverse_pullback_impl,
    optimized_reverse_fwds_ir, optimized_reverse_pullback_ir, build_reverse_interp,
    _forward_entry_name, _is_foreign_forward_carrier, _foreign_has_hand_frule,
    __pop_blk_stack!, _bulk_save!, _bulk_restore!, _inner_ctx, _inner_self_ctx, _alloc_tape

using Core: MethodInstance

# ---------------------------------------------------------------------------
# Coupling point 1 (forward_interp.jl): a `primal_mi` reached while dualizing turns out to be one
# of DifferReverse's own carriers (`reverse_fwds_impl`/`reverse_pullback_impl`) — reached when
# forward-differentiating a `rev_gradient`/`value_and_gradient!`/`Tape` pullback call. Fetch the
# real reverse-mode optimized IR via a nested `DifferReverse` interpreter instead of trying to
# re-infer the throwing carrier stub, which only `DifferReverse`'s own interpreter recognizes.
# `nested_forward=true` makes that nested interpreter's own `src_inlining_policy` also protect
# forward hand rules (`_foreign_has_hand_frule` below), so `Stack`'s `push!`/`pop!` (which only
# has a forward hand rule, defined further down) survives as a call for this dualizer to see.
# ---------------------------------------------------------------------------

DifferForwards._is_foreign_mode_carrier(mi::MethodInstance) =
    is_reverse_fwds_impl(mi) || is_reverse_pullback_impl(mi)

function DifferForwards._foreign_mode_primal_ir(interp, primal_mi::MethodInstance,
                                                reason::Ref{String}, edges::Vector{Any})
    rinterp = build_reverse_interp(; world=interp.world, nested_forward=true)
    if is_reverse_fwds_impl(primal_mi)
        return optimized_reverse_fwds_ir(rinterp, primal_mi, reason, edges)
    else # is_reverse_pullback_impl(primal_mi), the only other way _is_foreign_mode_carrier is true
        return optimized_reverse_pullback_ir(rinterp, primal_mi, reason, edges)
    end
end

# ---------------------------------------------------------------------------
# Coupling points 2-3 (forward_interp.jl's `%new` arm, builtins.jl's `getfield`/`setfield!`
# arms): DifferReverse's self-similar-shadow bookkeeping types (`Stack`/`SingletonStack`/
# `CommsCell`/`Tape`) — reached under forward-over-reverse when the reverse-mode carrier's own
# constructor/field-access calls get inlined into raw `%new`/`getfield`/`setfield!` before the
# dualizer ever sees them as calls. `_foreign_selfsim_shadow_field` needs no override (its default
# body in DifferForwards is already fully generic — see that package's own comment on it).
# ---------------------------------------------------------------------------

DifferForwards._foreign_selfsim_shadow_type(@nospecialize(T::Type)) =
    T <: Stack || T <: SingletonStack || T <: CommsCell || T <: Tape

# `Stack.position` has no tangent (`Int`, tangent type `NoTangent`), but it's still lockstep
# bookkeeping that must stay identical between primal and shadow tape — not differentiable
# content, just an index. Mirroring the write with the **primal** value (rather than leaving the
# shadow field untouched) is what keeps a recycled shadow tape (forward-over-reverse's
# `Ctx{<:Tape}` case) from retaining a stale position from a previous call.
DifferForwards._foreign_selfsim_mirror_field(@nospecialize(T::Type), fi::Int) =
    T <: Stack && fieldname(T, fi) === :position

# ---------------------------------------------------------------------------
# Coupling point 5 (reverse_interp.jl's `src_inlining_policy`): when a `DifferReverse` interpreter
# was built on behalf of an outer forward-mode dualization (`nested_forward=true`, set by
# `_foreign_mode_primal_ir` above), also protect a call with a hand `frule!!` from being inlined
# away by the reverse-mode optimizer — `has_hand_frule` is DifferForwards' own predicate, callable
# directly since its signature is mode-agnostic.
# ---------------------------------------------------------------------------

DifferReverse._foreign_has_hand_frule(interp, mi::MethodInstance) = has_hand_frule(interp, mi)

# ---------------------------------------------------------------------------
# Coupling point 6 (reverse_interp.jl's `has_hand_reverse_rule`/`reverse_fwds_recursive_ci`): a
# `Dual` (forward-mode carrier) reaching reverse-mode dispatch is reverse-over-forward, which is
# unsupported — reject cleanly rather than crashing inside `fcodual_type`. Also names the forward
# entry point (`frule!!`/`dualized_impl`) in the resulting bail message for
# `_composition_bail_message`.
# ---------------------------------------------------------------------------

# Both overrides type their argument as `::Type` (the default hooks in `reverse_interp.jl` leave
# it untyped) so this is a genuinely more-specific *added* method, not a same-signature overwrite
# across modules — see the "deliberately untyped" note on `_is_foreign_mode_carrier`
# (`forward_interp.jl`) for why that distinction matters for package extensions.
DifferReverse._is_foreign_forward_carrier(@nospecialize(P::Type)) = P <: Dual

DifferReverse._forward_entry_name(@nospecialize(ftype::Type)) =
    ftype === typeof(frule!!)          ? "frule!!" :
    ftype === typeof(DifferForwards.dualized_impl) ? "dualized_impl" : nothing

# ---------------------------------------------------------------------------
# Coupling point 4: hand-written `frule!!` for the reverse-mode runtime's own primitives —
# `rules_ad_runtime.jl`, moved wholesale from the original monolith (it always needed exactly
# `DifferForwards` + `DifferReverse`, both present once this extension loads). See that file for
# the full rationale; only the surrounding `using`/`import` scoping is new to the split.
# ---------------------------------------------------------------------------

include("rules_ad_runtime.jl")

end # module DifferForwardsOverReverseExt
