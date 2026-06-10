# `spmd_types`

A type system for distributed (SPMD) tensor computations in PyTorch.  This
package provides a type checker for two type systems:

* **Local SPMD types**, which allow you to use Megatron-style differentiable
  collectives in a safe way by tracking whether or not your backward gradients
  are pending reduction or not.  The type checker verifies your code computes
  correct gradients.

* **Global SPMD types**, a DTensor-like abstraction for writing code that has
  the same semantics whether run on a single device or in a distributed
  fashion, but with explicit communication operations so you are never
  guessing when a redistribute occurs.  The type checker verifies your code
  gives equivalent results across different parallelizations.

Importantly, this type checking process can be performed entirely locally,
without having to actually run a full E2E distributed training run to check
for loss matching.  Our goal is to typecheck realistic training code with
minimal changes.  We have used local SPMD types to typecheck a realistic
pretraining codebase, and global SPMD types is actively under construction!

This package also comes with differentiable collectives that already have
correct typing rules; however, you are welcome to bring your own collectives
(as long as you specify correct typing rules for them!)

## Installation

```bash
pip install spmd_types
```

## User docs

```{toctree}
:maxdepth: 2

key_concepts
local_spmd_types
global_spmd_types
api/index
```

## Developer docs

```{toctree}
:maxdepth: 1

design
```
