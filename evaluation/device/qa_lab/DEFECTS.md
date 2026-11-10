"""EchoGesture A14 defect log.

Two defects were found during regression expansion.  Neither is fixed here:
production code is outside A14 ownership.  Each entry links a narrow,
self-contained reproduction that exits 0 while the defect is open.

D-1  Predictor reload replaces the module behind a fixed name
-----  ----------------------------------------------------------------
``run_manifest.load_predictor()`` registers every predictor module as
``echogesture_predictor``.  Loading a second predictor replaces the module
object, so a consumer that kept the first ``predict`` function (or its
globals) observes the second module's state.  A harness that loads predictors
more than once — e.g. evaluation followed by a demo reload — can therefore
run predictions through a torn module.  Reproduction:
``reproduce_repeated_load.py``.

D-2  Corrupt clip aborts paired evaluation and leaks work/
-----  ----------------------------------------------------------------
``robustness.paired_evaluate()`` removes ``output_dir/work`` with a bare
``shutil.rmtree`` after the loop.  A clip that fails to decode raises
``ValueError`` out of the loop and the intermediate ``work/`` directory of
encoded corruptions is left on disk; the next run in the same output
directory reuses the same paths.  Reproduction:
``reproduce_paired_cleanup.py``.

Notes (not defects)
-------------------
* ``robustness.paired_evaluate``'s ``failed_decode_count`` key records
  predictor failures on each leg, not decode failures; decode failure of a
  clean input raises instead (see D-2).  The existing tests pin this
  behaviour; it is unchanged.
* ``run_manifest.load_labels`` iterating a plain string yields single-letter
  labels; the vocabulary is still non-empty and uppercase, so this is a
  tolerable input-typing quirk rather than a contract violation.
