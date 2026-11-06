Use {{tool:measurement.evaluate}} to measure the current candidate. With no Shape
selection it uses the evaluator's full workload; select Shape IDs for focused probes.
Wait for each request to finish before editing measured sources. Retain the returned
Kernel and Record IDs. Inspect a saved result with {{tool:measurement.record_read}}
and find a Kernel's earlier measurements with {{tool:measurement.kernel_records}}.
Reuse this evidence when deciding the next experiment. For an unknown execution
outcome, reconcile retained records before submitting another measurement.
